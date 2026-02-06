import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from pandas.tseries.offsets import BDay
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import load_prices_csv, build_features
from src.utils import (
    time_split,
    assert_no_nan,
    ensure_datetime_index,
    upload_df_as_csv_to_s3,
    replace_inf_with_nan,
)


def get_target_ticker(df_prices: pd.DataFrame) -> str:
    t = os.getenv("TARGET_TICKER", "AAPL").strip().upper()
    if t not in df_prices.columns:
        raise ValueError(f"Brak tickera '{t}' w danych. Dostępne: {list(df_prices.columns)}")
    return t


def make_xy(
    features_df: pd.DataFrame,
    target_ticker: str,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    y = ret_TARGET.shift(-horizon_days)  (jutrzejszy / za H dni zwrot)
    X = wszystkie ret_* BEZ ret_TARGET (żeby nie robić autoregresji na skróty)
    """
    target_col = f"ret_{target_ticker}"
    if target_col not in features_df.columns:
        raise ValueError(f"Brak kolumny {target_col} w features_df")

    y = features_df[target_col].shift(-horizon_days)

    feature_cols = [c for c in features_df.columns if c.startswith("ret_") and c != target_col]
    X = features_df[feature_cols]

    # wyrównanie (ostatnie H dni mają y=NaN)
    X = X.iloc[:-horizon_days].copy()
    y = y.iloc[:-horizon_days].copy()

    assert X.index.equals(y.index)
    return X, y.astype(float), feature_cols


def main():
    if os.getenv("DRY_RUN", "0") == "1":
        print("DRY_RUN=1 -> OK (importy), kończę bez treningu i bez MLflow.")
        return

    data_path = os.getenv("DATA_PATH", "/data/Prices.csv")

    horizon_max = int(os.getenv("HORIZON_DAYS_MAX", "1"))  # możesz zostawić 1, albo trenować H=1..N
    include_monthly = os.getenv("INCLUDE_MONTHLY", "0") == "1"

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "sp500_return_regression")

    clip_low = float(os.getenv("CLIP_LOW", "-0.2"))
    clip_high = float(os.getenv("CLIP_HIGH", "0.2"))

    train_end = os.getenv("TRAIN_END", "2021-12-31")
    val_end = os.getenv("VAL_END", "2023-12-31")

    pred_bucket = os.getenv("PREDICTIONS_BUCKET", "predictions")
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if not s3_endpoint:
        raise ValueError("Brak MLFLOW_S3_ENDPOINT_URL (endpoint MinIO).")

    min_samples = int(os.getenv("MIN_SAMPLES_PER_SPLIT", "200"))

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    df_prices = load_prices_csv(data_path)
    ensure_datetime_index(df_prices, "df_prices")
    df_prices = df_prices.sort_index()

    def get_target_tickers(df_prices: pd.DataFrame) -> list[str]:
        env = os.getenv("TARGET_TICKER", "AAPL").strip().upper()
        if env == "ALL":
            return list(df_prices.columns)
        if env not in df_prices.columns:
            raise ValueError(f"Brak tickera '{env}' w danych. Dostępne: {list(df_prices.columns)}")
        return [env]

    target_tickers = get_target_tickers(df_prices)

    features_df = build_features(
        df_prices,
        include_monthly=include_monthly,
        clip_returns=True,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    features_df = replace_inf_with_nan(features_df).dropna()
    assert_no_nan(features_df, "features_df")

    last_asof = features_df.index.max()
    print(f"TARGETS={target_tickers}, H=1..{horizon_max}, asof={last_asof.date()}, include_monthly={include_monthly}")

    for target_ticker in target_tickers:
        for h in range(1, horizon_max + 1):
            X, y, feature_cols = make_xy(features_df, target_ticker, horizon_days=h)

            data = X.join(y.rename("y"), how="inner")
            data = replace_inf_with_nan(data).dropna()
            assert_no_nan(data, f"data[{target_ticker}][H{h}]")

            train_df, val_df, test_df = time_split(data, train_end=train_end, val_end=val_end)

            if len(train_df) < min_samples or len(val_df) < min_samples or len(test_df) < min_samples:
                print(
                    f"⚠️ Skip {target_ticker} H{h}: za mało danych "
                    f"(train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)})"
                )
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["y"].astype(float)

            X_val = val_df[feature_cols]
            y_val = val_df["y"].astype(float)

            X_test = test_df[feature_cols]
            y_test = test_df["y"].astype(float)

            model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )

            with mlflow.start_run(run_name=f"{target_ticker}_H{h}_reg") as run:
                run_id = run.info.run_id
                created_at = datetime.now(timezone.utc).isoformat()

                mlflow.log_param("target_ticker", target_ticker)
                mlflow.log_param("horizon_days", h)
                mlflow.log_param("train_end", train_end)
                mlflow.log_param("val_end", val_end)
                mlflow.log_param("clip_low", clip_low)
                mlflow.log_param("clip_high", clip_high)
                mlflow.log_param("include_monthly", include_monthly)
                mlflow.log_param("model", "HistGradientBoostingRegressor")
                mlflow.log_param("features", "ret_only_excluding_target_ret")

                model.fit(X_train, y_train)

                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

                val_mae = mean_absolute_error(y_val, val_pred)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
                test_mae = mean_absolute_error(y_test, test_pred)
                test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

                mlflow.log_metric("val_mae", val_mae)
                mlflow.log_metric("val_rmse", val_rmse)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("test_rmse", test_rmse)

                mlflow.sklearn.log_model(model, artifact_path="model")

                pred_test_df = pd.DataFrame({
                    "asof_date": test_df.index.astype(str),
                    "target_ticker": target_ticker,
                    "horizon_days": h,
                    "y_true": y_test.values,
                    "y_pred": test_pred,
                    "run_id": run_id,
                    "created_at": created_at,
                    "split": "test",
                })
                test_key = f"{target_ticker}/H{h}/predictions_test.csv"
                upload_df_as_csv_to_s3(pred_test_df, pred_bucket, test_key, s3_endpoint)

                x_last = X.iloc[[-1]]
                pred_last = float(model.predict(x_last)[0])

                forecast_for = (last_asof + BDay(h)).date().isoformat()
                forecast_df = pd.DataFrame([{
                    "asof_date": last_asof.date().isoformat(),
                    "forecast_for_date": forecast_for,
                    "target_ticker": target_ticker,
                    "horizon_days": h,
                    "pred_ret_next": pred_last,
                    "pred_sign": int(pred_last > 0),
                    "run_id": run_id,
                    "created_at": created_at,
                    "split": "forecast_latest",
                }])
                forecast_key = f"{target_ticker}/H{h}/forecast_latest.csv"
                upload_df_as_csv_to_s3(forecast_df, pred_bucket, forecast_key, s3_endpoint)

                print(
                    f"✅ {target_ticker} H{h} MAE/RMSE "
                    f"val={val_mae:.5f}/{val_rmse:.5f} test={test_mae:.5f}/{test_rmse:.5f}"
                )
                print(f"✅ uploaded: {test_key} + {forecast_key}")

    print("✅ All done.")


if __name__ == "__main__":
    main()