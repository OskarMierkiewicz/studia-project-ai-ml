import os
import pandas as pd
import numpy as np


def load_prices_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Brak kolumny '{date_col}' w pliku. Kolumny: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_features(
    df_prices: pd.DataFrame,
    include_monthly: bool = False,
    clip_returns: bool = True,
    clip_low: float = -0.2,
    clip_high: float = 0.2,
) -> pd.DataFrame:
    """
    Produkcyjne cechy:
      - ret_* liczone TYLKO z cen
      - opcjonalnie miesięczne staty (lag1) — ale domyślnie OFF (bo u Ciebie raczej szkodziły)

    Zwraca df z:
      - cenami (opcjonalnie można je potem wyrzucić w train.py)
      - ret_* (zawsze)
      - monthly lag1 (opcjonalnie)
    """
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        raise TypeError("df_prices musi mieć indeks DatetimeIndex.")

    df = df_prices.sort_index().copy()

    # returns
    rets = df.pct_change().add_prefix("ret_")

    out = df.copy()

    if include_monthly:
        monthly = df.resample("ME").agg(["mean", "median", "std"])
        monthly.columns = [f"{t}_{s}" for t, s in monthly.columns]
        monthly.index = monthly.index.to_period("M")
        monthly_lag1 = monthly.shift(1)

        daily = df.copy()
        daily["ym"] = daily.index.to_period("M")
        out = daily.join(monthly_lag1, on="ym").drop(columns=["ym"])

    out = pd.concat([out, rets], axis=1)

    # cleanup
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    if clip_returns:
        ret_cols = [c for c in out.columns if c.startswith("ret_")]
        out.loc[:, ret_cols] = out[ret_cols].clip(clip_low, clip_high)

    return out