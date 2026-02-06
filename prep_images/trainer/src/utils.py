import os
import pandas as pd
import numpy as np
import boto3


def time_split(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_datetime_index(df, "df")
    df = df.sort_index()

    train = df.loc[:train_end].copy()
    val = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1) : val_end].copy()
    test = df.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1) :].copy()
    return train, val, test


def assert_no_nan(df: pd.DataFrame, name: str = "df"):
    n = int(df.isna().sum().sum())
    if n > 0:
        raise ValueError(f"{name} zawiera NaN (liczba: {n}).")


def ensure_datetime_index(df: pd.DataFrame, name: str = "df"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index musi być DatetimeIndex.")


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def _s3_client(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)


def upload_df_as_csv_to_s3(df: pd.DataFrame, bucket: str, key: str, endpoint_url: str):
    s3 = _s3_client(endpoint_url)
    _ensure_bucket(s3, bucket)
    body = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")