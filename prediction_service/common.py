# pylint: disable=C0114, E0401, W0621, W1203
from datetime import datetime, timedelta
from io import BytesIO
import logging
from google.cloud import storage
import yfinance as yf  # type: ignore
from yfinance import shared
import pandas as pd
from config import GCLOUD_SETTINGS  # type: ignore

def download_financial_data(
    tickers: str, days: int, logger: logging.Logger
) -> pd.DataFrame | None:
    """Function to dowload data from yahoo finance"""

    logger.info(f"Started downloading data for tickers {tickers} for {days} days")

    try:
        today = datetime.now()
        start_date = today - timedelta(days=days)

        data: pd.DataFrame = yf.download(tickers, start=start_date, group_by="ticker")

        if not shared._ERRORS:
            logger.info(f"Data for tickers {tickers} successfully downloaded")
            return data
        else:
            logger.error(f"Data download failed {shared._ERRORS}")
            return None
    except Exception as e:
        logger.error(f"Data download failed {e}")
        return None

def dowload_saved_data(
    storage_client: storage.Client,
    bucket_name: str,
    ticker: str,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Download previously saved ticker data"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{ticker}.csv")
        byte_data = blob.download_as_bytes()
        return pd.read_csv(BytesIO(byte_data), index_col="Date", parse_dates=True)
    except Exception as e:
        logger.error(f"Error downloading saved data for ticker {ticker} {e}")
        return None

def upload_to_bucket(
    storage_client: storage.Client,
    bucket_name: str,
    df_string: str,
    destination_blob_name: str,
    logger: logging.Logger,
):
    """Upload to GCP bucket"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(df_string)

        logger.info(f"Dataframe uploaded to {destination_blob_name}.")
    except Exception as e:
        logger.error(f"Upload failed {e}")

def download_data(ticker: str) -> pd.DataFrame:
    """Download ticker data from bucket"""
    storage_client = storage.Client(GCLOUD_SETTINGS["project"])
    bucket = storage_client.bucket(GCLOUD_SETTINGS["bucket"])
    blob = bucket.blob(f"{ticker}.csv")
    byte_data = blob.download_as_bytes()
    return pd.read_csv(BytesIO(byte_data), index_col="Date", parse_dates=True)

