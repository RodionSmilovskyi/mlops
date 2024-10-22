# pylint: disable=C0114, E0401, W0621
import flask
import functions_framework
import logging
from datetime import datetime, timedelta
from typing import Final
import pandas as pd
import yfinance as yf  # type: ignore
from yfinance import shared
from google.cloud import storage  # type: ignore

TICKERS: Final[list[str]] = ["AMAT", "QCOM"]


def upload_to_bucket(
    storage_client: storage.Client,
    bucket_name: str,
    source_file_name: str,
    destination_blob_name: str,
    logger: logging.Logger,
):
    """Upload to GCP bucket"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        logger.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        logger.error(f"Upload failed {e}")


def download_financial_data(
    tickers: list[str], days: int, logger: logging.Logger
) -> pd.DataFrame | None:
    """Function to dowload data from yahoo finance"""

    ticker_str: str = " ".join(TICKERS)
    logger.info(f"Started downloading data for tickers {ticker_str} for {days} days")

    try:
        today = datetime.now()
        start_date = today - timedelta(days=days)

        data: pd.DataFrame = yf.download(ticker_str, start=start_date)

        if not shared._ERRORS:
            logger.info(f"Data for tickers {ticker_str} successfully downloaded")
            return data
        else:
            logger.error(f"Data download failed {shared._ERRORS}")
            return None
    except Exception as e:
        logger.error(f"Data download failed {e}")
        return None


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


@functions_framework.http
def upload(request: flask.Request) -> flask.typing.ResponseReturnValue:
    """Upload financial data"""
    logger = logging.getLogger("mlops_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    days = request.args.get("days", 1, type=int)
    project = request.args.get("project", type=str)
    bucket = request.args.get("bucket", type=str)
    
    if project is None:
        return ("project is not provided", 500)
    
    if bucket is None:
        return ("bucket is not provided", 500)
    
    data = download_financial_data(TICKERS, days, logger)

    if data is not None:
        data.to_csv("data.csv")
        
        storage_client = storage.Client(project)
        upload_to_bucket(
            storage_client, bucket, "data.csv", "data.csv", logger
        )

    return ("Data updated successfully", 200)
