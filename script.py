# pylint: disable=C0114, E0401, W0621
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Final, TypedDict
import pandas as pd
import yfinance as yf  # type: ignore
from yfinance import shared
from google.cloud import storage  # type: ignore
from google.cloud import logging as cloud_logging

Params = TypedDict("Params", {"bucket": str, "days": int, "local": str})

TICKERS: Final[list[str]] = ["AMAT", "QCOM"]
LOCAL_PATH: Final[str] = os.path.dirname(os.path.abspath(__file__))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


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
        
        data: pd.DataFrame = yf.download(
            ticker_str, start=start_date
        )

        if not shared._ERRORS:
            logger.info(f"Data for tickers {ticker_str} successfully downloaded")
            return data
        else:
            logger.error(f"Data download failed {shared._ERRORS}")
            return None
    except Exception as e:
        logger.error(f"Data download failed {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument(
        "--local", type=str, default=os.path.dirname(os.path.abspath(__file__))
    )
    parser.add_argument("--days", type=int, default=1)

    args = parser.parse_args()

    logger = logging.getLogger("mlops_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    local_path = os.path.join(args.local, "data.csv")

    data = download_financial_data(TICKERS, args.days, logger)

    if data is not None:
        data.to_csv(local_path)

    if data is not None and args.project is not None and args.bucket is not None:
        storage_client = storage.Client(args.project)
        cloud_logging_client = cloud_logging.Client(project=args.project)
        cloud_handler = cloud_logging_client.get_default_handler()
        logger.addHandler(cloud_handler)

        upload_to_bucket(storage_client, args.bucket, local_path, "data.csv", logger)
