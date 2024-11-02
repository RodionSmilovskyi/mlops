# pylint: disable=C0114, E0401, W0621
from io import BytesIO
import logging
from datetime import datetime, timedelta
from typing import Final, cast
import flask
import functions_framework
import pandas as pd
import yfinance as yf  # type: ignore
from yfinance import shared
from google.cloud import storage  # type: ignore

TICKERS: Final[list[str]] = ["AMAT", "QCOM"]


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
    tickers = request.args.get("tickers", type=str, default="")

    if project is None:
        return ("project is not provided", 500)

    if bucket is None:
        return ("bucket is not provided", 500)

    if len(tickers) == 0:
        return ("tickers are not provided", 500)

    data = download_financial_data(tickers, days, logger)

    if data is not None:
        storage_client = storage.Client(project)

        for t in tickers.split(","):
            ticker_name = t.strip().lower()
            new_ticker_df = cast(pd.DataFrame, data[ticker_name.upper()]).round(3)
            
            saved_ticker_df = dowload_saved_data(
                storage_client, bucket, ticker_name, logger
            )

            if saved_ticker_df is not None and saved_ticker_df.empty is False:
                merged_df = saved_ticker_df.combine_first(new_ticker_df)
                upload_to_bucket(
                    storage_client,
                    bucket,
                    merged_df.to_csv(),
                    f"{ticker_name}.csv",
                    logger,
                )
            else:
                upload_to_bucket(
                    storage_client,
                    bucket,
                    new_ticker_df.to_csv(),
                    f"{ticker_name}.csv",
                    logger,
                )

        return ("Data updated successfully", 200)
    else:
        return ("Data update error", 500)
