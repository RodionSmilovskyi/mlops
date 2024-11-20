# pylint: disable=C0114, E0401, W0621, W1203
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fastapi.responses import JSONResponse
import pandas as pd
from typing import cast
from fastapi import FastAPI
from google.cloud import storage  # type: ignore

from config import DAYS, GCLOUD_SETTINGS, LOCAL_PATH, TICKERS, TRAINING_EPOCHS, WINDOW
from common import (
    dowload_saved_data,
    download_data,
    download_financial_data,
    upload_to_bucket,
)
from classifier import load_stock_classifier_state, predict, prepare_dataset, prepare_prediction_dataset, train

app = FastAPI(title="Stock precictor")
storage_client = storage.Client(GCLOUD_SETTINGS["project"])

from google.cloud import logging as cloud_logging
cloud_logging_client = cloud_logging.Client(project=GCLOUD_SETTINGS["project"])
cloud_logging_client.setup_logging()

import logging as logger

@app.get("/update-model")
def update_model():
    """Load new data and provide additional training to model"""

    logger.info(f"Started downloading data for tickers {TICKERS}")

    data = download_financial_data(TICKERS, DAYS, logger)

    if data is not None:
        for t in TICKERS:
            ticker_name = t.strip().lower()
            new_ticker_df = cast(pd.DataFrame, data[ticker_name.upper()])

            saved_ticker_df = dowload_saved_data(
                storage_client, GCLOUD_SETTINGS["bucket"], ticker_name, logger
            )

            if saved_ticker_df is not None and saved_ticker_df.empty is False:
                merged_df = saved_ticker_df.combine_first(new_ticker_df)
                upload_to_bucket(
                    storage_client,
                    GCLOUD_SETTINGS["bucket"],
                    merged_df.to_csv(),
                    f"{ticker_name}.csv",
                    logger,
                )
            else:
                upload_to_bucket(
                    storage_client,
                    GCLOUD_SETTINGS["bucket"],
                    new_ticker_df.to_csv(),
                    f"{ticker_name}.csv",
                    logger,
                )

    logger.info(f"Finished downloading data for tickers {TICKERS}")
    
    datasets = []
    start_date_str = "2016-01-01"
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    
    for t in TICKERS:
        ticker_name = t.strip().lower()
        df = download_data(ticker_name)
        training_df = prepare_dataset(df, start_date_str, end_date_str)
        datasets.append(training_df)
    
    df = pd.concat(datasets, axis = 0)
    model, optimizer, loss, is_new = load_stock_classifier_state(WINDOW)

    train(model, loss, optimizer, df, WINDOW, 1000 if is_new is True else TRAINING_EPOCHS)
    
    return {"status": "ok"}


@app.get("/get-predictions")
def get_predictions():
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    month_ago = datetime.now() - relativedelta(months=1)
    start_date_str = month_ago.strftime("%Y-%m-%d")
    model, optimizer, loss, is_new = load_stock_classifier_state(WINDOW)
    predictions = []
    
    for t in TICKERS:
        ticker_name = t.strip().lower()
        df = download_data(ticker_name)
        prepared_df = prepare_prediction_dataset(df, start_date_str, end_date_str).tail(1)
        date_time = prepared_df.index[0].date()
        action = predict(model, prepared_df)
        predictions.append({
            "action": action,
            "ticker": t,
            "date": date_time.strftime("%Y-%m-%d")
        })
        
    return JSONResponse(predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
