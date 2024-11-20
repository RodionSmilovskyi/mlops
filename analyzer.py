# pylint: disable=C0114, E0401, W0621
import argparse
import os
from io import BytesIO
import logging
from typing import Final, TypedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from google.cloud import storage  # type: ignore
from prediction_service.common import download_data
from prediction_service.config import GCLOUD_SETTINGS, LOCAL_PATH, WINDOW

class AnalyzerConfig(TypedDict):
    """Settings for stock analysis"""
    ticker: str

def plot_chart_raw(df: pd.DataFrame, column:str) -> Figure:
    """Data before prepocessing"""
    fig, ax = plt.subplots(figsize=(12,6))
    df = df.resample("1d").mean()
    df = df.dropna()
    y = df[column]
    x = np.arange(len(y))
    
    ax.plot(df.index, y, label=column)
    
    x = df.index.map(pd.Timestamp.toordinal)
    coefs = np.polyfit(x, y, 1)
    trend = np.poly1d(coefs)
    trend_y = trend(x)
    
    ax.plot(df.index, trend_y, color='green', label='Linear Trend')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.legend()
    
    return fig

def plot_autocorrelation(df: pd.DataFrame, column: str) -> Figure:
    fig = plot_acf(df[column])
    
    return fig

def plot_processing(df: pd.DataFrame, column: str) -> Figure:
    fig, ax = plt.subplots(3 ,1, figsize=(8,6))
    df = df.resample("1d").mean()
    df = df.dropna()
    y = df[column].values
    x = np.arange(len(y))
    
    ax[0].set_title('Original data in processing window with trend line')
    x_trend = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_trend, y)
    trend = model.predict(x_trend)
    ax[0].plot(x, y, label=column)   
    ax[0].plot(x, trend, color='green', label='Linear Trend')
    ax[0].set_xticks(x)
    
    ax[1].set_title('Detrended data')
    y_detrended = y - trend
    ax[1].plot(x, y_detrended, label=column)
    ax[1].set_xticks(x)
    
    ax[2].set_title('Scaled data')
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_detrended.reshape(-1,1))
    ax[2].plot(x, y_scaled, label=column)
    ax[2].set_xticks(x)
    
    plt.subplots_adjust(hspace=0.5)

    return fig


def analyze(config: AnalyzerConfig):
    """Function to extract various stat data"""
    df = download_data(config["ticker"]).head(WINDOW)
    fig = plot_processing(df, "Adj Close")
    
    ticker_data_dir = os.path.join(LOCAL_PATH, "data", config["ticker"])
    if not os.path.exists(ticker_data_dir):
        os.makedirs(ticker_data_dir)
    
    fig.savefig(os.path.join(LOCAL_PATH, os.path.join(ticker_data_dir, f'{config["ticker"]}_adj_close_processed.png')))
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str)
    
    args = parser.parse_args()
    config = AnalyzerConfig(ticker=args.ticker)
    
    analyze(config)
    