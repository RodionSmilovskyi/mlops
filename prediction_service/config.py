# pylint: disable=C0114, E0401, W0621
import os
from typing import TypedDict, Final


class GCloudSettings(TypedDict):
    """Defines setting for gcloud"""
    project: str
    bucket: str


GCLOUD_SETTINGS: Final[GCloudSettings] = GCloudSettings(
    project="mlops-439321", bucket="rodions-mlops"
)


class PredictorConfig(TypedDict):
    """Settings for stock analysis"""

    ticker: str
    start_date: str
    end_date: str

WINDOW: Final[int] = 6
TRAINING_EPOCHS: Final[int] = 300
DAYS: Final[int] = 3000
TICKERS: Final[list[str]] = ["AMAT", "QCOM", "CSCO"]
LOCAL_PATH: Final[str] = os.path.dirname(os.path.abspath(__file__))
