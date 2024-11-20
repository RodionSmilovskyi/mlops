# pylint: disable=C0114, E0401, W0621
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import mlflow

from config import WINDOW

mlflow.set_tracking_uri(
    "https://mlflow-service-349597430624.northamerica-northeast1.run.app"
)
mlflow.set_experiment("stock-predictor")


class StockClassifier(nn.Module):
    """Basic NN for stock buy/sell action"""

    def __init__(self, num_features: int):
        super(StockClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass"""
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


def save_stock_classifier_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.BCELoss,
    path: str,
    ticker_str: str,
    num_features: int,
):
    """Save model and optimizer state for training and inference"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_state_dict": loss.state_dict(),
        },
        os.path.join(path, f"model_{ticker_str}_{num_features}.pth"),
    )


def load_stock_classifier_state(
    num_features: int,
) -> tuple[StockClassifier, optim.Adam, nn.BCELoss, bool]:
    """Load model and optimizer state"""
    # model = StockClassifier(num_features)
    filter_expr = f"""
        params.units="{num_features}"
    """
    runs = mlflow.search_runs(
        filter_string=filter_expr, order_by=["metrics.accuracy DESC"]
    )

    if len(runs) > 0:
        model_url = f"runs:/{runs["run_id"][0]}/model"
        model = mlflow.pytorch.load_model(model_url)
        print(f"Used model {runs["tags.mlflow.runName"][0]}")
        is_new = False
    else:
        model = StockClassifier(num_features)
        is_new = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCELoss()

    return model, optimizer, loss, is_new


def train(
    model: nn.Module,
    criterion: nn.BCELoss,
    optimizer: optim.Adam,
    df: pd.DataFrame,
    num_features: int,
    n_epochs: int = 100,
):
    """Training loop"""
        
    train_df, test_df = train_test_split(df, test_size=0.3)
    
    x_train = train_df.filter(like="feature_").values
    y_train = train_df["output"].values

    with mlflow.start_run():
        params = {"epochs": n_epochs, "learning_rate": 0.001, "units": num_features}

        mlflow.log_params(params)

        for epoch in range(n_epochs):
            permuted_indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[permuted_indices]
            y_shuffled = y_train[permuted_indices]
            x_tensor = torch.tensor(x_shuffled, dtype=torch.float32)
            y_tensor = torch.tensor(y_shuffled, dtype=torch.float32).unsqueeze(1)

            model.train()

            ouputs = model(x_tensor)
            loss = criterion(ouputs, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
            mlflow.log_metric("loss", f"{loss:2f}", step=epoch)

        report = evaluate(model, test_df)
        mlflow.log_metric("accuracy", report["accuracy"])

        mlflow.pytorch.log_model(model, "model")


def evaluate(model: nn.Module, df: pd.DataFrame) -> dict:
    """Generate classification report"""
    x = df.filter(like="feature_").values
    y = df["output"].values

    x_tensor = torch.tensor(x, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        outputs = model(x_tensor)
        predicted = (outputs >= 0.5).float().numpy()

    report = classification_report(y, predicted, target_names=["Sell", "Buy"], output_dict=True)

    return report

def predict(model: nn.Module, df: pd.DataFrame) -> str:
    x = df.filter(like="feature_").values
    x_tensor = torch.tensor(x, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        output = model(x_tensor).squeeze().item()
        predicted = "Buy" if output >= 0.5 else "Sell"
        
    return predicted

def prepare_prediction_dataset(df: pd.DataFrame, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    pred_df = df.loc[start_date_str:end_date_str][["Adj Close"]]  # type: ignore
    pred_df = pred_df.rename(columns={"Adj Close": "feature_0"})
    for lag in range(1, WINDOW):
        pred_df[f"feature_{lag}"] = pred_df["feature_0"].shift(lag)
        
    pred_df = pred_df.dropna()
    pred_df = pred_df.apply(detrend, axis=1).apply(scale, axis=1)
    
    return pred_df

def prepare_dataset(
    df: pd.DataFrame, start_date_str: str, end_date_str: str
) -> pd.DataFrame:
    """Add features and output"""

    training_df = df.loc[start_date_str:end_date_str][["Adj Close"]]  # type: ignore
    training_df = training_df.rename(columns={"Adj Close": "feature_0"})
    values = training_df["feature_0"].shift(-1) - training_df["feature_0"]
    training_df["next_day_price"] = values.values
    training_df = training_df.dropna()
    training_df["output"] = training_df["next_day_price"].apply(
        lambda x: 1 if x > 0 else 0
    )
    training_df = training_df.drop(["next_day_price"], axis=1)
    training_df = training_df.reset_index(drop=True)

    for lag in range(1, WINDOW):
        training_df[f"feature_{lag}"] = training_df["feature_0"].shift(lag)

    training_df = training_df.dropna()
    training_df = training_df.apply(detrend, axis=1).apply(scale, axis=1)

    return training_df


def detrend(row):
    """Remove trend from price data"""
    feature_values = row.filter(like="feature_")
    feature_array = feature_values.values
    x = np.arange(len(feature_array)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, feature_array)
    trend = model.predict(x)
    feature_array = feature_array - trend
    for i, val in enumerate(feature_array):
        row[f"feature_{i}"] = val

    return row


def scale(row):
    """Scale values to 0..1 range"""
    scaler = MinMaxScaler()
    feature_values = row.filter(like="feature_")
    feature_array = feature_values.values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(feature_array)
    for i, val in enumerate(scaled_values):
        row[f"feature_{i}"] = val

    return row
