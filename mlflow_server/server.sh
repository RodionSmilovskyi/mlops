#!/bin/bash 

mlflow db upgrade postgresql://mlflow-dbi:mlflow@35.203.123.246/mlflow-db
mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri postgresql://mlflow-dbi:mlflow@35.203.123.246/mlflow-db \
  --artifacts-destination gs://rodions-mlflow/mlruns