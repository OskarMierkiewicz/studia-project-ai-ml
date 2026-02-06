#!/bin/sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:?MLFLOW_DEFAULT_ARTIFACT_ROOT is required}"

echo "Starting MLflow server..."
echo "MLFLOW_BACKEND_STORE_URI=$MLFLOW_BACKEND_STORE_URI"
echo "MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT"

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT"