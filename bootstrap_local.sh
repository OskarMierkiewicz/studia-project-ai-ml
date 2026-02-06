#!/usr/bin/env bash
set -euo pipefail

INFRA_NS="mlops-infra"
APPS_NS="mlops-apps"

MINIO_RELEASE="minio"
MINIO_ROOT_USER="minio"
MINIO_ROOT_PASSWORD="minio12345"
MINIO_PVC_SIZE="5Gi"

MLFLOW_IMAGE="mlflow-local:latest"
API_IMAGE="api-go-local:latest"
TRAINER_IMAGE="trainer-local:latest"

MLFLOW_DIR="./prep_images/mlflow"
API_DIR="./prep_images/api-go"
TRAINER_DIR="./prep_images/trainer"

log() { echo -e "\n\033[1;34m[bootstrap]\033[0m $*"; }
die() { echo -e "\n\033[1;31m[error]\033[0m $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Brak komendy: $1"; }

need_cmd docker
need_cmd kubectl
need_cmd helm

log "Sprawdzam dostęp do klastra Kubernetes..."
if ! kubectl cluster-info >/dev/null 2>&1; then
  die "Nie mam dostępu do klastra K8s. Włącz Kubernetes w Docker Desktop i spróbuj ponownie."
fi

# Build images
log "Buduję obraz MLflow: $MLFLOW_IMAGE"
docker build -t "$MLFLOW_IMAGE" "$MLFLOW_DIR"

log "Buduję obraz API Go: $API_IMAGE"
docker build -t "$API_IMAGE" "$API_DIR"

log "Buduję obraz Trainer: $TRAINER_IMAGE"
docker build -t "$TRAINER_IMAGE" "$TRAINER_DIR"

# Smoke tests
log "Smoke test: MLflow image (mlflow --version)"
docker run --rm --entrypoint mlflow "$MLFLOW_IMAGE" --version >/dev/null

log "Smoke test: Trainer image (python imports)"
docker run --rm --entrypoint python "$TRAINER_IMAGE" -c \
"import pandas, numpy, sklearn, mlflow, boto3; from src.features import load_prices_csv, build_features; from src.utils import time_split; print('ok')"


log "Smoke test: API image (binary starts)"
cid=$(docker run -d -e MINIO_ENDPOINT=example:9000 -e MINIO_ACCESS_KEY=x -e MINIO_SECRET_KEY=y -e MINIO_BUCKET=b -e PREDICTIONS_KEY=k "$API_IMAGE")
sleep 1
docker rm -f "$cid" >/dev/null

log "✅ Obrazy zbudowane i przetestowane."

minikube image load mlflow-local:latest
minikube image load trainer-local:latest
minikube image load api-go-local:latest

# Namespaces
log "Tworzę namespace’y (jeśli nie istnieją): $INFRA_NS, $APPS_NS"
kubectl create ns "$INFRA_NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl create ns "$APPS_NS"  --dry-run=client -o yaml | kubectl apply -f -

# MinIO via Helm
log "Instaluję/aktualizuję MinIO przez Helm"
helm repo add minio https://charts.min.io/ >/dev/null 2>&1 || true
helm repo update >/dev/null

helm upgrade --install "$MINIO_RELEASE" minio/minio \
  -n "$INFRA_NS" \
  --set mode=standalone \
  --set replicas=1 \
  --set rootUser="$MINIO_ROOT_USER" \
  --set rootPassword="$MINIO_ROOT_PASSWORD" \
  --set persistence.enabled=true \
  --set persistence.size="$MINIO_PVC_SIZE" \
  --set resources.requests.memory=256Mi \
  --set resources.requests.cpu=100m \
  --set service.type=ClusterIP \
  --set consoleService.type=ClusterIP

log "Czekam aż MinIO wstanie..."
kubectl -n "$INFRA_NS" wait --for=condition=available deploy/"$MINIO_RELEASE" --timeout=180s

# Create buckets (Job)
log "Tworzenie bucketów: mlflow, predictions"
kubectl -n "$INFRA_NS" port-forward svc/minio 9000:9000 >/dev/null 2>&1 & #port forward in background
PF_PID=$!
sleep 2

cleanup_pf() {
  kill "$PF_PID" >/dev/null 2>&1 || true
}
trap cleanup_pf EXIT

if command -v mc >/dev/null 2>&1; then
  log "Wykryto lokalne 'mc' -> tworzę buckety automatycznie"
  mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
  mc mb --ignore-existing local/mlflow >/dev/null
  mc mb --ignore-existing local/predictions >/dev/null
  log "✅ Buckety utworzone."
else
  log "Nie mam lokalnego 'mc'. Utwórz buckety ręcznie w MinIO Console:"
  log "1) Otwórz konsolę: kubectl -n $INFRA_NS port-forward svc/minio-console 9001:9001"
  log "2) Zaloguj się: $MINIO_ROOT_USER / $MINIO_ROOT_PASSWORD"
  log "3) Utwórz buckety: mlflow i predictions"
fi

# Apply infra/apps via Kustomize
log "Wdrażam INFRA przez kustomize: k8s/infra"
kubectl apply -k k8s/infra

log "Czekam na rollout postgres i mlflow..."
kubectl -n "$INFRA_NS" rollout status deploy/postgres --timeout=180s
kubectl -n "$INFRA_NS" rollout status deploy/mlflow --timeout=180s

log "Wdrażam APPS przez kustomize: k8s/apps"
kubectl apply -k k8s/apps

log "Czekam na rollout api-go..."
kubectl -n "$APPS_NS" rollout status deploy/api-go --timeout=180s

log "✅ Gotowe."

cat <<EOF

Następne kroki:
1) (opcjonalnie) port-forward:
   kubectl -n $INFRA_NS port-forward svc/mlflow 5000:5000
   kubectl -n $INFRA_NS port-forward svc/minio-console 9001:9001
   kubectl -n $APPS_NS  port-forward svc/api-go 8080:8080

2) Uruchom trening (Job):
   kubectl -n $APPS_NS delete job trainer --ignore-not-found
   kubectl -n $APPS_NS apply -f k8s/apps/trainer-job.yaml
   kubectl -n $APPS_NS logs -f job/trainer

EOF