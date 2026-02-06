FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY start_mlflow.sh /app/start_mlflow.sh
RUN chmod +x /app/start_mlflow.sh

EXPOSE 5000

ENTRYPOINT ["/app/start_mlflow.sh"]