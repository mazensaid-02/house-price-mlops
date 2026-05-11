# ── BASE IMAGE ────────────────────────────────────────────────
FROM python:3.12-slim

# ── METADATA ──────────────────────────────────────────────────
LABEL maintainer="mlops-engineer"
LABEL project="house-price-mlops"

# ── ENVIRONMENT ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# ── WORKING DIRECTORY ─────────────────────────────────────────
WORKDIR /app

# ── INSTALL DEPENDENCIES ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1 && \
    pip install --no-cache-dir \
        pandas==2.2.2 \
        numpy==1.26.4 \
        scikit-learn==1.4.2 \
        xgboost \
        mlflow==2.14.3 \
        bentoml==1.2.2 \
        pydantic==2.6.4 \
        python-multipart==0.0.6

# ── COPY SOURCE CODE ──────────────────────────────────────────
COPY src/ ./src/

# ── EXPOSE PORT ───────────────────────────────────────────────
EXPOSE 3000

# ── START SERVICE ─────────────────────────────────────────────
CMD ["bentoml", "serve", "service:svc", "--working-dir", "src", "--port", "3000", "--host", "0.0.0.0"]
