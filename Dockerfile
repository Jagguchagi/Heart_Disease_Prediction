FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt

RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app src ./src
COPY --chown=app:app models/heart_disease_pipeline.joblib ./models/heart_disease_pipeline.joblib

USER app
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]