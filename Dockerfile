FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG MODEL_ARTIFACT=models/model_pipeline.joblib
RUN test -f "${MODEL_ARTIFACT}" && \
    if [ "${MODEL_ARTIFACT}" != "models/model_pipeline.joblib" ]; then \
        cp "${MODEL_ARTIFACT}" models/model_pipeline.joblib && \
        rm -f "${MODEL_ARTIFACT}" "${MODEL_ARTIFACT}.manifest.json"; \
    fi

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]