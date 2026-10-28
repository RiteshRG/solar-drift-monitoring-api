GCP_PROJECT_ID = "solar-drift-monitoring-2026"

# Google Cloud Storage bucket used for production data and monitoring artifacts.
BUCKET_NAME = "solar-drift-monitoring-2026-868330784277"

# Object path where the drift-monitoring report is stored.
REPORT_FILE = "reports/drift_report.csv"

# Minimum number of production records required to run drift monitoring.
MINIMUM_CURRENT_RECORDS = 50

# Number of newly collected production records required to trigger monitoring.
TRIGGER_SIZE = 50

# Object path that stores the production-record count from the last monitoring run.
STATE_FILE = "monitoring/monitoring_state.json"

# GCS lock prevents overlapping scheduled drift checks.
MONITORING_LOCK_FILE = "monitoring/monitoring.lock"

# Immutable per-run drift reports.
RUN_REPORT_PREFIX = "reports/runs/"

# Cloud Run prediction endpoint used to generate production data.
API_URL = "https://solar-monitoring-api-868330784277.asia-south1.run.app/predict"

# Existing production model loaded by the prediction API.
PRODUCTION_MODEL_PATH = "models/model_pipeline.joblib"

# Approved training/reference dataset used by monitoring and retraining.
TRAINING_DATA_PATH = "data/train.csv"

# Proposed output path for locally trained candidates; never deploy this directly.
CANDIDATE_MODEL_PATH = "models/candidate_model_pipeline.joblib"

# Machine-readable metadata paired with the candidate model artifact.
CANDIDATE_MANIFEST_PATH = "models/candidate_model_pipeline.manifest.json"

# Reproducible candidate training and evaluation configuration.
MODEL_EVALUATION_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42
MODEL_MINIMUM_R2 = 0.80