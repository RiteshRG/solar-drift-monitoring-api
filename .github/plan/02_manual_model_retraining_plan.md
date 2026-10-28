# Manual Model Retraining Implementation Plan

## Goal and boundaries

Implement a human-started retraining workflow using `data/train.csv`. Automatic
drift monitoring only writes GCS reports; it must not start model training,
quality evaluation, promotion, or deployment.

The current model is a scikit-learn `StandardScaler` and
`GradientBoostingRegressor` pipeline. Inputs are `IRRADIATION`,
`AMBIENT_TEMPERATURE`, `HOUR_SIN`, and `HOUR_COS`; target is `AC_POWER`.
The deployed API loads the model from `PRODUCTION_MODEL_PATH`.

## Step-by-step plan

1. **Confirm data quality and baseline — implemented**
   - Validate `TRAINING_DATA_PATH` for required columns, numeric finite values,
     and missing data.
   - Record the production model hash and evaluate the reproducible baseline.
   - Keep `data/validation.csv` separate: it represents a different plant and
     must not be mixed into training without an explicit decision.
   - Use the specified 80/20 split with seed 42 and report exact sample counts.

2. **Implement explicit local candidate training — implemented**
   - `scripts/train_model.py` runs only when manually invoked and is not called
     by drift monitoring or automatically by CI.
   - It uses the current feature schema and pipeline, checks compatibility
     with the pinned Python/scikit-learn runtime, and writes only
     `CANDIDATE_MODEL_PATH`.
   - The manifest records data and artifact hashes, split identity, row counts,
     code revision, dependency versions, model settings, and metrics.

3. **Implement evaluation and model-contract tests — implemented**
   - `scripts/evaluate_model.py` checks artifact/data/split hashes and evaluates
     candidate and production baseline on identical holdout rows.
   - The mandatory R² gate is strictly greater than 0.80. RMSE and MAE are
     reported; human approval is still required because their policy limits
     are not yet configured.
   - Focused tests cover schema, finite data, reproducible splits, metric
     validation, and the strict R² boundary.

4. **Add GitHub validation and reviewed deployment — implemented, cloud setup pending**
   - `.github/workflows/model-retraining.yml` runs tests/evaluation on
     model-related pushes to `main`.
   - The workflow requires the protected `production` environment, verifies
     the same candidate hash, builds/pushes a versioned image, deploys a
     no-traffic revision, then routes traffic directly without a smoke test.
   - Configure branch protection, production-environment reviewers,
     Workload Identity Federation variables/trust, and least-privilege GCP IAM
     before relying on deployment.

5. **Run candidate training manually**
   - A person investigates the drift report and decides retraining is
     appropriate before running the training command.
   - Review data validation, metadata, metrics, and candidate artifact hash.
   - Reject a candidate that does not pass the gate. Keep production unchanged.

6. **Document and verify boundaries — implemented**
   - README describes local commands and required GitHub/GCP configuration.
   - Unit tests and the pinned container validation confirm drift monitoring
     remains separate from training.

## Expected changes

- Implemented: training/evaluation scripts, model-contract tests, manifest
  handling, PR quality gate, protected deployment workflow, and README guide.
- Still required before production deployment: agree RMSE/MAE acceptance
  policy; configure GitHub branch protection and production reviewers; set up
  Workload Identity Federation and deployment IAM in the chosen Google
  account/project.

## Blocking decisions

The R² threshold is set to `> 0.80`. RMSE/MAE thresholds remain unconfigured,
so a human reviewer must consider them before approving deployment. Production
logs currently lack measured AC-power labels, and the available 469-row
training set limits statistical confidence.
