"""Week 3: log an evaluation run to MLflow (params + metrics + artifacts).

Backend is settings.mlflow_tracking_uri — default local sqlite (works with `mlflow ui`);
set MLFLOW_TRACKING_URI=http://localhost:5000 to use the docker postgres-backed server.
"""
from pathlib import Path

from app.config import settings


def log_to_mlflow(report: dict, artifacts=()) -> str:
    """Log one eval run; returns the MLflow run id. Called best-effort by the runner."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(report.get("params", {}))
        # MLflow metrics must be numeric — drop the None/NaN ones (e.g. a skipped judge).
        metrics = {
            k: v for k, v in report.get("metrics", {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(metrics)
        for k in ("n_rows", "n_answerable", "n_out_of_scope"):
            if isinstance(report.get(k), int):
                mlflow.log_metric(k, report[k])
        mlflow.set_tags({"git_commit": report.get("git_commit", "unknown")})
        for a in artifacts:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
        return run.info.run_id
