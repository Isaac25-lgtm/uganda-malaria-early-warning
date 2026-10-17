"""Full corrected core SARIMAX run, isolated from production."""
from pathlib import Path
import json
from threadpoolctl import threadpool_limits
from app.services.models.sarimax import run_sarimax_validation

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/sarimax'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        result=run_sarimax_validation(
            dataset_path=ROOT/'baseline/data/modelling/modelling_dataset_2020_2025.csv',
            artifact_path=OUT/'summary.json',predictions_path=OUT/'predictions.csv',
            district_metrics_path=OUT/'district_metrics.csv',diagnostics_path=OUT/'diagnostics.json',
            model_card_path=OUT/'model_card.json',checkpoint_dir=ROOT/'experiments/sarimax_checkpoints',
            record_pipeline_run=False,
            progress_callback=lambda payload:print(json.dumps(payload),flush=True))
    print(json.dumps({'status':result.status,'runtime_seconds':result.runtime_seconds,'job_counts':result.job_counts}),flush=True)


if __name__=='__main__':main()
