"""Local review evidence bundle, excluding secrets, raw extracts and model binaries."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from research_extensions_2026_10.run_study import OUT,RESULTS,write_json


def main():
    final=json.loads((RESULTS/'finalization.json').read_text())
    assert final['complete'] and final['passed']
    numerical=json.loads((RESULTS/'result_verification.json').read_text())
    assert numerical['passed'] and numerical['full_scenario_completion_required']
    root=OUT.parent
    tracked=['.gitignore','README.md','app/services/case_data_cleaning.py','app/services/explainability.py',
        'app/services/model_validation.py','app/services/modelling_dataset.py','app/services/models/baseline.py',
        'app/services/models/pooled_tree_validation.py','app/services/models/sarimax.py','app/services/report_data_service.py',
        'app/services/risk_classification.py','app/templates/dashboard/index.html',
        'app/templates/lifecycle/model_evaluation.html','docs/explainability_api.md']
    patch=subprocess.run(['git','diff','--binary','--',*tracked],cwd=root,capture_output=True,check=True).stdout
    (OUT/'corrections.patch').write_bytes(patch)
    files=[OUT/'corrections.patch',root/'docs/HANDOVER.md',root/'README.md']
    files.extend(p for p in OUT.glob('*.md') if p.name!='HANDOFF_DRAFT.md')
    files.extend(p for p in OUT.glob('*.py') if p.name!='run_study_initial_threads2.py')
    files.extend(OUT.glob('environment*lock.txt'))
    files.extend([root/'tests'/name for name in ['test_research_corrections.py','test_research_statistics.py','test_holdout_checkpoints.py']])
    files.extend((OUT/'figures').glob('*.png'));files.extend((OUT/'figures').glob('*.pdf'))
    files.append(OUT/'presentation/poster_evidence_draft_A1.pdf')
    csv_names=['master_metrics','per_origin_metrics','district_metric_distribution','legacy_corrected_comparison',
        'legacy_recomputed_metrics','paired_comparisons','warning_workload_cost_sensitivity','metric_uncertainty',
        'threshold_sensitivity','four_model_common_sample','four_model_cost_sensitivity','frozen_adapted_methods_rescored',
        'corrected_rf_adapted_literature_comparisons','missingness_district_year','missingness_year','missingness_position_audit',
        'reporting_completeness_performance_sensitivity','2026_locked_metrics','2026_locked_paired_comparisons',
        '2026_historical_prototype_audit_metrics','weekly_simple_baseline_metrics','weekly_rf_simple_baseline_paired',
        'grouped_permutation_summary','grouped_permutation_per_origin']
    files.extend(RESULTS/(name+'.csv') for name in csv_names)
    json_names=['result_verification','legacy_table_reproduction','2026_model_reconstruction_audit','2020_live_source_audit',
        '2025_final_weeks_live_audit','raw_ingestion_provenance','sarimax_coverage_reasons','visual_review_record',
        'pdf_quality_checks','finalization','analysis_specification','weekly_simple_baseline_interpretation']
    files.extend(RESULTS/(name+'.json') for name in json_names)
    files.extend([RESULTS/'verification/verification.json',RESULTS/'verification/verification_targeted.json',RESULTS/'live/http_checks.json',OUT/'baseline/manifest.json',OUT/'baseline/supporting_inputs/manifest.json'])
    files=sorted(set(files));assert all(p.exists() for p in files)
    records=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');destination=OUT/f'review_evidence_bundle_{stamp}.zip'
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Local student/supervisor review, not public-release clearance or a full project backup.',
        'excludes':['.env and credentials','raw surveillance and private response payloads','row-level prediction CSVs','trained model binaries','large rasters'],
        'start_here':'research_extensions_2026_10/HANDOFF.md','files':records}
    write_json(OUT/'review_bundle_manifest.json',manifest)
    with ZipFile(destination,'w',compression=ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in files:bundle.write(path,path.relative_to(root).as_posix())
        bundle.write(OUT/'review_bundle_manifest.json','review_bundle_manifest.json')
        bundle.writestr('START_HERE.txt','Open research_extensions_2026_10/HANDOFF.md. This is a local evidence/review bundle, not a full data backup or authorization to publish. Raw restricted extracts, credentials and model binaries stay in the private workspace. The recorded environment, original repository and authorized data are required to reproduce the runs.\n')
    with ZipFile(destination) as bundle:
        assert bundle.testzip() is None
        for entry in records:assert hashlib.sha256(bundle.read(entry['path'])).hexdigest()==entry['sha256']
    result={'path':str(destination),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'verified_entries':len(records)}
    write_json(OUT/'review_bundle_receipt.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
