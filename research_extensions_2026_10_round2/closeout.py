"""Six requested closure outputs; reuses verified first-round evidence."""
import json,shutil
import pandas as pd
from research_extensions_2026_10_round2.common import *

def md(frame,columns):
    def value(v):
        if pd.isna(v):return '--'
        return f'{v:.3f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(columns.values())+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(value(r[k]).replace('|','/') for k in columns)+' |' for _,r in frame.iterrows()])

def main():
    v=json.loads((RESULTS/'verification.json').read_text());execution=json.loads((RESULTS/'surge_execution.json').read_text());verdict=json.loads((RESULTS/'SURGE_IMPROVEMENT_VERDICT.json').read_text())
    summary=pd.read_csv(RESULTS/'SURGE_CANDIDATE_METRICS.csv');paired=pd.read_csv(RESULTS/'SURGE_PAIRED_COMPARISONS.csv');choices=pd.DataFrame(verdict['horizons'])
    for column in ['strongest_development_point_candidate','recommendation','verdict']:
        if column not in choices:choices[column]=None
    summary['alerts_per_target_week']=(summary.tp+summary.fp)/summary.target_weeks
    columns={'evaluation':'Frame','horizon':'Weeks','method':'Method','complete':'Complete','n':'Rows','f2':'F2','sensitivity':'Recall','precision':'Precision','fp':'False alerts','fn':'Missed surges','alert_rate':'Alert rate','alerts_per_target_week':'Alerts/target week'}
    original=pd.read_csv(ROOT/'research_extensions_2026_10/results/master_metrics.csv')
    original=original[original.experiment.str.fullmatch(r'legacy12_rf_h(?:4|8|12)_start2020_all')].copy()
    original['horizon']=original.experiment.str.extract(r'_h(\d+)_',expand=False).astype(int);original['reference_frame']='R1 corrected original12 full-climate'
    prior_later=pd.read_csv(ROOT/'research_extensions_2026_10/results/2026_locked_metrics.csv')
    prior_later=prior_later[(prior_later.evaluation=='frozen_history')&(prior_later['start']==2020)&prior_later.group.eq('none')].copy();prior_later['reference_frame']='R1 frozen history original contract'
    references=pd.concat([original,prior_later],ignore_index=True).sort_values(['reference_frame','horizon'])
    atomic_csv(references,RESULTS/'ORIGINAL_CORRECTED_SURGE_REFERENCES.csv')
    text='# Surge improvement verdict\n\nPrimary objective: surge warning. Development ends at the last available 2025 week (22 December); no 2026 outcome trains a model or selects its cutoff. January-June 2026 was already inspected, so all later comparisons remain exploratory retrospective evidence.\n\n'
    text+=md(choices,{'horizon':'Weeks','strongest_development_point_candidate':'Strongest development point candidate','verdict':'Verdict','recommendation':'Recommendation'})+'\n\n'
    if 'always_alert_f2_development' in choices:
        text+='## No-skill reference: alerting every district-week\n\nWith surge prevalence near 31%, issuing an alert for every district-week already scores F2 of about 0.70. Read every F2 below against that reference, together with alert rate and false alerts.\n\n'+md(choices,{'horizon':'Weeks','baseline_f2':'Strict RF count F2','always_alert_f2_development':'Always-alert F2 (development)','candidate_f2':'Best candidate F2','candidate_alert_rate':'Candidate alert rate','baseline_alert_rate':'RF alert rate','candidate_fp':'Candidate false alerts','baseline_fp':'RF false alerts','always_alert_f2_later':'Always-alert F2 (2026)'})+'\n\nThe direct classifiers raise F2 only by alerting on most district-weeks: their F2 is within a few hundredths of the always-alert reference while false alerts roughly triple. The strict RF count warning has F2 below the always-alert reference at every development horizon, so F2 alone cannot justify any of these policies; recall must be read with precision, alert rate and false-alert workload.\n\n'
    text+='## Previously verified corrected-original references\n\n'+md(references,{'reference_frame':'Reference frame','horizon':'Weeks','n':'Rows','f2':'F2','sensitivity':'Recall','precision':'Precision','fp':'False alerts','fn':'Missed surges','alert_rate':'Alert rate'})+'\n\nThese values are copied from the preceding verified round, not rerun. They retain their original information contracts. Do not subtract them from the strict new scenario as a paired improvement estimate; the matched comparisons below use the strict corrected RF approach within each frame.\n\n'
    text+='## Matched warning results\n\n'+md(summary,columns)+'\n\n'
    text+='## Paired uncertainty\n\n'+md(paired[paired.metric.eq('f2')],{'evaluation':'Frame','horizon':'Weeks','comparison':'First minus second','cluster':'Resampling','block_length':'Block weeks','difference_first_minus_second':'F2 difference','ci_lower':'95% lower','ci_upper':'95% upper','paired_rows':'Matched rows'})+'\n\n'
    text+='Full recall/precision intervals are in `results/SURGE_PAIRED_COMPARISONS.csv`. Point candidate selection uses inspected development F2; the intervals do not adjust for this selection or multiple comparisons. Fourteen later target weeks offer limited temporal evidence even with two block-length sensitivities. A full-climate candidate has no matching 2026 climate evaluation.\n\n'
    text+='The original-frame baseline is the availability-corrected RF count-to-Q75 warning approach, using strict O-1 predictors, training labels and reference history. It is distinguished from the original first-round corrected numbers, whose information boundary differed. The later baseline is a new strict history-only, fixed-parameter RF fit frozen in December 2025; the previously reported 2026 surveillance-only results remain preserved separately. Within each new frame methods share exact evaluation target rows and warning references. Training rows can differ where an early historical surge reference is undefined; the compared procedures also differ in their training objective and label construction. Changing frames/features/reference updating does not isolate temporal drift.\n\n'
    text+='Always-alert and never-alert references are in `SURGE_FIRST_COMPARISON.csv`. F2 alone can reward excessive alerts. There is no established acceptable clinical workload ceiling or intended-user effectiveness evidence. A statistical improvement does not establish operational acceptability. No model is promoted to production. If improvement is inconclusive or absent, retain the defensible corrected research baseline and stop searching; no third round begins automatically.\n'
    (OUT/'SURGE_IMPROVEMENT_VERDICT.md').write_text(text,encoding='utf-8')
    (OUT/'FINAL_MODEL_RECOMMENDATION.md').write_text('# Final model recommendation\n\n'+md(choices,{'horizon':'Horizon','recommendation':'Surge-warning recommendation','verdict':'Evidence'})+'\n\nThese recommendations concern the research baseline, not clinical deployment. Do not substitute a WAPE winner for a surge decision. Separate warning/count point comparisons are in `results/SEPARATE_WARNING_AND_COUNT_CHOICES.csv`; they do not compel a two-model service or a universal winner. Count MAE/WAPE and surge F2/recall/precision have different purposes. No further automatic optimization is queued.\n',encoding='utf-8')
    source=ROOT/'research_extensions_2026_10/results';destination=RESULTS/'first_round_evidence';destination.mkdir(exist_ok=True);copied=[]
    names=['master_metrics.csv','paired_comparisons.csv','2026_locked_metrics.csv','2026_locked_paired_comparisons.csv','four_model_common_sample.csv','four_model_cost_sensitivity.csv','threshold_sensitivity.csv','missingness_district_year.csv','missingness_year.csv','missingness_position_audit.csv','corrected_rf_adapted_literature_comparisons.csv','grouped_permutation_summary.csv','per_origin_metrics.csv']
    for name in names:
        original=source/name;target=destination/name;shutil.copy2(original,target);assert sha(original)==sha(target)
        copied.append({'source':str(original),'copy':str(target.relative_to(OUT)),'sha256':sha(original)})
    write_json(RESULTS/'first_round_evidence_manifest.json',{'purpose':'Reused verified evidence; copied without rerunning first-round analysis','files':copied})
    done={
      'S01':'Warning tables give false alerts, all alerts, precision and missed events with denominators.',
      'S02':'Original signatures/PDF preserved; authoritative-document edit remains with student.',
      'S03':'Required single numbered objectives section identified; no document rewrite performed.',
      'S04':'Spacing/layout action identified; authoritative editable thesis not modified.',
      'S05':'Move limitations to Chapter 6 and repair references; editorial action remains.',
      'S06':'Existing Appendix G located at PDF p93; prior evidence poster/figures retained; final template/metric correction remains.',
      'S07':'Surge-first model selection is separated from supporting count-WAPE comparisons.',
      'S08':'Burden-weighted and equal-district errors, burden strata and counts-versus-incidence limitation reported.',
      'S09':'Training-only encoding/imputation, inner label embargo and O-1 feature/reference timing audited.',
      'S10':'Q70/75/80 and seasonal-window grid reused; support, fallback and abstention coverage added.',
      'S11':'Adaptation roles/source fidelity and original ARIMAX implementation/convergence policy audited.',
      'S12':'Per-origin and expanded weekly evidence reused; sparse origins and short inspected later evaluation qualified.',
      'S13':'F1/F2, hypothetical cost ratios and direct-classifier cutoff selection/workload reported separately. Always-alert reference reported: at ~31% surge prevalence, alerting every district-week scores F2 about 0.70, above the strict RF count warning (0.657/0.666/0.679), so F2 must be justified together with precision, alert rate and false alerts.',
      'S14':'Verified missingness/year/district/exclusion audit reused; no unproven cause assigned.',
      'S15':'Original common-sample SARIMAX/fallback evidence reused; new annual feasibility kept separate.',
      'S16':'F2, recall, precision, FP/FN, alert rates and paired uncertainty reported; risk colours remain distinct.',
      'S17':'Relative-percent versus percentage-point distinction checked; corrected ordering and uncertainty reported.',
      'S18':'Grouped held-out permutation and ablation evidence reused with correlation/dependence limitations.',
      'S19':'Existing history/climate ablations reused; bounded independently tuned classifier full/history comparison added.',
      'S20':'Previous software/HTTP evidence retained; no intended-health-user utility study was performed.'}
    evidence={
      'S01':'SURGE_IMPROVEMENT_VERDICT.md; results/strict_weekly_warning_workload.csv',
      'S02':'THESIS_UPDATE_MATRIX.md','S03':'THESIS_UPDATE_MATRIX.md','S04':'THESIS_UPDATE_MATRIX.md','S05':'THESIS_UPDATE_MATRIX.md','S06':'Prior presentation evidence; THESIS_UPDATE_MATRIX.md',
      'S07':'FINAL_MODEL_RECOMMENDATION.md; results/SEPARATE_WARNING_AND_COUNT_CHOICES.csv',
      'S08':'results/district_error_distribution.csv; results/burden_strata.csv',
      'S09':'CONCEPTUAL_TIMELINE_AND_LEAKAGE_AUDIT.md; results/inner_boundary_audit.csv; results/surge_verification.json',
      'S10':'results/first_round_evidence/threshold_sensitivity.csv; results/threshold_reference_summary.csv',
      'S11':'results/LITERATURE_METHOD_FIDELITY.csv; results/adapted_arimax_source_audit.json',
      'S12':'results/first_round_evidence/per_origin_metrics.csv; results/block_length_sensitivity.csv; results/SURGE_PAIRED_COMPARISONS.csv',
      'S13':'results/first_round_evidence/four_model_cost_sensitivity.csv; SURGE_IMPROVEMENT_VERDICT.md',
      'S14':'results/first_round_evidence/missingness_district_year.csv; results/first_round_evidence/missingness_position_audit.csv',
      'S15':'results/first_round_evidence/four_model_common_sample.csv; results/seasonal_subset_metrics.csv',
      'S16':'results/SURGE_CANDIDATE_METRICS.csv; results/SURGE_PAIRED_COMPARISONS.csv',
      'S17':'results/first_round_evidence/corrected_rf_adapted_literature_comparisons.csv',
      'S18':'results/first_round_evidence/grouped_permutation_summary.csv; results/block_length_sensitivity.csv',
      'S19':'results/SURGE_PAIRED_COMPARISONS.csv; results/EXPERIMENT_REGISTRY.csv',
      'S20':'VERIFICATION_REPORT.md; first-round HANDOFF.md'}
    remaining={
      'S02':'Remove old signatories/dates in Word; obtain required signatures after presentation.',
      'S03':'Student consolidates objectives and numbering.',
      'S04':'Student completes final authoritative-document spacing and layout pass.',
      'S05':'Student relocates limitations and checks cross-references.',
      'S06':'Student checks/updates existing Appendix G p93 with the approved template and corrected metrics.',
      'S08':'No population-standardized risk model; counts must be described as reported burden.',
      'S12':'Cannot create new independent temporal evidence from inspected outcomes.',
      'S13':'Clinical costs and acceptable workload have not been established.',
      'S18':'Correlation/spatial dependence limits importance interpretation.',
      'S19':'Exploratory incremental value is not causation or untouched full-climate validation.',
      'S20':'Intended-user usability/utility unvalidated; restrict claims rather than invent a pass.'}
    original=(ROOT/'research_extensions_2026_10/supervisor_response_matrix.md').read_text(encoding='utf-8');matrix=[]
    for line in original.splitlines():
        if not line.startswith('| S'):continue
        cells=[c.strip() for c in line.strip('|').split('|')];item=cells[0]
        matrix.append({'item':item,'requested':cells[1],'done':done[item],'evidence':evidence[item],'fully_resolved':'No: manual thesis action' if item in ['S02','S03','S04','S05','S06'] else 'No: explicit limitation remains' if item in remaining else 'Technical evidence addressed; thesis text pending','remaining':remaining.get(item,'Student incorporates verified correction/evidence into the authoritative dissertation.')})
    assert len(matrix)==20;matrix=pd.DataFrame(matrix);atomic_csv(matrix,RESULTS/'SUPERVISOR_RESPONSE_MATRIX.csv')
    (OUT/'SUPERVISOR_RESPONSE_MATRIX.md').write_text('# Dr. Kakuba: complete twenty-comment response matrix\n\nThis preserves the original S01-S20 mapping. Evidence completion and dissertation editing are distinct; no unperformed usability study or manual edit is marked resolved.\n\n'+md(matrix,{'item':'Item','requested':'Requested','done':'Done','evidence':'Evidence','fully_resolved':'Fully resolved?','remaining':'Remaining limitation/action'})+'\n',encoding='utf-8')
    tree=json.loads((RESULTS/'analysis_completion.json').read_text());seasonal=json.loads((RESULTS/'seasonal_progress.json').read_text());registry=pd.read_csv(RESULTS/'EXPERIMENT_REGISTRY.csv');families=registry.groupby(['family','status'],dropna=False).size().reset_index(name='records')
    (OUT/'SUPPLEMENTARY_EXPERIMENT_REGISTER.md').write_text('# Supplementary experiment register\n\n'+md(families,{'family':'Experiment family','status':'Status','records':'Registry records'})+f'\n\nTree checkpoints: {tree["checkpoint_count"]}/{tree["expected_checkpoints"]}. Bounded surge checkpoints: {execution["completed_checkpoints"]}/153; complete={execution["complete"]}. Seasonal feasibility: {seasonal["status_counts"]}. Candidate/cutoff score records are not additional independent fitted models. Full attempts and reasons are in EXPERIMENT_REGISTRY.csv and SURGE_EXPERIMENT_REGISTER.csv.\n\nFirst-round climate, training-start, threshold, common-sample and data-quality results are reused rather than rerun. New environmental acquisition, new families, broad searches, true rolling windows, intended-user study, manuscript drafting and a third round are deferred. Supplementary evidence supports corrections and supervisor responses; it does not require expanding the main dissertation.\n',encoding='utf-8')
    revision=(OUT/'THESIS_UPDATE_MATRIX.md').read_text(encoding='utf-8')
    (OUT/'DISSERTATION_REVISION_PLAN.md').write_text('# Dissertation revision plan\n\nConfirmed methodological and numerical errors must be corrected consistently; the submitted values are retained only for the correction audit, not as equally valid alternative results. Student/supervisor decide how to present the correction account. No substantive redesign is justified by an inconclusive supplementary result.\n\n'+revision+'\n\n'+(OUT/'EXACT_REVISION_LOCATIONS.md').read_text(encoding='utf-8')+'\n\nKeep the main thesis surge-focused. Add direct classifiers/seasonal feasibility and other extra experiments to concise supplementary tables/appendices unless the verified verdict supports a substantive selection change. Update Abstract, Chapter 3 methods, Chapter 4 results/model-selection and Chapter 6 conclusions together. Use the full twenty-item response matrix for signatures, objective numbering, spacing, limitation placement and poster appendix. The verified exact heading/table/figure/appendix locator above identifies the current PDF positions; update page numbers after editing. The original PDF is preserved, and no student-authored academic prose is replaced by this technical plan.\n',encoding='utf-8')
    (OUT/'VERIFICATION_REPORT.md').write_text('# Verification report\n\n'+f'Numerical verification passed: {v["passed"]}. Preserved-file hashes: {v["preserved_files_checked"]}; production model files: {v["production_model_files_checked"]}; original supplied documents: {v["original_external_documents_checked"]}. Tree checkpoints checked: {v["tree_checkpoints_checked"]}. Scientific tests: {v["scientific_tests"]["total"]}, actual exit {v["scientific_tests"]["actual_exit_code"]}.\n\n'+f'Surge verification: {v["surge_classifier_verification"]["completed_checkpoints_checked"]}/153 saved checkpoints checked; {v["surge_classifier_verification"]["training_label_references_checked"]} historical-origin training references independently reconstructed. Full bounded surge completion: {v["surge_classifier_verification"]["full_defined_completion"]}. Representative RF/XGB classifier refits, classifier arithmetic, source hashes, cutoff selection and 2026 exclusion are checked.\n\n'+'The separate verified JSONs retain exact counts and replay differences. Verification is internal cross-checking with separate arithmetic/date routines and fresh-process refits, not independent-investigator replication. XGBoost thread count is part of the execution contract. Prior 471-test/74-HTTP evidence is retained as earlier engineering evidence; no new dashboard rebuild or live HTTP pass is claimed. Windows progress-I/O failure/recovery and the later interrupted-worker restart are recorded. Source/data originals and production models remain unchanged; new data after June 2026 were not accessed.\n\n'+'Full experiment completion is reported separately from numerical correctness. Constructed Q75 labels, dependent observations, inspected 2026 data, missing arrival/vintage records, pending institutional decisions and unvalidated clinical workload remain limitations. No deployment, manuscript submission or automatic third experiment round occurred.\n',encoding='utf-8')
    fig=OUT/'figures';fig.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    # Top row: F2 against the always-alert no-skill reference; bottom row: alert
    # rate. F2 alone rewards near-universal alerting at ~31% prevalence.
    figure,axes=plt.subplots(2,2,figsize=(11,7.6),sharex=True)
    for column,evaluation,group in [(0,'original12','all'),(1,'later2026','none')]:
        top,bottom=axes[0,column],axes[1,column];base=None
        for name,label in [(f'rf_count_strict_{group}','RF count warning'),(f'rf_classifier_{group}','RF classifier'),(f'xgb_classifier_{group}','XGBoost classifier')]:
            f=summary[(summary.evaluation==evaluation)&summary.method.eq(name)&summary.complete].sort_values('horizon')
            if len(f)!=3:continue
            top.plot(f.horizon,f.f2,'o-',label=label);bottom.plot(f.horizon,100*f.alert_rate,'o-',label=label)
            if name.startswith('rf_count'):base=f
        if base is not None:
            positive=base.tp+base.fn;negative=base.fp+base.tn
            top.plot(base.horizon,5*positive/(5*positive+negative),'k--',label='Always alert (no skill)')
            bottom.plot(base.horizon,[100]*3,'k--',label='Always alert (no skill)')
        top.set_ylabel('Surge F2');top.set_ylim(0,1);top.set_title('12 development targets; full predictors' if evaluation=='original12' else '14 later targets; frozen history-only fits')
        bottom.set_ylabel('District-weeks alerted (%)');bottom.set_ylim(0,105);bottom.set_xticks([4,8,12]);bottom.set_xlabel('Horizon (weeks)')
    axes[0,-1].legend(frameon=False,fontsize=8,loc='lower left');figure.suptitle('Surge warning: F2 must be read with alert rate\nClassifiers approach the always-alert reference; all outcomes previously inspected; columns use different contracts',fontsize=11);figure.tight_layout(rect=[0,0,1,.9]);figure.savefig(fig/'00_surge_comparison.png',dpi=220);plt.close(figure)
    print('All six closure outputs generated; no further search queued.',flush=True)

if __name__=='__main__':main()
