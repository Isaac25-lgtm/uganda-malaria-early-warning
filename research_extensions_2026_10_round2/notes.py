"""Technical revision pointers, without altering or ghostwriting the dissertation."""
import pandas as pd
from research_extensions_2026_10_round2.common import *

def main():
    prior=ROOT/'research_extensions_2026_10/results';old=pd.read_csv(prior/'legacy_recomputed_metrics.csv');new=pd.read_csv(prior/'master_metrics.csv');changes=[]
    for model in ['rf','xgb']:
        for h in HORIZONS:
            before=old[(old.model==model)&(old.horizon==h)].iloc[0];after=new[new.experiment.eq(f'legacy12_{model}_h{h}_start2020_all')].iloc[0]
            for metric in (['wape','f2','precision','sensitivity'] if model=='rf' else ['wape']):
                changes.append({'section':'Chapter 4, Table 4.5/4.8; Chapter 6 conclusions','pdf_pages':'51 (printed 36), 53 (printed 38), 70-71 (printed 55-56)','model':model,'horizon':h,'metric':metric,'submitted_value':before[metric],'corrected_value':after[metric],'reason':'Count: corrected inner tuning; warning: prediction plus explicit 52/53-week threshold-policy change. See decomposed first-round table.','source_artifact':'research_extensions_2026_10/results/legacy_corrected_comparison.csv','source_sha256':sha(prior/'legacy_corrected_comparison.csv'),'status':'Verified numerical evidence; student/supervisor choose dissertation amendment/version labels'})
    atomic_csv(pd.DataFrame(changes),RESULTS/'corrections_for_dissertation.csv')
    rows=pd.DataFrame(changes);counts=rows[rows.metric.eq('wape')]
    lines=['# Current corrected baseline','',
        'The first-round corrected baseline is preserved; round-two availability scenarios are separate scientific specifications. The recorded Git HEAD is `13fbb2e6941e61a2fe59407530a5b85150790b9b`; source corrections remain uncommitted as found at round-two start. No new application correction, commit, tag, push or model promotion occurs in this round.','',
        '| Model | Weeks | Original WAPE % | Corrected WAPE % |','| --- | --- | --- | --- |']
    for r in counts.itertuples():lines.append(f'| {r.model} | {r.horizon} | {r.submitted_value:.4f} | {r.corrected_value:.4f} |')
    lines+=['','Source: first-round `legacy_recomputed_metrics.csv`, `master_metrics.csv` and decomposed `legacy_corrected_comparison.csv`. Original RF count metrics are unchanged at 4/12 weeks to reporting precision; eight-week corrected tuning changes its count result. Warning changes also include the explicitly versioned 52/53-week reference policy. Original 12w XGBoost has the lower corrected WAPE point estimate; its paired interval versus RF includes zero.','',
        'This is a selected-source freeze and SHA manifest, not a full disk backup. `baseline_manifest.json` hashes current source and first-round artifacts; `baseline_source/` copies selected source. Trained models, raw surveillance and large rasters are not recopied. The initial thesis freeze in the first-round folder remains separate. New source, environment and model execution controls are in this folder.']
    (OUT/'CURRENT_CORRECTED_BASELINE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'THESIS_UPDATE_MATRIX.md').write_text('''# Thesis update matrix

Page numbers below explicitly distinguish PDF viewer pages from printed thesis pages. These are review instructions/evidence pointers, not replacement academic prose. The original PDF remains unchanged.

| Location | Required revision decision | Evidence/status |
| --- | --- | --- |
| Abstract and Chapter 6 | Align headline performance and selected-model claims with the supervisor-approved corrected version; explain warnings/false alerts | First-round corrected counts; this round's availability sensitivities must not silently replace them |
| Chapter 3 tuning and availability methods | Explain inner label embargo, case/label/reference cutoffs, preprocessing, exact estimator thread count and assumed source availability | `inner_boundary_audit.csv`, `worked_timelines.csv`, new source code and tuning JSON |
| PDF p51 / printed p36, Tables 4.5-4.6 | RF 8w WAPE changes 38.5616 to 39.7440; XGBoost 12w changes 40.7035 to 39.2143. Recalculate relative baseline reductions and other selected count metrics consistently | `corrections_for_dissertation.csv`; original count table and corrected first-round results. The original seasonal benchmark is preserved separately |
| PDF p52 / printed p37, model selection discussion | Separate count-WAPE point selection from warning-F2 selection; qualify constrained nonseasonal SARIMAX/fallback and unequal coverage | First-round strict common-sample table; new seasonal feasibility subset is supplementary |
| PDF p53 / printed p38, Table 4.8 and following paragraph | Correct affected warning confusion counts and metrics consistently. The statement RF was selected in all six positions no longer holds for corrected count-WAPE point ranking | RF retains higher F2; 12w XGBoost lower WAPE point estimate has zero-including paired intervals |
| PDF p54-55 / printed p39-40, Tables 4.9-4.10 and section 4.7.1 | Distinguish frozen adapted ARIMAX rescore from standardized refitting. Corrected 8w point ordering reverses; intervals include zero | First-round literature comparisons; source/convergence audit and small diagnostic rerun here. Original 0.33% relative vs 0.129 percentage-point distinction was already correct |
| PDF p70-71 / printed p55-56, Chapter 6 | Remove universal-winner implications; qualify climate evidence, prototype utility, baseline strength and reporting-vintage limitations | Climate support is exploratory; equal-information results and uncertainty here; 2026 retrospective warning limitations remain |
| Limitations and recommendations | Explain seasonal-reference fallback/support, district heterogeneity, dependency, information parity, uncertain issue times and new-experiment selection | Reference diagnostics, block-length sensitivity and matched comparisons; no calibrated outbreak probability or operational benefit established |
| Signatures/objectives/spacing/poster appendix | Manual authoritative-document actions with supervisor/template confirmation | First-round S01-S20 matrix retained; no automatic thesis/PDF rewrite |

Revised results need consistent table/figure/caption version labels throughout. Student writes/reviews final academic text. Confirmed incorrect results must be corrected; student and supervisor decide how to present the correction account, with submitted values retained only as provenance. Pending REC/MoH status and any use of 2026 results require the actual institutional decision; local tool access is not institutional approval.
''',encoding='utf-8')
    (OUT/'CONCEPTUAL_TIMELINE_AND_LEAKAGE_AUDIT.md').write_text('''# Conceptual timeline and leakage audit

`results/worked_timelines.csv` provides concrete 4/8/12-week examples for origin week 22 December 2025. Dates label Monday week starts. They are not timestamps proving a report or forecast existed on that Monday. A complete origin-week case total cannot be known at the start of that same week. The current-week-count scenario is conditional on later issuance after that total becomes available, which reduces elapsed operational lead time; it is not evidence for a Monday-issued four-week-ahead service.

The old inner split withheld the latest target dates but retained other labels occurring after the earliest validation feature origin. At h8, that includes up to seven weeks of labels beyond the earliest inner origin. The corrected original split stops labels at the earliest inner origin. This round additionally subtracts the assumed case delay for both inner and outer labels. The target remains the original week, so a stricter delay tests a harder information contract.

Delay 0 adds the observed current-week case count to the original historical features. Delay 1 ends case inputs/labels/reference at O-1. Delay 2 shifts the historical lag and rolling inputs another week and ends labels/reference at O-2. Each row's interpolation right anchor must be available by its allowed case cutoff. Training-only medians/missing indicators and district encoders are refitted per outer fit. Weather retains origin-relative lags 4/8/12. Environmental product observation dates do not establish their historical publication times; later reanalysis vintage is retained and disclosed.

Every model/horizon/delay uses the same three original parameter candidates and earliest original tuning fold, with MAE and deterministic candidate-order ties. Parameters are then fixed. Differences between delay scenarios can include new inputs, less recent labels, different reference labels and different selected parameters. They do not isolate one causal effect of information delay. Count error and warning-label changes are separated in interpretation.

All new 2020-2025 comparisons are exploratory development evidence. Actual case-report arrival/revision timestamps and issue-time climate vintages are unavailable. No genuine prospective test or operational source-availability proof is claimed. No new surveillance extraction or data after June 2026 is used.

The fresh XGBoost replay demonstrates thread-dependent refitting in the installed version: recorded six-thread outer fits reproduce saved predictions; a one-thread refit differs. The seven-week inner-boundary problem and numerical execution-thread problem are distinct. Retain exact threads, package versions, seed, candidate order, row order, source signatures and CSV serialization precision for reproduction. Existing saved models/results were not changed.
''',encoding='utf-8')
    (OUT/'SUPERVISOR_S01_S20_EVIDENCE_RESPONSE.md').write_text('''# Supervisor evidence response, round two

The authoritative twenty-item mapping remains `../research_extensions_2026_10/supervisor_response_matrix.md`. It was checked against the supplied supervisor PDF; this round does not renumber or invent new supervisor requirements.

| Item(s) | Additional evidence in this round | Remaining ownership |
| --- | --- | --- |
| S01, S13, S16 | Equal-information confusion counts/F1/F2/workload; predefined objective point-choice ledger | Student explains denominators and supervisor accepts reporting version; costs are hypothetical |
| S08, S14 | Equal-weight district distributions, burden strata, worst-decile examples, missingness associations | No incidence adjustment or causal explanation; region metadata unavailable |
| S09 | Availability-aware feature/label/reference contracts and worked dates; fresh tuning audit | Explain raw vs transformed features and information timing |
| S10 | Seasonal reference observations/years, fallback, prespecified sufficiency/abstention coverage | Threshold labels remain study-defined; no independent outbreak register |
| S11, S15, S17 | Adapted ARIMAX source/fitting audit, small fresh subset and annual seasonal feasibility study | Distinguish subset versus national evidence, failed fits and unequal tuning effort |
| S12, S19 | Existing 92-week evidence retained; block-length sensitivity added; strict case-information comparison | All inspected development data; new protected full-climate validation absent |
| S18 | First-round grouped permutation retained, with its correlation/spatial/constant-time limitations | No claim that a new correlation-aware conditional importance method was run |
| S02-S07 | First-round manual editorial/structure/poster actions unchanged | Student/supervisor author and approve revised document |
| S20 | Prior software checks remain engineering evidence; no end-user study added | Intended health-user validation still required for utility claims |
''',encoding='utf-8')
    (OUT/'PUBLICATION_READINESS.md').write_text('''# Publication readiness

Proposed primary question: **What incremental surge-warning value do climate inputs and modelling choices provide beyond surveillance history under explicit information and workload constraints?** Surge detection is primary; case-count WAPE is a separate secondary objective and may use different models, histories and fitting objectives. Model complexity, history length and warning stability support that question. The present round strengthens baseline fairness and diagnostics; it does not supply new untouched full-climate validation.

Corrected national district-week evaluation, transparent missingness/threshold rules and a reproducible prototype are defensible contributions. Prior Ugandan forecasting and integrated early-warning systems preclude an unqualified first-in-Uganda claim. Ten primary-source roles/adaptation limits are recorded in `LITERATURE_METHOD_FIDELITY.csv`; that focused registry is not a systematic review.

Before a paper: agree the primary estimand and corrected reporting version, document actual REC/MoH decisions and data-use/publication clearance, establish permissible source/code sharing, freeze selection before a genuinely uninspected later evaluation and resolve operational availability assumptions. An honest exploratory retrospective paper may still be possible, subject to journal/supervisor judgement; a new protocol cannot restore protection to already inspected January-June 2026 outcomes.

Vegetation/soil moisture, a comprehensive seasonal SARIMAX study, true rolling training windows, calibrated outbreak probabilities, spatial/hierarchical models and intended-user utility were not added to this bounded round. Their absence is explicit; they are not mandatory just to increase algorithm count. The new annual SARIMA subset is a feasibility result, not a universal seasonal-model verdict.

No manuscript submission, external data sharing, production model promotion, Git commit/push or authoritative dissertation rewrite has been performed. Final academic prose and institutional decisions remain with the student and supervisor.
''',encoding='utf-8')
    print('Revision and interpretation notes written.',flush=True)

if __name__=='__main__':main()
