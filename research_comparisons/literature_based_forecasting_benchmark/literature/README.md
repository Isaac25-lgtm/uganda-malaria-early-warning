# Literature gate

The per-run literature verification is stored with each run under:

```
runs/<RUN_ID>/literature/
  paper_verification.md              per-source verification (citation, DOI, method, adaptations)
  paper_to_code_specification.json   structured spec consumed by the challenger code
  adaptation_register.csv            each adaptation vs the source + reason + leakage impact
  excluded_methods.md                Ippolito / EPIDEMIA / Katwesige-MLP exclusion rationale
```

For run `20260828`, see `../runs/20260828/literature/`.

No model is fitted until the gate passes (`src/run.py --validate-gate`). Where a
methodological element could not be confirmed in an authoritative accessible
source, it is recorded as "not specified / not verified in accessible source" —
never invented.
