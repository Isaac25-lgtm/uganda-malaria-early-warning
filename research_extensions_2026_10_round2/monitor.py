"""Read-only progress observer; never launches or modifies experiments."""
import json,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from research_extensions_2026_10_round2.safe_io import read_json
OUT=Path(__file__).resolve().parent
def read(path):
    try:return read_json(path)
    except (FileNotFoundError,json.JSONDecodeError):return {}
def main():
    started=time.monotonic()
    while time.monotonic()-started<10*3600:
        # Completed checkpoints are immutable. Do not hold legacy mutable status
        # snapshots open while their writers attempt a Windows replacement.
        checkpoints=list((OUT/'checkpoints').glob('*.json'))
        state={'at_utc':datetime.now(timezone.utc).isoformat(),'tree_checkpoints':len(checkpoints),'expected':479,'horizons':{h:{'completed':sum(f'_h{h}_' in p.name for p in checkpoints),'planned':159 if h==12 else 160} for h in [4,8,12]},'classical':dict(Counter(read(p).get('status','writing') for p in (OUT/'classical_jobs').glob('*.json'))),'ridge_horizons_saved':sum((OUT/f'results/direct_ridge_h{h}_predictions.csv').exists() for h in [4,8,12]),'surge_checkpoints':len(list((OUT/'surge_checkpoints').glob('*.json'))),'surge_expected':153,'numerical_verification_exists':(OUT/'results/verification.json').exists(),'handoff_exists':(OUT/'HANDOFF.md').exists(),'final_complete':(OUT/'HANDOFF.md').exists()}
        print(json.dumps(state),flush=True)
        if state['final_complete']:return
        time.sleep(45)
if __name__=='__main__':main()
