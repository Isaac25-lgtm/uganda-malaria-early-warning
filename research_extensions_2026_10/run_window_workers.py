"""Separate horizon workers own disjoint checkpoints; no duplicate writers."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_extensions_2026_10'


def main():
    workers={};handles=[]
    for h in [4,8,12]:
        stdout=open(OUT/f'window_h{h}.stdout.log','ab',buffering=0)
        stderr=open(OUT/f'window_h{h}.stderr.log','ab',buffering=0);handles.extend([stdout,stderr])
        workers[h]=subprocess.Popen([sys.executable,'-u','-m','research_extensions_2026_10.run_study','expanded','--horizon',str(h)],
            cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    while True:
        limits=json.loads((OUT/'compute_limits.json').read_text())
        alive=[h for h,p in workers.items() if p.poll() is None]
        if len(alive)==2:
            limits['expanded_threads_by_horizon']={str(h):(4 if h==8 else 3) for h in alive}
        elif len(alive)==1:limits['expanded_threads_by_horizon']={str(alive[0]):6}
        temporary=OUT/'compute_limits.next.json'
        temporary.write_text(json.dumps(limits,indent=2),encoding='utf-8')
        os.replace(temporary,OUT/'compute_limits.json')
        counts={str(h):len(list((OUT/'experiments/predictions').glob(f'weekly_common_rf_h{h}_*.csv'))) for h in workers}
        state={'at_utc':datetime.now(timezone.utc).isoformat(),'completed_fold_jobs':counts,'expected_per_horizon':368,
               'workers':{str(h):{'pid':p.pid,'exit_code':p.poll()} for h,p in workers.items()},'alive_horizons':alive}
        temporary=OUT/'results/window_worker_state.next.json'
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
        os.replace(temporary,OUT/'results/window_worker_state.json')
        print(json.dumps(state),flush=True)
        if not alive:break
        time.sleep(30)
    for f in handles:f.close()
    if any(p.returncode for p in workers.values()):raise SystemExit('One or more window workers failed. Check per-horizon stderr logs.')
    print('All 1,104 expanded window jobs complete.',flush=True)


if __name__=='__main__':main()
