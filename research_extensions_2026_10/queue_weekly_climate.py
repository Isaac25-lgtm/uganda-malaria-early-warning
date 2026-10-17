"""Fill spare slots with one bounded comparator; do not interfere with window writers."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_extensions_2026_10'


def read_window_state():
    # Older supervisors wrote directly; tolerate a read during that brief write.
    for attempt in range(10):
        try:
            return json.loads((OUT/'results/window_worker_state.json').read_text())
        except (json.JSONDecodeError,PermissionError):
            if attempt==9:raise
            time.sleep(.2)


def main():
    workers={};handles=[]
    while True:
        state=read_window_state()
        pending=[]
        for h in [4,8,12]:
            mainworker=state['workers'][str(h)]
            if mainworker['exit_code'] not in [None,0]:raise SystemExit('Window queue failed; paused secondary scheduling.')
            # Once the matched 2020 full-climate scenario is finished, its
            # independent comparator can fill spare slots while shorter starts
            # continue. Prefixes remain disjoint; no scientific choice changes.
            full_ready=len(list((OUT/'experiments/predictions').glob(f'weekly_common_rf_h{h}_start2020_all_*.csv')))==92
            if h not in workers and (mainworker['exit_code']==0 or full_ready):
                stdout=open(OUT/f'weekly_climate_h{h}.stdout.log','ab',buffering=0);stderr=open(OUT/f'weekly_climate_h{h}.stderr.log','ab',buffering=0);handles.extend([stdout,stderr])
                workers[h]=subprocess.Popen([sys.executable,'-u','-m','research_extensions_2026_10.run_study','weekly_ablation','--horizon',str(h)],
                    cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            elif h not in workers:pending.append(h)
        active=[h for h,p in workers.items() if p.poll() is None]
        if len(state['alive_horizons'])>=2:threads={str(h):1 for h in active}
        elif len(state['alive_horizons'])==1:threads={str(h):2 if len(active)==1 else 1 for h in active}
        else:threads={str(h):6 if len(active)==1 else (4 if h==8 else 2) for h in active}
        temporary=OUT/'climate_compute_limits.next.json'
        temporary.write_text(json.dumps({'threads_by_horizon':threads},indent=2),encoding='utf-8');os.replace(temporary,OUT/'climate_compute_limits.json')
        counts={str(h):len(list((OUT/'experiments/predictions').glob(f'weekly_climate_rf_h{h}_*.csv'))) for h in [4,8,12]}
        report={'at_utc':datetime.now(timezone.utc).isoformat(),'pending_window_horizons':pending,'active_climate_horizons':active,
            'completed_fold_jobs':counts,'expected_per_horizon':92,'workers':{str(h):{'pid':p.pid,'exit_code':p.poll()} for h,p in workers.items()}}
        temporary=OUT/'results/climate_worker_state.next.json';temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');os.replace(temporary,OUT/'results/climate_worker_state.json')
        if len(workers)==3 and not active:break
        time.sleep(30)
    for f in handles:f.close()
    if any(p.returncode for p in workers.values()):raise SystemExit('Climate comparator failed; inspect its horizon stderr log.')
    print('All 276 exploratory weekly climate comparator fits complete.',flush=True)


if __name__=='__main__':main()
