"""Single bounded launcher: two disjoint horizon writers, three threads each."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from research_extensions_2026_10_round2.safe_io import write_json

OUT=Path(__file__).resolve().parent

def main(interrupted=False):
    pending=[12] if interrupted else [4,8,12];active={};finished={};logs=[]
    if interrupted:
        for h in [4,8]:finished[h]={'pid':None,'exit_code':None,'status':'Interrupted after approximately 2h50 initial pass; next pass is the one permitted resume, not a new initial budget.'}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    while pending or active:
        while pending and len(active)<2:
            h=pending.pop(0);stdout=(OUT/f'tree_h{h}.stdout.log').open('a',encoding='utf-8');stderr=(OUT/f'tree_h{h}.stderr.log').open('a',encoding='utf-8');logs.extend([stdout,stderr])
            active[h]=subprocess.Popen([sys.executable,'-u','-m','research_extensions_2026_10_round2.run_trees','--horizon',str(h)],stdout=stdout,stderr=stderr,env=env)
        for h,p in list(active.items()):
            if p.poll() is not None:finished[h]={'pid':p.pid,'exit_code':p.returncode};del active[h]
        state={'at_utc':datetime.now(timezone.utc).isoformat(),'pending':pending,'active':{h:p.pid for h,p in active.items()},'finished':finished,'complete':not pending and not active}
        write_json(OUT/'controller_state.json',state)
        if pending or active:time.sleep(15)
    for log in logs:log.close()
    print(json.dumps(state),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-interrupted',action='store_true');args=parser.parse_args();main(args.resume_interrupted)
