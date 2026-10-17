"""One bounded resume pass, then independent analysis/verification/handoff."""
import json,os,subprocess,sys,time
from pathlib import Path
from research_extensions_2026_10_round2.safe_io import read_json,write_json
OUT=Path(__file__).resolve().parent

def read(path):
    try:return read_json(path)
    except (FileNotFoundError,json.JSONDecodeError):return {}

def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    first={};pending=[];active={};done={};streams=[]
    while True:
        controller=read(OUT/'controller_state.json')
        for key,record in controller.get('finished',{}).items():
            h=int(key)
            if h in first:continue
            first[h]=read(OUT/f'results/tree_status_h{h}.json')
            first[h]['initial_process_exit_code']=record['exit_code']
            write_json(OUT/'first_pass_status.json',first)
            if first[h].get('unrun',0)>0:pending.append(h)
        for h,p in list(active.items()):
            if p.poll() is not None:done[h]=p.returncode;del active[h]
        # Once the initial controller has no queued horizon, it cannot add a
        # competing worker. Count its still-active workers before each resume.
        capacity=max(0,2-len(controller.get('active',{}))-len(active)) if controller.get('pending')==[] else 0
        while pending and capacity>0:
            h=pending.pop(0);stdout=(OUT/f'resume_h{h}.stdout.log').open('a',encoding='utf-8');stderr=(OUT/f'resume_h{h}.stderr.log').open('a',encoding='utf-8');streams.extend([stdout,stderr])
            active[h]=subprocess.Popen([sys.executable,'-u','-m','research_extensions_2026_10_round2.run_trees','--horizon',str(h)],env=env,stdout=stdout,stderr=stderr)
            capacity-=1
        write_json(OUT/'resume_state.json',{'pending':pending,'active':{h:p.pid for h,p in active.items()},'finished':done,'initial_active':controller.get('active',{}),'combined_worker_count':len(controller.get('active',{}))+len(active)})
        if controller.get('complete') and len(first)==3 and not pending and not active:break
        time.sleep(15)
    for stream in streams:stream.close()
    while not read(OUT/'support_status.json').get('complete'):time.sleep(30)
    while not (OUT/'audit_final_execution.json').exists():time.sleep(15)
    steps=[]
    for module,args in [('analyze',[]),('verify',[]),('report',[])]:
        started=time.monotonic()
        with (OUT/f'final_{module}.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/f'final_{module}.stderr.log').open('w',encoding='utf-8') as stderr:
            p=subprocess.run([sys.executable,'-u','-m','research_extensions_2026_10_round2.'+module,*args],env=env,stdout=stdout,stderr=stderr)
        steps.append({'step':module,'exit_code':p.returncode,'seconds':time.monotonic()-started})
        (OUT/'finalization.json').write_text(json.dumps({'steps':steps,'complete':False},indent=2))
        if p.returncode:raise SystemExit(p.returncode)
    (OUT/'finalization.json').write_text(json.dumps({'steps':steps,'complete':True,'passed':all(s['exit_code']==0 for s in steps),'human_visual_review_and_packaging':'pending'},indent=2))

if __name__=='__main__':main()
