"""Sequential support queue with real exit codes and retained logs."""
import os,subprocess,sys,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1');results=[]
    for module,args in [('baselines',[]),('diagnostics',[]),('classical',['seasonal']),('classical',['ridge'])]:
        tag=module+('_'+args[0] if args else '');started=time.monotonic()
        with (OUT/f'{tag}.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/f'{tag}.stderr.log').open('w',encoding='utf-8') as stderr:
            p=subprocess.run([sys.executable,'-u','-m','research_extensions_2026_10_round2.'+module,*args],stdout=stdout,stderr=stderr,env=env)
        results.append({'step':tag,'exit_code':p.returncode,'seconds':time.monotonic()-started})
        (OUT/'support_status.json').write_text(json.dumps({'steps':results,'complete':False},indent=2))
    (OUT/'support_status.json').write_text(json.dumps({'steps':results,'complete':True},indent=2))
if __name__=='__main__':main()
