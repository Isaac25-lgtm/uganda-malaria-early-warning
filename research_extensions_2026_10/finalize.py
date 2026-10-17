"""Run the finishing checks only after all bounded workers succeed."""
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from research_extensions_2026_10.run_study import OUT,RESULTS,write_json


def ready():
    for filename in ['window_worker_state.json','climate_worker_state.json']:
        try:state=json.loads((RESULTS/filename).read_text())
        except json.JSONDecodeError:return False
        if any(w['exit_code'] not in [None,0] for w in state['workers'].values()):
            raise RuntimeError('A bounded worker failed; inspect its horizon log.')
        if len(state['workers'])!=3 or any(w['exit_code'] is None for w in state['workers'].values()):return False
    audit=RESULTS/'2026_model_reconstruction_audit.json'
    return audit.exists() and json.loads(audit.read_text())['complete']


def main():
    directory=RESULTS/'finalization';directory.mkdir(parents=True,exist_ok=True)
    while not ready():
        print('Waiting for complete weekly fits and the historical reconstruction audit.',flush=True);time.sleep(30)
    commands=[('tables',['analyze_study','tables']),('diagnostics',['summarize_diagnostics']),
        ('simple_baselines',['summarize_simple_baselines']),('figures',['build_figures']),('poster',['build_poster']),
        ('pdf_render',['pdf_quality_checks']),('targeted_and_syntax',['verify_suite','--targeted-only']),
        ('independent_results',['verify_results','--require-complete']),('handoff',['build_handoff','--require-complete'])]
    results=[]
    for name,arguments in commands:
        started=time.monotonic()
        print('Starting final check:',name,flush=True)
        with (directory/f'{name}.stdout.log').open('w',encoding='utf-8') as stdout,(directory/f'{name}.stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.run([sys.executable,'-u','-m','research_extensions_2026_10.'+arguments[0],*arguments[1:]],
                cwd=OUT.parent,stdout=stdout,stderr=stderr)
        results.append({'step':name,'exit_code':child.returncode,'seconds':round(time.monotonic()-started,3)})
        write_json(RESULTS/'finalization.json',{'checked_at_utc':datetime.now(timezone.utc).isoformat(),'steps':results,
            'complete':len(results)==len(commands),'passed':all(r['exit_code']==0 for r in results)})
        if child.returncode:raise SystemExit('Finishing check failed: '+name)
    print('All numerical finishing checks and draft handoff generation complete. Final human review remains.',flush=True)


if __name__=='__main__':main()
