"""Capture actual child exit codes, without PowerShell stderr interpretation."""
from __future__ import annotations
import ast
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_extensions_2026_10/results/verification'


def main(targeted_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    groups={'application':['-m','unittest','discover','-s','tests','-v'],
            'literature_benchmark':['-m','unittest','research_comparisons.literature_based_forecasting_benchmark.tests.test_benchmark','-v']}
    if targeted_only:
        groups={'scientific_targeted':['-m','unittest','tests.test_research_corrections','tests.test_research_statistics','tests.test_holdout_checkpoints','-v']}
    for name,args in groups.items():
        started=time.monotonic()
        with (OUT/f'{name}.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/f'{name}.stderr.log').open('w',encoding='utf-8') as stderr:
            process=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stdout,stderr=stderr)
        result={'suite':name,'exit_code':process.returncode,'seconds':round(time.monotonic()-started,3)}
        results.append(result);print(json.dumps(result),flush=True)
        if process.returncode:break
    python_files=[]
    for folder in ['app','tests','research_extensions_2026_10','research_comparisons/literature_based_forecasting_benchmark']:
        for path in (ROOT/folder).rglob('*.py'):
            if 'baseline' in path.parts or '__pycache__' in path.parts or 'runs' in path.parts:continue
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));python_files.append(path.relative_to(ROOT).as_posix())
    js_results=[]
    for path in (ROOT/'app/static/js').glob('*.js'):
        process=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
        js_results.append({'path':path.relative_to(ROOT).as_posix(),'exit_code':process.returncode,'error':process.stderr[:1000]})
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'test_suites':results,'parsed_python_files':len(python_files),
            'javascript_checks':js_results,'passed':all(r['exit_code']==0 for r in results+js_results)}
    (OUT/('verification_targeted.json' if targeted_only else 'verification.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--targeted-only',action='store_true');main(parser.parse_args().targeted_only)
