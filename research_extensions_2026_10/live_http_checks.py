"""Real HTTP checks. Credentials arrive on stdin and are never persisted.

Start run.py first, then answer the hidden credential prompts.
Only a DHIS2 identity check is performed; no surveillance is extracted.
"""
from __future__ import annotations
import csv
import hashlib
import getpass
import io
import json
import re
import sys
import time
from pathlib import Path

import requests
from zipfile import ZipFile
from xml.etree import ElementTree
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent / 'results' / 'live'
BASE = 'http://127.0.0.1:5000'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    checks = []

    def check(path, expected=200, *, params=None):
        started = time.monotonic()
        response = session.get(BASE+path, params=params, timeout=180, allow_redirects=False)
        entry = {'path': path, 'params': params, 'expected': expected,
                 'status': response.status_code, 'seconds': round(time.monotonic()-started,3),
                 'bytes': len(response.content), 'passed': response.status_code==expected}
        if response.headers.get('Content-Type','').startswith('application/json'):
            payload = response.json()
            entry['payload_status'] = payload.get('status')
            entry['reason'] = payload.get('reason')
        checks.append(entry)
        print(json.dumps(entry), flush=True)
        return response

    check('/', 302)
    check('/api/dashboard/summary',401)
    login = check('/auth/login')
    csrf = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login.text).group(1)
    credentials = {'username':getpass.getpass('DHIS2 username (hidden): '),
                   'password':getpass.getpass('DHIS2 password (hidden): ')}
    response = session.post(BASE+'/auth/login',data={**credentials,'csrf_token':csrf},timeout=90,allow_redirects=False)
    credentials.clear()
    authenticated = response.status_code==302 and response.headers.get('Location')=='/'
    checks.append({'path':'/auth/login','status':response.status_code,'passed':authenticated,'live_dhis2_identity_check':True})
    if not authenticated:
        (OUT/'http_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        raise SystemExit('Live authentication failed; no further authenticated checks performed.')
    pages=['/','/forecasts/','/climate/','/modelling/','/preprocessing/','/dhis2/','/how-it-works/','/settings/','/reports/']
    for path in pages: check(path)
    for horizon in (4,8,12):
        for name in ['summary','weekly-series','risk-table','map','climate-summary','model-performance']:
            check('/api/dashboard/'+name,params={'horizon':horizon})
        for name in ['overview','model-selection','threshold-method']:
            check('/api/explainability/'+name,params={'horizon':horizon})
    for path in ['/api/explainability/validation-method','/api/explainability/feature-importance?model=random_forest',
                 '/api/system/pipeline-status','/api/system/governance','/api/lifecycle/surveillance','/api/lifecycle/data-quality']:
        response=check(path)
        if path=='/api/system/governance' and response.status_code==200:
            (OUT/'governance_live.json').write_text(json.dumps(response.json(),indent=2),encoding='utf-8')
    for path in ['/api/dashboard/summary?horizon=999','/api/dashboard/risk-table?district_id=invalid',
                 '/api/explainability/overview?horizon=999','/reports/generate?report_type=invalid&format=csv']:
        check(path,400)
    # Read the real registry: do not guess formats or require unsupported exports.
    from app.services.report_data_service import report_registry
    from dotenv import load_dotenv
    load_dotenv()
    import csv as csv_module
    with open('data/modelling/modelling_dataset_2020_2025.csv',encoding='utf-8') as handle:
        district = next(csv_module.DictReader(handle))['district_id']
    for spec in report_registry():
        key=spec['report_type']
        parameters={'report_type':key}
        if key=='district_forecast':parameters['district_id']=district
        for fmt in spec['formats']:
            if fmt=='html':
                check('/reports/print',params=parameters)
                continue
            response=check('/reports/generate',params={**parameters,'format':fmt})
            if response.status_code != 200: continue
            path=OUT/f'{key}.{fmt}';path.write_bytes(response.content)
            if fmt=='pdf':
                reader=PdfReader(io.BytesIO(response.content));assert len(reader.pages)>0
                assert any((page.extract_text() or '').strip() for page in reader.pages)
            elif fmt=='xlsx':
                with ZipFile(io.BytesIO(response.content)) as workbook:
                    assert 'xl/workbook.xml' in workbook.namelist()
                    ElementTree.fromstring(workbook.read('xl/workbook.xml'))
                    sheets=[n for n in workbook.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')]
                    assert sheets
                    for sheet in sheets: ElementTree.fromstring(workbook.read(sheet))
            elif fmt=='csv': assert len(list(csv.reader(io.StringIO(response.content.decode('utf-8-sig')))))>1
    page=check('/')
    csrf=re.search(r'name="csrf_token"[^>]*value="([^"]+)"',page.text).group(1)
    response=session.post(BASE+'/auth/logout',data={'csrf_token':csrf},timeout=30,allow_redirects=False)
    checks.append({'path':'/auth/logout','status':response.status_code,'passed':response.status_code==302})
    check('/api/dashboard/summary',401)
    session.close()
    result={'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
            'scope':'Real local HTTP; live DHIS2 login only. No surveillance extraction or production publication.'}
    (OUT/'http_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
