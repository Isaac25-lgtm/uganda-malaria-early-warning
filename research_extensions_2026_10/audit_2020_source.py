"""Small current-vintage audit of the leading 2020 gaps; preserve frozen data."""
import csv
import getpass
import hashlib
import json
from datetime import date,datetime,timezone
from pathlib import Path
from dotenv import load_dotenv
from app.services.dhis2_client import DHIS2Client,DHIS2HTTPError,read_district_level_from_env
from app.services.dhis2_extraction import parse_analytics_rows
from research_extensions_2026_10.run_study import OUT,RESULTS,write_json


def main(periods=None,folder='dhis2_2020_start_audit',report_name='2020_live_source_audit.json'):
    load_dotenv();private=OUT/'private'/folder;private.mkdir(parents=True,exist_ok=True)
    if (private/'manifest.json').exists():raise SystemExit('Completed audit preserved; do not overwrite its vintage.')
    mapping=list(csv.DictReader(open('data/reference/dhis2_orgunit_mapping_review.csv',encoding='utf-8-sig')))
    by_uid={r['dhis2_orgunit_id']:r for r in mapping};periods=periods or [f'2020W{w:02d}' for w in range(1,7)]
    for period in periods:
        if date.fromisocalendar(int(period[:4]),int(period.split('W')[1]),7)>date(2026,6,30):
            raise ValueError('Source audit exceeds the June 2026 surveillance cap.')
    old=list(csv.DictReader(open(OUT/'baseline/data/modelling/dhis2_malaria_raw_snapshot_2020_2025_run170.csv',encoding='utf-8')))
    oldvalues={(r['district_id'],r['period']):r['confirmed_malaria_cases'] for r in old if r['period'] in periods}
    # The frozen analysis snapshot blanks adjudicated outcomes. Source-vintage
    # comparisons must restore the original reported value, without fitting it.
    for row in csv.DictReader(open('data/reference/malaria_case_exclusions.csv',encoding='utf-8-sig')):
        key=row['district_id'],f"{row['epi_year']}W{int(row['epi_week']):02d}"
        if key in oldvalues and oldvalues[key]=='':oldvalues[key]=row['original_cases']
    username=getpass.getpass('DHIS2 username (hidden): ');password=getpass.getpass('DHIS2 password (hidden): ')
    with DHIS2Client.from_credentials(username,password) as client:
        username=password='';client.verify_login()
        errors=[]
        def capture_error(response,*args,**kwargs):
            if response.status_code>=400:
                try:body=response.json()
                except ValueError:body={'message':response.text[:1000]}
                errors.append({'http_status':response.status_code,'body':body})
        client.session.hooks['response'].append(capture_error)
        try:payload=client.analytics_query(dx='fUflbWWhouR',periods=periods,level=read_district_level_from_env())
        except DHIS2HTTPError as exc:
            write_json(private/'error_response.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'errors':errors})
            safe={'status':'incomplete_source_error','periods':periods,'http_status':exc.status_code,
                'source_message':str(errors[-1]['body'].get('message',''))[:500] if errors else '',
                'frozen_data_modified':False}
            write_json(RESULTS/report_name,safe);print(json.dumps(safe),flush=True);raise
    (private/'response.json').write_text(json.dumps(payload),encoding='utf-8')
    rows=parse_analytics_rows(payload,fallback_dx='fUflbWWhouR',fallback_dx_name='confirmed malaria cases')
    records=[];newvalues={}
    for r in rows:
        assert r.period in periods and r.dhis2_orgunit_id in by_uid and r.data_element_id=='fUflbWWhouR'
        key=by_uid[r.dhis2_orgunit_id]['district_id'],r.period;assert key not in newvalues
        newvalues[key]=str(r.value);records.append({'district_id':key[0],'period':r.period,'current_cases':str(r.value),'frozen_cases':oldvalues.get(key,'')})
    from decimal import Decimal
    appeared=[k for k in newvalues if k not in oldvalues];disappeared=[k for k in oldvalues if k not in newvalues]
    changed=[k for k in set(newvalues)&set(oldvalues) if Decimal(newvalues[k])!=Decimal(oldvalues[k])]
    path=private/'comparison.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['district_id','period','current_cases','frozen_cases']);writer.writeheader();writer.writerows(records)
    result={'extracted_at_utc':datetime.now(timezone.utc).isoformat(),'periods':periods,'expected_rows':146*len(periods),
        'frozen_rows':len(oldvalues),'current_rows':len(newvalues),'newly_available_rows':len(appeared),
        'disappeared_rows':len(disappeared),'changed_common_values':len(changed),'raw_comparison_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'frozen_data_modified':False,'purpose':'Compare requested historical periods with current DHIS2; preserve original reported errors, fitted-model data and the locked 2026 protocol.',
        'period_coverage':[{'period':p,'frozen_rows':sum(k[1]==p for k in oldvalues),'current_rows':sum(k[1]==p for k in newvalues)} for p in periods]}
    write_json(private/'manifest.json',result);write_json(RESULTS/report_name,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
