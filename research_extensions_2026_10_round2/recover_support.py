"""Resume the interrupted seasonal stage; retain its original failed attempt."""
import json,os,subprocess,sys,time
from pathlib import Path
from research_extensions_2026_10_round2.safe_io import read_json,write_json
OUT=Path(__file__).resolve().parent

def main():
    if '--child' in sys.argv:
        from research_extensions_2026_10_round2 import classical
        classical.write_json=write_json
        classical.seasonal()
        return
    if '--wait-pid' in sys.argv and os.name=='nt':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
        api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];api.WaitForSingleObject.restype=wintypes.DWORD
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=api.OpenProcess(0x00100000,False,int(sys.argv[sys.argv.index('--wait-pid')+1]))
        if handle:
            try:
                while api.WaitForSingleObject(handle,0)==258:time.sleep(15)
            finally:api.CloseHandle(handle)
    while True:
        try:status=read_json(OUT/'support_status.json')
        except (FileNotFoundError,json.JSONDecodeError):status={}
        if status.get('complete'):break
        time.sleep(15)
    write_json(OUT/'support_attempt_history.json',{'initial_queue':status,'recovery_reason':'Windows sharing/access error while atomically saving seasonal_progress.json; complete fit checkpoints preserved. Recovery changes progress I/O only, not model code/settings.','original_error_log':'classical_seasonal.stderr.log'})
    latest={s['step']:s['exit_code'] for s in status['steps']}
    if latest.get('classical_seasonal')!=0:
        status['complete']=False;status['recovery_running']=True;write_json(OUT/'support_status.json',status)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1');started=time.monotonic()
        with (OUT/'classical_seasonal_recovery.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'classical_seasonal_recovery.stderr.log').open('w',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,'-u','-m','research_extensions_2026_10_round2.recover_support','--child'],env=env,stdout=stdout,stderr=stderr)
        status['steps'].append({'step':'classical_seasonal','attempt':'checkpoint_recovery','exit_code':result.returncode,'seconds':time.monotonic()-started})
        latest['classical_seasonal']=result.returncode
    status['recovery_running']=False;status['complete']=all(v==0 for v in latest.values());write_json(OUT/'support_status.json',status)
    write_json(OUT/'support_recovery_completion.json',{'passed':status['complete'],'latest_exit_codes':latest,'progress_writer':'safe_io.write_json; identical classical.py model/checkpoint signature'})
    if not status['complete']:raise SystemExit(1)
    print('Interrupted seasonal stage recovered; all support stages successful.',flush=True)

if __name__=='__main__':main()
