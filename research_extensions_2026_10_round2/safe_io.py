"""Brief snapshot reads and bounded Windows atomic-write retries."""
import json,os,time,uuid
from pathlib import Path

def read_json(path):
    # Close before decoding. Avoid repeatedly opening mutable status files while
    # a legacy writer without retries owns them; monitor uses immutable fits.
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.next')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')
    deadline=time.monotonic()+3
    try:
        while True:
            try:os.replace(temp,path);return
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.05)
    finally:
        if temp.exists():temp.unlink()
