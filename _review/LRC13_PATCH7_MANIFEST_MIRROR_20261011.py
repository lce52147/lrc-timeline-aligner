"""Compatibility manifest mirror only; the frozen GPU outputs remain in their actual source folder.

Current serial A/B worker was already launched with the PATCH7 folder.
The chain watches a differently named path; mirror ONLY manifest metadata
to its monitoring path. Score and final-A7 readers use the actual folder.
"""
from pathlib import Path
import json,time
v=Path(__file__).resolve().parent
source=v/'LRC13_PATCH7_PROVISIONAL_RERUN_20261011'/'PATCH7_SERIAL_MANIFEST.json'
target=v/'LRC13_FINAL7_PROVISIONAL_RERUN_20261011'/'PATCH7_SERIAL_MANIFEST.json'
target.parent.mkdir(exist_ok=True)
old=-1
while True:
    if source.is_file():
        body=source.read_bytes()
        payload=json.loads(body.decode('utf8'))
        tmp=target.with_suffix('.json.tmp')
        tmp.write_bytes(body)
        tmp.replace(target)
        n=len(payload.get('cases',[]))
        if n!=old:
            old=n
            print('MANIFEST_MIRROR',n,'/14',flush=True)
        if n==14:
            print('MANIFEST_MIRROR_14_COMPLETE',flush=True)
            break
    time.sleep(3)
