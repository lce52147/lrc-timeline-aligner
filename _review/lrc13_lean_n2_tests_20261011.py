"""N2 default-off and Whisper/RAW candidate eligibility regression, no GPU."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
import auto_lrc as a
parser=a.build_parser()
args=parser.parse_args(['test.flac'])
assert args.timing_source_policy=='default'
assert parser.parse_args(['test.flac','--timing-source-policy','ctc-first']).timing_source_policy=='ctc-first'
a_set=((SimpleNamespace(source='ctc-current'),SimpleNamespace(source='ctc-raw-vocal-independent-fusion'),
        SimpleNamespace(source='whisperx-vocal-independent-fusion'),SimpleNamespace(source='local-ctc')),
       (SimpleNamespace(source='raw-vocal-independent-fusion'),SimpleNamespace(source='ctc-local-whisper-vocal-independent-consensus'),
        SimpleNamespace(source='ctc-current')))
assert a.apply_central_timing_source_policy(a_set,'default') is a_set
selected=a.apply_central_timing_source_policy(a_set,'ctc-first')
assert [[x.source for x in row] for row in selected]==[['ctc-current','local-ctc'],['ctc-current']]
assert len(a_set[0])==4 and len(a_set[1])==3
# B11 exact audited source names (classification shares substring rule).
b=json.loads((root/'_review/LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
audit={x.get(side+'_source') for side in ('main','lrc13') for x in b['rows'] if x.get(side+'_source')}
audit={x for x in audit if 'raw' in x.lower() or 'whisper' in x.lower()}
assert audit
for x in audit:
 input_sets=((SimpleNamespace(source='ctc-current'),SimpleNamespace(source=x)),)
 assert [z.source for z in a.apply_central_timing_source_policy(input_sets,'ctc-first')[0]]==['ctc-current'],x
try:a.apply_central_timing_source_policy(((SimpleNamespace(source='raw-asr-current'),),),'ctc-first')
except a.LrcError:pass
else:raise AssertionError('missing fallback must fail closed')
print('N2_DEFAULT_UNCHANGED_AND_ALL_B11_WHISPER_RAW_EXCLUDED',len(audit),'sources')
