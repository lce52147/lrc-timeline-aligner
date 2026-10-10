"""BK-posthoc 30-row P5 timing invariant audit of frozen A7 LRCs."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import auto_lrc as a
review=ROOT/'_review'
d=json.loads((review/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf-8'))['corpus'][37]
assert d['id']=='09.夢遊' and d['BK_sung_rows']==30
f='38_09.夢遊'
bk=Path(d['human_BK'])
actual=review/'lrc13_option_a_A7_39_20261010'/f/'candidate.lrc'
main=review/'lrc13_option_a_A7_main_9069dbc8_20261010'/f/'main.lrc'
def parse(file):
    e=a.load_lyrics(file).entries
    e=a.remove_generated_title_cards(e,Path(d['audio_path']))
    e,_=a.remove_instrumental_markers(e)
    return [(a.normalize_match_text(a.entry_sung_text(x)),x.source_time_cs/100.) for x in e]
ref,seg,base=parse(bk),parse(actual),parse(main)
assert len(ref)==len(seg)==len(base)==30
assert all(x[0]==y[0]==z[0] for x,y,z in zip(ref,seg,base))
assert ref[11][1]==63.9 and ref[28][1]==174.25
assert ref[27][1]==164.65 and seg[27][1]==160.89 and base[27][1]==164.79
lines=['# P5: Dream 30-row frozen A7 timing audit (2026-10-11)','',
'Scope: timing values below are the frozen A7 outputs before P5, not a fresh GPU replay.',
'P5 preserves CTC evidence and modifies F4/F5 flags only; unit tests show no timestamp mutation.',
'P5 acceptance on 39 fresh skeleton-layer runs: **PENDING**.','',
'| # | BK(s) | pinned main(s) | A7 segment(s) | segment error(s) |',
'|---:|---:|---:|---:|---:|']
for i,(b,m,s) in enumerate(zip(ref,base,seg),1):
    lines.append(f'| {i} | {b[1]:.2f} | {m[1]:.2f} | {s[1]:.2f} | {s[1]-b[1]:+.2f} |')
lines.extend(['','P5 new F5 synthetic regression: repeated Dream #28 (global CTC 164.79 vs segment 160.89) must flag skeleton_global_ctc_disagreement and downgrade trust. This does not change 160.89 to the correct BK time.','',
'Full unit suite: 711 tests (2 skipped), exit 0.',''])
out=review/'LRC13_SKELETON_P5_FROZEN_AUDIT_20261011.md'
out.write_text('\n'.join(lines),encoding='utf-8')
print('FROZEN_DREAM_AUDIT_WRITTEN',out,'rows',len(ref),flush=True)
