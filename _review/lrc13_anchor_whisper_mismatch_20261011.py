"""Offline preliminary mismatch analysis; ASR evidence is not verified sung ground truth."""
from pathlib import Path
import json,unicodedata,re,difflib
from collections import defaultdict
import pykakasi
v=Path(__file__).resolve().parent
base=json.loads((v/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))['rows']
asr=json.loads((v/'LRC13_P1_RAW_ASR_PARAGRAPHS_20261011.json').read_text(encoding='utf8'))['songs']
k=pykakasi.kakasi()
def rom(s):
 s=unicodedata.normalize('NFKC',s).casefold()
 units=k.convert(s)
 return re.sub('[^a-z0-9]','', ''.join(x.get('hepburn','') for x in units).lower())
def similarity(a,b):
 return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
names=['03. Choir ‘S’ Choir','01 My Dearest','01.ミレニアの水槽']
results=[]
for name in names:
 lrc=[r for r in base if r['song']==name]
 track=next(z for z in asr if z['id'].lower().replace(' ','')==name.lower().replace(' ',''))
 events=[{'start':float(e[0]),'end':float(e[1]),'text':e[2],
     'romaji':rom(e[2])} for e in track['events'] if len(e)>=3]
 items=[]
 for r in lrc:
  target=rom(r['text'])
  ranked=sorted(((similarity(target,e['romaji']),j) for j,e in enumerate(events)),
    reverse=True)
  sim,index=ranked[0]
  matched=events[index]
  rel=matched['start']-r['CTC_s']
  if abs(rel)>12 and sim>.6:label='LEXICAL_MATCH_OUTSIDE_CTC_12S'
  elif sim<.35:label='NO_RELIABLE_WHOLE_ASR_TEXT_MATCH'
  elif abs(rel)>2 and sim>.6:label='LEXICAL_MATCH_CTC_OFFSET'
  else:label='AMBIGUOUS_OR_NEAR'
  items.append({'line':r['line'],'lyric':r['text'],
    'CTC':r['CTC_s'],'BK_posthoc':r['BK_s'],'CTC_abs_error_posthoc':r['errors']['CTC_s'],
    'match_text':matched['text'],'match_start':matched['start'],
    'similarity':round(sim,4),'ctc_minus_asr_s':round(-rel,3),'label':label,
    'alt_matches':[{'score':round(s,3),'time':events[j]['start']}
                   for s,j in ranked[1:4]]})
 gaps=[e for e in events if max((similarity(e['romaji'],rom(r['text'])) for r in lrc),default=0)<.35]
 results.append({'song':name,'ASR_events':len(events),'lyric_rows':len(lrc),
   'raw_transcription_extra_or_hallucinated_events':len(gaps),
   'events_without_lyric_match':gaps[:35],
   'lines':items,'large_CTC_errors':[x for x in items if x['CTC_abs_error_posthoc']>1],
   'weak_ASR_matching_lines':[x for x in items if x['similarity']<.35],
   'caution':'ASR misrecognitions, repeated choruses and multi-line ASR spans prevent definitive lyric-omission inference.'})
(v/'LRC13_ANCHOR_ASR_LYRIC_TRIAGE_20261011.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
for r in results:
 print('SONG',r['song'].encode('unicode_escape').decode(),'EVENTS',r['ASR_events'],
    'WEAK',len(r['weak_ASR_matching_lines']),'LARGE_CTC',len(r['large_CTC_errors']))
 for x in r['large_CTC_errors'][:8]:
  print('ROW',x['line'],'ctc_error',x['CTC_abs_error_posthoc'],'text_sim',x['similarity'],
     'match_time',x['match_start'],'delta',x['ctc_minus_asr_s'],x['label'])
