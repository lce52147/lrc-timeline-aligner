"""BK-blind N3: frozen MMS full-song emission and s_min=0 parity, serial 39 GPU songs.
N3 comparison thresholds fixed: seconds per sung syllable 0.10, 0.12, 0.15.
Flag songs when Opus original naive raw-lyric vowel-group sec/line < 0.35.
Only reports/stems read, never references/BK.
"""
from pathlib import Path
import os,sys,json,re,math,subprocess,time,traceback
import numpy as np
import torch,torchaudio
V=Path(__file__).resolve().parent;ROOT=V.parent
sys.path.insert(0,str(ROOT/'scripts'))
import ctc_align as ctc
THRESHOLDS=(0.10,0.12,0.15)
SPEED_FLAG=.35
VOWELS=re.compile('[aeiouy]+',re.I)
FILES=V/'LRC13_LEAN_N3_GPU_20261011'
FILES.mkdir(exist_ok=True)
CORPUS=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
MANIFEST=FILES/'MANIFEST.json'
DONE={x['index']:x for x in json.loads(MANIFEST.read_text(encoding='utf8'))['cases'] if x['status']=='ZERO_PARITY_PASS'} if MANIFEST.exists() else {}
cases=list(DONE.values())
def manifest(status,i=0):
 MANIFEST.write_text(json.dumps({'status':status,'cases':cases,'thresholds':THRESHOLDS,'bk_used':False},ensure_ascii=False,indent=2),encoding='utf8')
 print('N3',status,'done',len(DONE),'index',i,flush=True)
bundle=torchaudio.pipelines.MMS_FA
device='cuda' if torch.cuda.is_available() else 'cpu'
model=bundle.get_model(with_star=False).to(device).eval()
if device=='cuda':ctc.configure_cuda_cpu_feature_extractor(model)
tokenizer=bundle.get_tokenizer()
aligner=bundle.get_aligner()
from contextlib import nullcontext
if device=='cuda':
 from torch.nn.attention import SDPBackend,sdpa_kernel
 context=lambda:sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
else:context=lambda:nullcontext()
for i,item in enumerate(CORPUS,1):
 if i in DONE:continue
 leaf=f'{i:02d}_{item["id"]}'
 reportfiles=list((V/'lrc13_option_a_A7_main_9069dbc8_20261010'/leaf/'reports').glob('*.align-report.json'))
 assert len(reportfiles)==1
 report=json.loads(reportfiles[0].read_text(encoding='utf8'))
 rows=report['assignments']
 stem=Path(report['ctc_audio_path'])
 assert stem.exists(),stem
 out=FILES/leaf
 out.mkdir(exist_ok=True)
 beginnings=[float(r['ctc_first_token_start']) for r in rows]
 romans=[''.join(z['char'] for z in r['ctc_token_spans']) for r in rows]
 assert all(s and all(c in tokenizer.dictionary for c in s) for s in romans)
 lyric_syl=[max(1,len(VOWELS.findall(s))) for s in romans]
 is_flagged=any((b-a)/max(1,len(VOWELS.findall(str(rows[j].get('text') or romans[j]))))<SPEED_FLAG for j,(a,b) in enumerate(zip(beginnings,beginnings[1:])))
 # Exact Opus song flag must derive from raw sung lyric strings, stored in report's assignment text if available.
 # Recapture text from BK only AFTER output frozen is forbidden: instead use original complete LRC (lyric text, not times).
 from pathlib import Path as _Path
 import unicodedata
 # Mirror Opus B11 text: provenance source is recorded main final LRC lyric text, timestamps ignored.
 parsed=[]
 for line in (V/'lrc13_option_a_A7_main_9069dbc8_20261010'/leaf/'main.lrc').read_text(encoding='utf8').splitlines():
  match=re.match(r'^\s*(?:\[\d+:\d+(?:\.\d+)?\])+\s*(.*)$',line)
  if match and match.group(1).strip():parsed.append(match.group(1).strip())
 # Some files have additional metadata/time rows; provenance cleanup alignment required.
 if len(parsed)!=len(rows):
  raise ValueError('SPEED_FLAG_TEXT_ROW_MISMATCH '+str(i)+' '+str(len(parsed))+' '+str(len(rows)))
 is_flagged=any((b-a)/max(1,len(VOWELS.findall(parsed[j])))<SPEED_FLAG for j,(a,b) in enumerate(zip(beginnings,beginnings[1:])))
 print('N3_GPU_BEGIN',i,'flag',is_flagged,'rows',len(rows),flush=True)
 t=time.monotonic()
 waveform=ctc.decode_audio(stem)
 duration=waveform.shape[1]/ctc.SAMPLE_RATE
 with torch.inference_mode(),context():
  emissions,_=model(waveform if device=='cuda' else waveform.to(device))
 emission=emissions[0].cpu()
 spans=aligner(emission,tokenizer(romans))
 zero_starts=[float(a[0].start)*duration/len(emission) if a else None for a in spans]
 assert len(zero_starts)==len(beginnings)
 tol=duration/len(emission)+0.003
 differences=[{'line':j+1,'original':round(a,3),'zero':round(b,3) if b is not None else None,'diff':round(abs(a-b),3) if b is not None else None} for j,(a,b) in enumerate(zip(beginnings,zero_starts)) if b is None or abs(a-b)>tol]
 name=out/'EMISSIONS_BK_BLIND.npy'
 np.save(name,emission.numpy(),allow_pickle=False)
 job={'index':i,'id':item['id'],'status':'ZERO_PARITY_PASS' if not differences else 'ZERO_PARITY_FAIL',
 'rows':len(rows),'flagged':is_flagged,'syllables':lyric_syl,
 'ctc_starts':beginnings,'zero_starts':zero_starts,'original_text':parsed,
 'romans':romans,'duration':duration,'seconds_per_frame':duration/len(emission),
 'parity_errors':differences,'elapsed_s':round(time.monotonic()-t,2),'emission':str(name),
 'full_song_model':'MMS_FA','device':device,'stem':str(stem)}
 (out/'ZERO_PARITY.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf8')
 cases.append(job)
 if differences:
  manifest('ZERO_PARITY_FAILED',i)
  raise SystemExit('PARITY_FAILED '+str(i)+' '+str(differences[:3]))
 DONE[i]=job
 manifest('ZERO_PARITY_IN_PROGRESS',i)
manifest('ZERO_PARITY_39_COMPLETE')
