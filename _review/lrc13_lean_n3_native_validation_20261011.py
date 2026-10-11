from pathlib import Path
import json,subprocess,sys,numpy as np,time
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
meta=json.loads((V/'LRC13_LEAN_N3_GPU_20261011'/'MANIFEST.json').read_text(encoding='utf8'))['cases']
from torchaudio.pipelines import MMS_FA
dic=MMS_FA.get_tokenizer().dictionary
binary=V/'lrc13_lean_n3_native_dp_20261011.exe'
assert binary.is_file()
indexes=[int(sys.argv[1])] if len(sys.argv)>1 else [4,6]
for index in indexes:
 x=next(z for z in meta if z['index']==index)
 emission=np.load(x['emission'],mmap_mode='r')
 if emission.dtype!=np.float32:raise TypeError(emission.dtype)
 testpath=V/'LRC13_LEAN_N3_NATIVE_PROBE_20261011'
 testpath.mkdir(exist_ok=True)
 baseline=V/'LRC13_LEAN_N3_CONSTRAINED_20261011'/f'{index:02d}_{x["id"]}'/'FROZEN_N3.json'
 exp=json.loads(baseline.read_text(encoding='utf8')) if baseline.is_file() else None
 for delta in (0.1,0.12,0.15):
  inp=testpath/'input.txt';out=testpath/'output.txt'
  rows=[str(syl)+'|'+','.join(str(dic[c]) for c in roman) for syl,roman in zip(x['syllables'],x['romans'])]
  inp.write_text('\n'.join([str(x['emission']),str(emission.shape[0])+' '+str(emission.shape[1]),repr(x['seconds_per_frame']),repr(delta),str(len(rows))]+rows)+'\n',encoding='utf8')
  t=time.monotonic()
  p=subprocess.run([str(binary),str(inp),str(out)],cwd=ROOT,capture_output=True,encoding='utf8',errors='replace')
  if p.returncode:print('NATIVE_FAIL',index,delta,p.stderr[-1000:]);raise SystemExit(p.returncode)
  parsed=out.read_text(encoding='utf-8-sig').splitlines()
  score=float(parsed[0]);frames=[int(y) for y in parsed[1].split(',')]
  assert len(frames)==len(x['romans'])
  clocks=[f*x['seconds_per_frame'] for f in frames]
  diffs=[]
  if exp:
   e=exp['threshold_results'][str(delta)]
   if e['status']=='CONSTRAINED_VITERBI_EXACT':
    diffs=[(j+1,round(a-b,3)) for j,(a,b) in enumerate(zip(clocks,e['starts'])) if abs(a-b)>x['seconds_per_frame']+.001]
   else:diffs=[(j+1,round(a-b,3)) for j,(a,b) in enumerate(zip(clocks,x['zero_starts'])) if abs(a-b)>x['seconds_per_frame']+.001]
  print('NATIVE_CHECK',index,'delta',delta,'seconds',round(time.monotonic()-t,2),'score',round(score,2),'diff_count',len(diffs),'sample',diffs[:10],flush=True)
  if diffs:raise SystemExit('NATIVE_PARITY_MISMATCH')
print('NATIVE_CROSSCHECK_PASS')
