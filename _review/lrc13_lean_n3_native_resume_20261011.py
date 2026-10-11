"""Exact N3 duration-constrained CTC native continuation from a frozen manifest.

Uses same augmented CTC state machine, emissions, labels, thresholds and
checkpoint traceback as the original Python implementation. BK never read.
Crosschecked native vs Python entire lines for songs #4 and #6 at all deltas.
"""
from pathlib import Path
import json,subprocess,time,sys,math
import numpy as np
from torchaudio.pipelines import MMS_FA
V=Path(__file__).resolve().parent
IN=V/'LRC13_LEAN_N3_GPU_20261011'/'MANIFEST.json'
OUT=V/'LRC13_LEAN_N3_CONSTRAINED_20261011'
MANIFEST=OUT/'MANIFEST.json'
binary=V/'lrc13_lean_n3_native_dp_20261011.exe'
if not binary.is_file():
    csc=Path(r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe')
    source=V/'lrc13_lean_n3_native_dp_20261011.cs'
    assert csc.is_file() and source.is_file(), 'N3_NATIVE_COMPILER_MISSING'
    subprocess.run([str(csc),'/nologo','/optimize+','/platform:x64',
        '/out:'+str(binary),str(source)],check=True)
assert binary.is_file()
inp=json.loads(IN.read_text(encoding='utf8'))
assert inp.get('status')=='ZERO_PARITY_39_COMPLETE'
deltas=(0.10,0.12,0.15)
existing=json.loads(MANIFEST.read_text(encoding='utf8')) if MANIFEST.exists() else {}
done={x['index']:x for x in existing.get('cases',[]) if x.get('status')=='DONE'}
dic=MMS_FA.get_tokenizer().dictionary
def emit(phase,index):
    allcases=list(done.values())
    assert len(set(x['index'] for x in allcases))==len(allcases)
    payload={'status':phase,'completed':len(allcases),'total':39,'cases':allcases,
        's_min':deltas,'BK_not_accessed':True,
        'algorithm':'EXACT_CTC_VITERBI_PER_LINE_AGE_CONSTRAINT',
        'native_speedup_crosschecked_songs':[4,6],
        'native_logic_does_not_change_scoring_policy':True}
    MANIFEST.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print('N3_NATIVE_PROGRESS',phase,len(allcases),index,flush=True)
for job in inp['cases']:
    i=job['index']
    if i in done:continue
    print('N3_NATIVE_BEGIN',i,'flagged',job['flagged'],flush=True)
    output=OUT/f'{i:02d}_{job["id"]}'
    output.mkdir(exist_ok=True)
    frozen=output/'FROZEN_N3.json'
    if frozen.exists():
        raise RuntimeError('REFUSE_UNMANIFESTED_RESULT '+str(frozen))
    matrix=np.load(job['emission'],mmap_mode='r',allow_pickle=False)
    assert matrix.dtype==np.float32 and matrix.ndim==2
    spf=job['seconds_per_frame']
    starts=job['zero_starts']
    result={}
    rows=[str(syl)+'|'+','.join(str(dic[c]) for c in roman)
          for syl,roman in zip(job['syllables'],job['romans'])]
    for minimum in deltas:
        k=str(minimum)
        violation=[j+1 for j,(a,b) in enumerate(zip(starts,starts[1:]))
             if b-a<math.ceil(job['syllables'][j]*minimum/spf-1e-10)*spf-1e-9]
        if not violation:
            result[k]={'status':'EXACT_BASELINE_FEASIBLE_GLOBAL_VITERBI_OPTIMUM',
                'starts':starts,'violating_rows_before':[],
                'algorithm_exact':True}
            continue
        input_file=output/f'native_input_{k}.txt'
        output_file=output/f'native_output_{k}.txt'
        lines=[str(job['emission']),
            str(matrix.shape[0])+' '+str(matrix.shape[1]),
            repr(spf),repr(minimum),str(len(rows))]+rows
        input_file.write_text('\n'.join(lines)+'\n',encoding='utf8')
        t=time.monotonic()
        p=subprocess.run([str(binary),str(input_file),str(output_file)],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         encoding='utf8',errors='replace')
        if p.returncode:
            raise RuntimeError('N3_NATIVE_SOLVER_FAILED '+str(i)+' '+k+' '+p.stderr[-2000:])
        payload=output_file.read_text(encoding='utf-8-sig').splitlines()
        score=float(payload[0])
        frames=[int(x) for x in payload[1].split(',')]
        assert len(frames)==len(starts)
        outstarts=[round(f*spf,6) for f in frames]
        for j,(a,b) in enumerate(zip(outstarts,outstarts[1:])):
            if b-a+spf*.51<job['syllables'][j]*minimum:
                raise ValueError('N3_NATIVE_DURATION_FAIL '+repr((i,k,j+1,a,b)))
        result[k]={'status':'CONSTRAINED_VITERBI_EXACT','starts':outstarts,
            'score':score,'violating_rows_before':violation,
            'elapsed_s':round(time.monotonic()-t,2),
            'algorithm_exact':True,'solver':'compiled-C#-crosschecked'}
        input_file.unlink()
        output_file.unlink()
    payload={'status':'N3_EXACT_FROZEN','index':i,'id':job['id'],
        'BK_read':False,'flagged':job['flagged'],
        'zero_parity':True,'H0_starts':starts,
        'threshold_results':result,'seconds_per_frame':spf}
    frozen.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    done[i]={'index':i,'id':job['id'],'flagged':job['flagged'],
        'status':'DONE','output':str(frozen)}
    emit('IN_PROGRESS',i)
if len(done)==39:
    emit('N3_EXACT_39_FROZEN',39)
