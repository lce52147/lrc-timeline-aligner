"""BK-BLIND experimental anchor-constrained MMS/CTC alignment (isolated _review).
Fixed comparison rules BEFORE any reference/BK score:
  normalized unique line, all other bigram Dice < .8; CTC score >= .08;
  existing vocal onset peak within .3 s; strictly ordered anchors.
  Switch thresholds delta = 0.5, 2.0, 5.0 nats / lyric line; >= coverage.
No Central, WhisperX, reviewer or production script modifications.
"""
from __future__ import annotations
import json,sys,re,unicodedata,math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torchaudio
V=Path(__file__).resolve().parent;ROOT=V.parent
sys.path[:0]=[str(ROOT/'scripts')]
import ctc_align as ctc
import auto_lrc as al
DELTA_THRESHOLDS=(0.5,2.0,5.0)
ANCHOR_CTC_MIN=.08
ANCHOR_MAX_DICE=.8
ANCHOR_ONSET_RADIUS=.3

def canonical(text:str)->str:
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC',text)
                   if c.isalnum()).replace(' ','')
def dice(a:str,b:str)->float:
    aa=Counter(a[i:i+2] for i in range(len(a)-1))
    bb=Counter(b[i:i+2] for i in range(len(b)-1))
    return 2*sum((aa&bb).values())/(sum(aa.values())+sum(bb.values())) if aa and bb else float(a==b)
def tokens_of(row):
    spans=row.get('ctc_token_spans') or []
    return canonical(''.join(str(x.get('char') or '') for x in spans))
def start_of(row)->float|None:
    v=row.get('ctc_first_token_start')
    return float(v) if isinstance(v,(int,float)) and math.isfinite(float(v)) else None
def end_of(row)->float|None:
    spans=row.get('ctc_token_spans') or []
    x=spans[-1].get('end') if spans else None
    return float(x) if isinstance(x,(int,float)) and math.isfinite(float(x)) else None
def choose_anchors(rows,features):
    names=[tokens_of(q) for q in rows]
    counts=Counter(names)
    candidates=[]
    logs=[]
    for i,(r,text) in enumerate(zip(rows,names)):
        reasons=[]
        if not text or counts[text]!=1 or any(
            k!=i and dice(text,other)>=ANCHOR_MAX_DICE
            for k,other in enumerate(names)):
            reasons.append('NONUNIQUE_OR_SIMILAR')
        score=r.get('ctc_score')
        if not isinstance(score,(int,float)) or score<ANCHOR_CTC_MIN:
            reasons.append('CTC_SCORE_BELOW_008')
        t=start_of(r)
        e=end_of(r)
        if t is None or e is None or e<t:reasons.append('NO_VALID_CTC_TOKEN_CLOCK')
        peak=al._strong_vocal_onset_near(features,t) if t is not None else None
        if peak is None or abs(peak[0]-t)>ANCHOR_ONSET_RADIUS:
            reasons.append('VOCAL_ONSET_NOT_WITHIN_03')
        logs.append({'line':i+1,'time':t,'ctc_score':score,
                     'peak_s':peak[0] if peak else None,'reasons':reasons})
        if not reasons:candidates.append(i)
    violating=set()
    for left,right in zip(candidates,candidates[1:]):
        if start_of(rows[left])>=start_of(rows[right]):
            violating.update((left,right))
    if violating:
        for i in violating:logs[i]['reasons'].append('NONMONOTONIC_ANCHOR_PAIR')
    return [i for i in candidates if i not in violating],logs

def viterbi_ctc_ll(emission,tokens:list[int],blank:int,
                   fixed_onset_frames:dict[int,int]|None=None)->float:
    """Same emission, transcript and frame-wise CTC Viterbi objective for H0/H1.

    H0 requires each line's *first* target token to emit at the existing
    global-CTC clock. H1 omits that constraint. Unlike independent line
    re-alignment, this retains blank/repeated-character transition rules
    across line boundaries and scores every frame exactly once.
    """
    x=emission.detach().cpu().numpy()
    T=len(x)
    if not tokens or T<len(tokens):raise ValueError('CTC_INSUFFICIENT_FRAMES')
    labels=np.full(2*len(tokens)+1,int(blank),dtype=np.int64)
    labels[1::2]=np.asarray(tokens,dtype=np.int64)
    S=len(labels)
    skip_allowed=np.zeros(S,dtype=np.bool_)
    skip_allowed[3::2]=labels[3::2]!=labels[1:-2:2]
    boundaries=sorted((int(frame),2*int(index)+1)
                      for index,frame in (fixed_onset_frames or {}).items())
    if any(f<0 or f>=T or j>=S for f,j in boundaries):
        raise ValueError('OUT_OF_WINDOW_H0_FIXED_ONSET')
    if any(a[0]>=b[0] or a[1]>=b[1]
           for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('NONMONOTONIC_H0_FIXED_ONSETS')
    dp=np.full(S,-np.inf,dtype=np.float64)
    dp[0]=0.
    next_boundary=0
    for t in range(T):
        stay=dp
        adv=np.empty(S,dtype=np.float64)
        adv[0]=-np.inf
        adv[1:]=dp[:-1]
        skip=np.full(S,-np.inf,dtype=np.float64)
        skip[2:]=np.where(skip_allowed[2:],dp[:-2],-np.inf)
        nxt=np.maximum(np.maximum(stay,adv),skip)+x[t,labels]
        if next_boundary<len(boundaries):
            frame,state=boundaries[next_boundary]
            if t<frame:
                nxt[state:]=-np.inf
            elif t==frame:
                nxt[:state]=-np.inf
                nxt[state+1:]=-np.inf
                next_boundary+=1
        dp=nxt
    score=float(max(dp[-2:]))
    if not math.isfinite(score):raise ValueError('NO_FINITE_CTC_PATH')
    return score

def evaluate_interval(emission,aligner,tokenizer,rows,left,right,
                      vocal_features,duration,blank):
    # left/right zero based anchor positions; -1 and len(rows) virtual.
    indexes=list(range(left+1,right))
    if not indexes:return {'status':'NO_INTERNAL_ROWS','rows':[]}
    wstart=max(0.,end_of(rows[left]) if left>=0 else 0.)
    wend=min(duration,start_of(rows[right]) if right<len(rows) else duration)
    interval={'left_anchor':left+1 if left>=0 else 0,
        'right_anchor':right+1 if right<len(rows) else len(rows)+1,
        'rows':[q+1 for q in indexes],
        'window':[round(wstart,4),round(wend,4)],'status':'PENDING'}
    if not math.isfinite(wstart) or not math.isfinite(wend) or wend<=wstart:
        interval['status']='INVALID_ANCHOR_WINDOW';return interval
    N=emission.shape[0]
    frame=lambda s:max(0,min(N,round(s/duration*N)))
    fst,fen=frame(wstart),frame(wend)
    if fen-fst<3:interval['status']='SHORT_EMISSION';return interval
    target_strings=[tokens_of(rows[j]) for j in indexes]
    dic=tokenizer.dictionary
    if any(not t or any(c not in dic for c in t) for t in target_strings):
        interval['status']='UNSUPPORTED_TOKEN';return interval
    target_ids=[[dic[c] for c in s] for s in target_strings]
    merged=[q for seq in target_ids for q in seq]
    sub=emission[fst:fen].contiguous()
    try:
        h1ll=viterbi_ctc_ll(sub,merged,blank)
        h1_spans=aligner(sub,tuple(target_ids))
    except Exception as exc:
        interval['status']='H1_ALIGN_ERROR';interval['reason']=str(exc)[:280];return interval
    if len(h1_spans)!=len(indexes) or any(not s for s in h1_spans):
        interval['status']='H1_INCOMPLETE';return interval
    h1_times=[(fst+float(s[0].start))*duration/N for s in h1_spans]
    h0_times=[start_of(rows[j]) for j in indexes]
    if any(t is None for t in h0_times):
        interval['status']='H0_NO_CLOCK';return interval
    # Hard-constrain first-token emission at every original global CTC
    # timestamp, *without* resetting CTC DP between lyric lines.
    offsets=[]
    counter=0
    for toks in target_ids:
        offsets.append(counter)
        counter+=len(toks)
    onset_frames={offset:(frame(clock)-fst) for offset,clock in zip(offsets,h0_times)}
    try:
        h0ll=viterbi_ctc_ll(sub,merged,blank,onset_frames)
        if h0ll>h1ll+0.0001:
            raise ValueError('H0_EXCEEDS_UNCONSTRAINED_H1')
    except Exception as exc:
        interval['status']='H0_CONSTRAINED_ERROR';interval['reason']=str(exc)[:260];return interval
    ft=np.asarray(vocal_features.frame_times,dtype=np.float64)
    pow=np.asarray(vocal_features.rms_db,dtype=np.float64)
    floor=float(np.median(pow[np.isfinite(pow)]))
    def coverage(times):
        stops=list(times[1:])+[wend]
        fractions=[]
        for s,e in zip(times,stops):
            mask=(ft>=s)&(ft<e)
            fractions.append(float(np.mean(pow[mask]>floor)) if np.any(mask) else 0.)
        return sum(fractions)/len(fractions)
    cov0=coverage(h0_times)
    cov1=coverage(h1_times)
    gain=(h1ll-h0ll)/len(indexes)
    interval.update({'status':'SCORED','LL_H0':round(h0ll,3),'LL_H1':round(h1ll,3),
        'LL_gain_per_line':round(gain,6),'coverage_H0':round(cov0,6),
        'coverage_H1':round(cov1,6),
        'h0':h0_times,'h1':h1_times,
        'switch':{str(x):(gain>=x and cov1>=cov0) for x in DELTA_THRESHOLDS}})
    return interval

def process_song(meta,report):
    stem=Path(report['ctc_audio_path'])
    if not stem.is_file():return {'status':'MISSING_VOCAL_STEM','stem':str(stem)}
    rows=report['assignments']
    features=al.analyze_vocal_onsets(stem,float(meta['duration']))
    anchors,diagnostics=choose_anchors(rows,features)
    print('ANCHOR_COUNT',len(anchors),'ROWS',len(rows),flush=True)
    bundle=torchaudio.pipelines.MMS_FA
    device='cuda' if torch.cuda.is_available() else 'cpu'
    waveform=ctc.decode_audio(stem)
    duration=float(waveform.shape[1])/ctc.SAMPLE_RATE
    model=bundle.get_model(with_star=False).to(device).eval()
    if device=='cuda':ctc.configure_cuda_cpu_feature_extractor(model)
    from contextlib import nullcontext
    if device=='cuda':
        from torch.nn.attention import SDPBackend,sdpa_kernel
        ctx=sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:ctx=nullcontext()
    with torch.inference_mode(),ctx:
        emissions,_=model(waveform if device=='cuda' else waveform.to(device))
    emission=emissions[0].cpu()
    tokenizer=bundle.get_tokenizer()
    aligner=bundle.get_aligner()
    blank=0
    edges=[-1]+anchors+[len(rows)]
    trials=[]
    for i,(left,right) in enumerate(zip(edges,edges[1:])):
        trial=evaluate_interval(emission,aligner,tokenizer,rows,left,right,features,
                                duration,blank)
        trial['id']=i
        trials.append(trial)
    results={str(delta):[start_of(row) for row in rows] for delta in DELTA_THRESHOLDS}
    marked={str(delta):[False]*len(rows) for delta in DELTA_THRESHOLDS}
    for t in trials:
        for delta in DELTA_THRESHOLDS:
            key=str(delta)
            switch=t.get('status')=='SCORED' and t['switch'][key]
            for n,j in enumerate([x-1 for x in t['rows']]):
                if switch:results[key][j]=t['h1'][n]
                else:marked[key][j]=True
    return {'status':'FROZEN_H0_H1','device':device,'stem':str(stem),
        'duration':duration,'num_rows':len(rows),'anchors':[i+1 for i in anchors],
        'anchor_diagnostics':diagnostics,'intervals':trials,
        'delta_results':results,'delta_review':marked,
        'thresholds':list(DELTA_THRESHOLDS),
        'BK_was_accessed':False,
        'emission_frame_count':int(emission.shape[0])}

if __name__=='__main__':
    meta=Path(sys.argv[1]);out=Path(sys.argv[2])
    inp=json.loads(meta.read_text(encoding='utf8'))
    value=process_song(inp['meta'],inp['report'])
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print('ANCHOR_GPU_DONE',value['status'],'INTERVALS',len(value.get('intervals',[])),flush=True)
