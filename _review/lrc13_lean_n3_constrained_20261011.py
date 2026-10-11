"""N3 exact whole-song CTC Viterbi with per-line minimum onset-to-next-onset frames.

Joint DP over CTC label-state and saturating line age. Hard duration is enforced
on the transition to the next lyric's FIRST emitted token. Never post-shift.
Checkpointed traceback uses bounded memory; no BK inputs or timing evaluations.
"""
from pathlib import Path
import numpy as np
import sys,json,math,time,traceback
V=Path(__file__).resolve().parent
F=V/'LRC13_LEAN_N3_GPU_20261011'
CORPUS=json.loads((F/'MANIFEST.json').read_text(encoding='utf8'))
assert CORPUS['status']=='ZERO_PARITY_39_COMPLETE'
DELTA=(0.10,0.12,0.15)
OUT=V/'LRC13_LEAN_N3_CONSTRAINED_20261011';OUT.mkdir(exist_ok=True)
MANIFEST=OUT/'MANIFEST.json'
cases={x['index']:x for x in json.loads(MANIFEST.read_text(encoding='utf8'))['cases'] if x.get('status')=='DONE'} if MANIFEST.is_file() else {}
def emit(phase,i=0):
 MANIFEST.write_text(json.dumps({'status':phase,'completed':len(cases),'total':39,'cases':list(cases.values()),
   's_min':DELTA,'BK_not_accessed':True,'algorithm':'EXACT_CTC_VITERBI_PER_LINE_AGE_CONSTRAINT'},ensure_ascii=False,indent=2),encoding='utf8')
 print('N3_DP',phase,'done',len(cases),'index',i,flush=True)

def plan(romans,ctc_dictionary,syl,minimum,spf):
 lines=[]
 for i,(roman,n) in enumerate(zip(romans,syl)):
  assert roman and all(c in ctc_dictionary for c in roman)
  cap=math.ceil(float(n)*minimum/spf-1e-10) if i<len(romans)-1 else 0
  # Age saturation can occur at cap; all transitions to next line require age >= cap.
  # cap=0 means any age sufficient (for last lyric irrelevant).
  ids=np.empty(2*len(roman),dtype=np.int32)
  ids[0::2]=[ctc_dictionary[c] for c in roman]
  ids[1::2]=0
  skip=np.zeros(len(ids),dtype=np.bool_)
  for j in range(2,len(ids),2):
   skip[j]=roman[j//2]!=roman[j//2-1]
  lines.append((cap,ids,skip))
 return lines

def step(logp,previous,lead,plans,pointers=None,step_index=0):
 """Advance all CTC augmented states one acoustic frame and optionally record parents."""
 updated=[]
 for line,(cap,ids,skip) in enumerate(plans):
  old=previous[line];m=len(ids); C=cap+1
  # Same-state, token-advance and legal token-skip. Onset duration accumulates.
  best=old.copy()
  chosen=np.zeros((m,C),dtype=np.uint8)
  advance=old[:-1,:]
  better=advance>best[1:,:]
  best[1:,:]=np.maximum(best[1:,:],advance)
  chosen[1:,:]=np.where(better,1,chosen[1:,:])
  for state in range(2,m,2):
   if skip[state]:
    better=old[state-2,:]>best[state,:]
    best[state,:]=np.maximum(best[state,:],old[state-2,:])
    chosen[state,:]=np.where(better,2,chosen[state,:])
  nxt=np.full((m,C),-np.inf,dtype=np.float32)
  point=np.full((m,C),255,dtype=np.uint8) if pointers is not None else None
  if cap:
   nxt[:,1:]=best[:,:-1]
   if point is not None:point[:,1:]=chosen[:,:-1]
   over=best[:,cap]>best[:,cap-1]
   nxt[:,cap]=np.maximum(best[:,cap],best[:,cap-1])
   if point is not None:point[:,cap]=np.where(over,chosen[:,cap]+4,chosen[:,cap-1])
  else:
   nxt[:,0]=best[:,0]
   if point is not None:point[:,0]=chosen[:,0]
  if line==0:
   incoming=lead
   typ=5   # from global prefix blank
  else:
   prior=previous[line-1]
   prev_final_blank=float(prior[-1,-1])
   prev_final_token=float(prior[-2,-1]) if ids[0]!=plans[line-1][1][-2] else -np.inf
   if prev_final_blank>=prev_final_token:
    incoming=prev_final_blank;typ=3
   else:incoming=prev_final_token;typ=4
  if incoming>nxt[0,0]:
   nxt[0,0]=incoming
   if point is not None:point[0,0]=typ
  # Emission for this state when finite (same model/log-prob for H0 and constrained).
  nxt+=logp[ids,None]
  updated.append(nxt)
  if pointers is not None:pointers[line][step_index,:,:]=point
 return updated,lead+float(logp[0])

def forward(logprob,plans,checkpoint=180):
 T=len(logprob)
 previous=[np.full((len(ids),cap+1),-np.inf,dtype=np.float32) for cap,ids,_ in plans]
 lead=0.
 checkpoints={-1:(previous,lead)}
 for t in range(T):
  previous,lead=step(logprob[t],previous,lead,plans)
  if t%checkpoint==checkpoint-1:checkpoints[t]=(previous,lead)
 score=float(max(previous[-1][-2:].max(axis=1)))
 if not math.isfinite(score):raise ValueError('N3_NO_FINITE_PATH')
 return score,checkpoints,previous,lead

def traceback(logprob,plans,checkpoints,final_states,final_lead,checkpoint=180):
 T=len(logprob)
 term=final_states[-1]
 flat_index=int(np.argmax(term[-2:].reshape(-1)))
 state=len(term)-2+flat_index//term.shape[1]
 age=flat_index%term.shape[1]
 row=len(plans)-1
 start_times=[None]*len(plans)
 target=T-1
 while target>=0:
  previous_endpoint=(target//checkpoint)*checkpoint-1
  initial,lead=checkpoints[previous_endpoint]
  length=target-previous_endpoint
  pointers=[np.empty((length,len(ids),cap+1),dtype=np.uint8) for cap,ids,_ in plans]
  states=[x.copy() for x in initial]
  for frame in range(previous_endpoint+1,target+1):
   states,lead=step(logprob[frame],states,lead,plans,pointers,frame-previous_endpoint-1)
  for frame in range(target,previous_endpoint,-1):
   kind=int(pointers[row][frame-previous_endpoint-1,state,age])
   if kind==255:raise ValueError(f'N3_BACKTRACE_DEAD_STATE frame={frame} row={row} state={state} age={age}')
   if kind in (3,4,5):
    assert state==0 and age==0,(row,state,age,kind)
    start_times[row]=frame
    if kind==5:
     assert row==0
     return [float(x) for x in start_times]
    row-=1
    age=plans[row][0] # previous line is mature at this transition
    state=len(plans[row][1])-(1 if kind==3 else 2)
   else:
    from_cap=kind>=4
    move=kind%4
    assert move in (0,1,2)
    state-=move
    age=age if from_cap else max(0,age-1)
  target=previous_endpoint
 raise ValueError('N3_BACKTRACE_NOT_REACHED_FIRST_ONSET')

def run_case(i,job):
 leaf=f'{i:02d}_{job["id"]}'
 output=OUT/leaf
 output.mkdir(exist_ok=True)
 file=output/'FROZEN_N3.json'
 if file.exists():raise RuntimeError('REFUSE_UNMANIFESTED_FROZEN_N3 '+str(file))
 logprob=np.load(job['emission'],mmap_mode='r',allow_pickle=False)
 from torchaudio.pipelines import MMS_FA
 tokenizer=MMS_FA.get_tokenizer()
 dictionary=tokenizer.dictionary
 S=job['zero_starts'];spf=job['seconds_per_frame']
 results={}
 for threshold in DELTA:
  key=str(threshold)
  plans=plan(job['romans'],dictionary,job['syllables'],threshold,spf)
  violation=[j+1 for j,(a,b) in enumerate(zip(S,S[1:])) if b-a<math.ceil(job['syllables'][j]*threshold/spf-1e-10)*spf-1e-9]
  if not violation:
   results[key]={'status':'EXACT_BASELINE_FEASIBLE_GLOBAL_VITERBI_OPTIMUM',
      'starts':S,'violating_rows_before':[], 'algorithm_exact':True}
   continue
  print('N3_DP_SOLVE',i,key,'violations',violation[:12],'states',sum(len(ids)*(cap+1) for cap,ids,_ in plans),flush=True)
  t=time.monotonic()
  score,checkpoints,finalstates,lead=forward(logprob,plans)
  frames=traceback(logprob,plans,checkpoints,finalstates,lead)
  starts=[round(f*spf,6) for f in frames]
  assert len(starts)==len(S)
  for j,(a,b) in enumerate(zip(starts,starts[1:])):
   if b-a+spf*.51<job['syllables'][j]*threshold:
    raise ValueError(f'N3_DURATION_VIOLATION {i} {key} {j+1} {a} {b}')
  results[key]={'status':'CONSTRAINED_VITERBI_EXACT','starts':starts,
   'score':score,'violating_rows_before':violation,'elapsed_s':round(time.monotonic()-t,2),
   'algorithm_exact':True}
  del checkpoints
 payload={'status':'N3_EXACT_FROZEN','index':i,'id':job['id'],'BK_read':False,
   'flagged':job['flagged'],'zero_parity':True,'H0_starts':S,
   'threshold_results':results,'seconds_per_frame':spf}
 file.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
 return {'index':i,'id':job['id'],'flagged':job['flagged'],'status':'DONE','output':str(file)}

if __name__=='__main__':
 ids=[int(sys.argv[1])] if len(sys.argv)>1 else [r['index'] for r in CORPUS['cases']]
 for job in CORPUS['cases']:
  i=job['index']
  if i not in ids or i in cases:continue
  print('N3_DP_BEGIN',i,'flagged',job['flagged'],flush=True)
  c=run_case(i,job)
  cases[i]=c
  emit('IN_PROGRESS',i)
 if len(cases)==39:emit('N3_EXACT_39_FROZEN')
