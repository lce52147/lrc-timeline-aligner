import sys,json,math
from pathlib import Path
import numpy as np
import torch,torchaudio
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lrc13_lean_n3_constrained_20261011 import plan,forward,traceback
for seed in range(6):
 rng=np.random.default_rng(seed)
 texts=['aba','aba','cc','da']
 dic={'a':1,'b':2,'c':3,'d':4}
 p=plan(texts,dic,[1,1,1,1],0.0,.02)
 emissions=rng.normal(size=(70,5)).astype(np.float32)
 emissions=emissions-np.log(np.exp(emissions).sum(axis=1))[:,None]
 import warnings
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  _,s=torchaudio.functional.forced_align(
   torch.tensor(emissions).unsqueeze(0),
   torch.tensor([list(map(dic.get,''.join(texts)))],dtype=torch.int32),blank=0)
 expected=float(s.sum())
 actual,check,state,lead=forward(emissions,p,checkpoint=13)
 starts=traceback(emissions,p,check,state,lead,checkpoint=13)
 assert abs(expected-actual)<1e-3,(seed,expected,actual)
 assert len(starts)==len(texts) and all(a<b for a,b in zip(starts,starts[1:])),starts
for seed in range(3):
 rng=np.random.default_rng(80+seed)
 em=rng.normal(size=(70,5)).astype(np.float32)
 em=em-np.log(np.exp(em).sum(axis=1))[:,None]
 p0=plan(['ab','cd'],dic,[2,1],0,.02)
 p1=plan(['ab','cd'],dic,[2,1],.10,.02)
 base,_,_,_=forward(em,p0,checkpoint=13)
 fixed,ck,dp,lead=forward(em,p1,checkpoint=13)
 times=traceback(em,p1,ck,dp,lead,checkpoint=13)
 assert len(times)==2 and times[1]-times[0]>=10,(seed,times)
 assert fixed<=base+1e-4,(seed,base,fixed)
print('N3_UNIT_6_ZERO_PLUS_3_CONSTRAINED_EXACT_DP_PASS')