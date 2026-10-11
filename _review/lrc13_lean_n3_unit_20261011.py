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
print('N3_UNIT_6_RANDOM_FROZEN_SMIN0_SCORE_AND_TRACE_PASS')
