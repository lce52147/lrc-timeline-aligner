"""Conservative independent raw ASR-text evidence for large skeleton clock moves.

Research evidence only: raw Whisper.cpp is used by the DP too, hence this
metric is correlated with selection. Never use BK or synthetic per-line ASR
timestamps here. The output is an auditable reason for atomic rollback.
"""
from __future__ import annotations
from collections import Counter
from typing import Callable,Sequence
import math
from segment_skeleton import _text_for_match,_dice_bigram

MATERIAL_CTC_CLOCK_MOVE_S=.20
MATERIAL_ASR_TEXT_GAIN=.25

def check_asr_corroboration(*,lyric_texts:Sequence[str],
                            global_times:Sequence[float],
                            skeleton_times:Sequence[float],
                            raw_asr_events:Sequence[tuple[float,float,str]],
                            romanizer:Callable|None=None)->dict:
    if not len(lyric_texts)==len(global_times)==len(skeleton_times):
        return {'status':'INVALID','fallback':True,'reason':'ROW_COUNT'}
    if not raw_asr_events:
        return {'status':'UNAVAILABLE','fallback':False,'reason':'NO_PARAGRAPH_EVENTS'}
    events=[]
    for start,end,text in raw_asr_events:
        if (math.isfinite(start) and math.isfinite(end)
                and end>start and text):
            normalized=_text_for_match(text,romanizer)
            if normalized:events.append((float(start),float(end),normalized))
    if not events:
        return {'status':'INVALID','fallback':True,'reason':'NO_USABLE_TRANSCRIPT'}
    def support(phrase:str,t:float)->float:
        return max((_dice_bigram(phrase,raw)
                   for start,end,raw in events if start-1.<=t<=end+1.),
                   default=0.)
    changed=[]
    for i,(line,old,new) in enumerate(zip(lyric_texts,global_times,skeleton_times),1):
        if abs(float(old)-float(new))<MATERIAL_CTC_CLOCK_MOVE_S:continue
        text=_text_for_match(line,romanizer)
        if not text:continue
        before=support(text,float(old))
        after=support(text,float(new))
        changed.append({'entry':i,'global':round(before,5),'skeleton':round(after,5),
                        'gain':round(after-before,5),
                        'delta_s':round(float(new)-float(old),3)})
    if not changed:
        return {'status':'NO_MATERIAL_MOVE','fallback':False,'changed':[]}
    mean_gain=sum(c['gain'] for c in changed)/len(changed)
    return {'status':'TESTED','changed':changed,
            'mean_gain':round(mean_gain,5),
            'minimum_material_gain':MATERIAL_ASR_TEXT_GAIN,
            'fallback':mean_gain<MATERIAL_ASR_TEXT_GAIN,
            'reason':('INSUFFICIENT_ASR_CORROBORATION'
                      if mean_gain<MATERIAL_ASR_TEXT_GAIN else 'CORROBORATED')}
