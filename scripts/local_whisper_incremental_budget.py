"""Bound incrementally new local Whisper evidence jobs during Central replay.

A first production observation pass is never limited. On later fixed-point
recollections, cached request identities remain intact while a deterministic,
song-wide spread of previously unseen requests is used. Defer means NO
evidence; nothing is imputed from an earlier or nearby candidate.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable,Mapping,Sequence,TypeVar

T=TypeVar("T")


def budget_incremental_whisper_requests(
    requests: Sequence[T],
    cache: Mapping[str,object],
    cache_key: Callable[[T],str],
    *,
    maximum_new: int=32,
) -> tuple[tuple[T,...],int]:
    """Preserve all genuine cache hits; defer excess previously unseen work."""
    if maximum_new<1:raise ValueError("maximum_new must be positive")
    original=tuple(requests)
    if not any(isinstance(k,str) and k.startswith("local-whisper:")
               for k in cache):
        return original,0
    seen_fresh=set()
    fresh=[]
    for item in original:
        key=cache_key(item)
        if key in cache or key in seen_fresh:continue
        seen_fresh.add(key)
        fresh.append(item)
    if len(fresh)<=maximum_new:
        return original,0
    grouped:dict[int,list[T]]=defaultdict(list)
    for item in fresh:
        grouped[int(getattr(item,"entry_index"))].append(item)
    for group in grouped.values():
        group.sort(key=lambda r:(
            len(getattr(r,"context_entry_indexes")),
            float(getattr(r,"window_end"))-float(getattr(r,"window_start")),
            str(getattr(r,"request_revision")),
        ))
    indexes=sorted(grouped)
    if len(indexes)>maximum_new:
        # Song-wide row coverage, not first-N bias (which drops late choruses).
        picks=[indexes[min(len(indexes)-1,
            int((i+.5)*len(indexes)/maximum_new))]
               for i in range(maximum_new)]
        selected_fresh=[grouped[i][0] for i in picks]
    else:
        selected_fresh=[grouped[i][0] for i in indexes]
        remaining=max(0,maximum_new-len(selected_fresh))
        level=1
        while remaining:
            next_level=[grouped[i][level] for i in indexes
                        if level<len(grouped[i])]
            if not next_level:break
            selected_fresh.extend(next_level[:remaining])
            remaining=maximum_new-len(selected_fresh)
            level+=1
    selected_keys={cache_key(item) for item in selected_fresh}
    kept=tuple(item for item in original if cache_key(item) in cache
               or cache_key(item) in selected_keys)
    return kept,len(fresh)-len(selected_fresh)
