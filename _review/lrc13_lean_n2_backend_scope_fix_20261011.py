"""N2 second minimal correction: do not evaluate non-CTC as final backends.
WhisperX/hybrid remain generation-level diagnostics, never final candidates.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'scripts'/'auto_lrc.py'
s=p.read_text(encoding='utf8')
old='''    preview_records: list[dict[str, object]] = []
    shared_retry_helper_cache: dict[str, tuple[str, object]] = {}
'''
new='''    if str(getattr(args, "timing_source_policy", "default")) == "ctc-first":
        # CTC is the single permitted final backend. All earlier ASR/WX
        # acquisition and CTC diagnostic flags stay intact; the opt-in policy
        # only prevents selecting a non-CTC backend's timestamp authority.
        generation_candidates = [
            item for item in generation_candidates
            if item.get("backend") == "ctc"
        ]
        if not generation_candidates:
            raise LrcError("ctc-first requires a CTC final backend")
    preview_records: list[dict[str, object]] = []
    shared_retry_helper_cache: dict[str, tuple[str, object]] = {}
'''
if s.count(old)!=1:raise ValueError('N2_BACKEND_SCOPE_CONFLICT '+str(s.count(old)))
t=s.replace(old,new,1)
if p.read_text(encoding='utf8')!=s:raise ValueError('N2_SOURCE_CONCURRENT_CHANGED')
p.write_text(t,encoding='utf8')
print('N2_BACKEND_CTC_ONLY_OPTOUT_SAFE_PATCHED')
