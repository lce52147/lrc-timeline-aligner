"""Single minimal guarded patch to scripts/auto_lrc.py for N2 policy (off by default).
Source edits only: one new CLI switch and one Central candidate eligibility guard.
Never alters audio, BK, OOGG or non-project files.
"""
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts'/'auto_lrc.py'
src=p.read_text(encoding='utf8')
original=src
def replace_once(old,new):
 global src
 n=src.count(old)
 if n!=1:raise ValueError('N2_SCOPE_CONFLICT '+str(n)+' '+repr(old[:90]))
 src=src.replace(old,new,1)
replace_once(
'def evaluate_backend_timing(\n    backend: str,',
'''def apply_central_timing_source_policy(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    policy: Literal["default", "ctc-first"] = "default",
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Only remove Whisper/RAW from FINAL timing eligibility.

    The candidates are still collected as diagnostics and review evidence.
    Default path is strictly identity-preserving (no copy, sort or mutation).
    Matches B11 classification: source contains "whisper" or "raw".
    """
    if policy == "default":
        return candidate_sets
    if policy != "ctc-first":
        raise ValueError(f"unsupported timing-source policy: {policy}")
    kept_rows = []
    for line, candidates in enumerate(candidate_sets, 1):
        kept = tuple(
            c for c in candidates
            if "whisper" not in c.source.lower()
            and "raw" not in c.source.lower()
        )
        if not kept:
            raise LrcError(f"ctc-first: no eligible time candidate at line {line}")
        kept_rows.append(kept)
    return tuple(kept_rows)


def evaluate_backend_timing(
    backend: str,''')
replace_once(
'    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",\n) -> EvaluatedBackendTiming:',
'    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",\n    timing_source_policy: Literal["default", "ctc-first"] = "default",\n) -> EvaluatedBackendTiming:')
replace_once(
'    if str(report.get("backend") or "") != backend:\n        raise LrcError("Backend identity does not match central evaluation report")\n    report_projection_revision = timing_report_evaluation_revision(report)',
'''    if str(report.get("backend") or "") != backend:
        raise LrcError("Backend identity does not match central evaluation report")
    candidate_sets = apply_central_timing_source_policy(
        candidate_sets, timing_source_policy
    )
    report_projection_revision = timing_report_evaluation_revision(report)''')
replace_once(
'    parser.add_argument(\n        "--reviewer-evidence",',
'''    parser.add_argument(
        "--timing-source-policy", choices=("default", "ctc-first"),
        default="default",
        help="N2 opt-in: Whisper/RAW remains diagnostic evidence but cannot select final timing.",
    )
    parser.add_argument(
        "--reviewer-evidence",''')
# First finalizing path has explicit arbiter mode.
final_start=src.index('def finalize_production_timing(')
preview_start=src.index('def prepare_backend_timing_preview(')
end=src.index('def assignment_has_timing(',preview_start)
final_part=src[final_start:preview_start]
call='        arbiter_mode=str(getattr(args, "arbiter", "off")),\n'
assert final_part.count(call)==1
final_part=final_part.replace(call,call+'        timing_source_policy=str(getattr(args, "timing_source_policy", "default")),\n',1)
preview=src[preview_start:end]
snippet='    evaluated = evaluate_backend_timing(\n'
assert preview.count(snippet)==3
preview=preview.replace('        diagnostics,\n    )\n    fixed_point_rounds:', '        diagnostics,\n        timing_source_policy=str(getattr(args, "timing_source_policy", "default")),\n    )\n    fixed_point_rounds:',1)
assert preview.count('        evaluated = evaluate_backend_timing(\n')==2
preview=preview.replace('            diagnostics,\n        )\n    arbiter_mode =', '            diagnostics,\n            timing_source_policy=str(getattr(args, "timing_source_policy", "default")),\n        )\n    arbiter_mode =',1)
preview=preview.replace('            arbiter_mode=arbiter_mode,\n        )\n    preview_report[', '            arbiter_mode=arbiter_mode,\n            timing_source_policy=str(getattr(args, "timing_source_policy", "default")),\n        )\n    preview_report[',1)
assert preview.count('timing_source_policy=str(getattr(args, "timing_source_policy", "default"))')==3
src=src[:final_start]+final_part+preview+src[end:]
assert original!=src
# Fail if another parallel edit changed source; replace exact content only.
if p.read_text(encoding='utf8')!=original:raise ValueError('N2_SOURCE_CHANGED_DURING_PATCH')
p.write_text(src,encoding='utf8')
print('N2_PATCH_OK',len(original),len(src),'exact_scope=Central_eligibility_and_new_default_off_CLI')