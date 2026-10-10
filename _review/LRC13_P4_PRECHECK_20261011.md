# P4 precheck (read only; P5 fresh validation still pending)

The P4 prompt claims `segment_skeleton_production.py` uses a novel fixed mapping for `score` that differs from whole-song CTC.

Source inspection disproves the **difference between paths**:
- `scripts/auto_lrc.py` in `run_ctc_alignment`, around lines 8647–8652:
  `confidence = 0.90 if ctc_score >= 0.08 else 0.70 if ctc_score >= 0.03 else 0.35`
  then `score = confidence if start is not None else 0.0`.
- `scripts/segment_skeleton_production.py` around line 225:
  identical mapping for local CTC scores.

Thus P4 can unify these paths in a shared conversion function for maintainability, but it cannot honestly claim the mapping was segment-only, nor predict that replacing it with the **same** formula will lower Central trusted counts. The semantics of CTC confidence and global acoustic evidence must be analyzed separately; do not substitute a new heuristic without cross-song validation.

No production script changes were made for P4. P5 must finish its own 39-song skeleton-level check first.
