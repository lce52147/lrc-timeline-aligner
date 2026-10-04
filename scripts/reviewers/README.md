# Reviewer timing generators

These runners are the reusable versions of the Stage 1 / Stage H evidence generators. They accept an arbitrary producer-times JSON and/or a repeatable `--song-id` subset instead of assuming the 2026-10-02 dev split.

## Runtime mapping

- `run_xlsr.py`, `run_hub.py`, `run_whisperx.py`: `.venv-asr\Scripts\python.exe`
- `build_hubp_plan.py`: `.venv-hfa\python.exe` (OpenJTalk planning only)
- `run_hubp.py`: `.venv-asr\Scripts\python.exe` (CUDA torch inference)
- `build_hfa_plan.py`: `.venv-asr\Scripts\python.exe` (`ctc_align` / pykakasi planning only)
- `run_hfa.py`: `.venv-hfa\python.exe` (ONNX Runtime; default `--provider cuda`)

`--song-id` may be repeated. If it is omitted, all songs present in the supplied input are selected. Model runners default to CUDA; CPU is available only as an explicit fallback where the upstream task permits it.

XLSR, HUB and WhisperX write their observations into the supplied producer-times JSON, matching the Stage 1 schema. HUBP and HFA keep the Stage H plan/raw-sidecar schema and therefore take explicit `--plan` / `--output` paths.

The generator scripts do not read reference timing files and do not change LRC output timestamps. Their output is reviewer evidence for the independent reviewer layer.
