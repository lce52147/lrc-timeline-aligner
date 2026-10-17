## LRC Timeline Aligner v1.3.1

v1.3.1 is an installation and packaging update. The v1.3 timing and alignment algorithm is unchanged.

### Timing results unchanged from v1.3.0

Measured on the same 39-song / 1487-line manually checked validation pool:

- PCO@0.2: **1362 / 1487 (91.6%)**
- Lines off by more than 0.5 s: **59**
- Lines off by more than 1 s: **40**, all review-flagged
- TRUSTED lines off by more than 1 s: **0**

### Installation changes

- Added a one-command `setup.ps1` flow for the repository-local `.venv-asr`.
- Pinned the validated ASR dependency set and kept CUDA/CPU PyTorch selection explicit.
- Pinned repository-local ffmpeg 8.1.2, whisper.cpp `b5454`, and the `ggml-large-v3` model identity.
- `align-lrc.ps1` now prefers the repository venv and repository-local ffmpeg.
- `-Prefetch` acquires the production/reviewer model assets and `doctor.py` verifies the runtime before use.
- Model identity lookup is deterministic when Hugging Face caches contain multiple snapshots or both safetensors and legacy PyTorch weights.

### Install

1. Download and unzip `lrc-timeline-aligner-v1.3.1.zip`.
2. In the extracted folder, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Prefetch`.
3. Put `Song.flac` and `Song.lyrics.txt` (or `.lyrics.lrc`) together and drag the FLAC onto `Align LRC.bat`.

The clean-install validation host used Python 3.11.5 and an NVIDIA GeForce RTX 4070 Ti with 12,282 MiB VRAM. The post-prefetch validation folder used 37.12 GiB excluding copied test songs; the measured third clean prefetch/install attempt took 2224.481 seconds (37 min 4 s) to reach the doctor gate. Peak VRAM was not instrumented.

Known limitation: rows where all acoustic models fail together, including very weak vocals, long instrumental gaps, and some repeated choruses, can still have large timing error. Those rows are intended to surface as REVIEW rather than silently trusted output.
