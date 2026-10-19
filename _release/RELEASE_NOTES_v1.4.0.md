## LRC Timeline Aligner v1.4.0

Usability update, algorithm unchanged.

### Timing results unchanged from v1.3.1

Measured on the same 39-song / 1487-line manually checked validation pool:

- PCO@0.2: **1362 / 1487 (91.6%)**
- Lines off by more than 0.5 s: **59**
- Lines off by more than 1 s: **40**, all review-flagged
- TRUSTED lines off by more than 1 s: **0**

### What's new in v1.4

- Review page opens automatically after a batch alignment that produced review rows; `LRC_NO_REVIEW=1` disables it; install paths with spaces are supported.
- Candidates can be listened to and applied by click or digit key; each chip shows how many other candidates agree within ±0.15 s; overflow candidates sit in a collapsible list.
- A/B playback alternates the current and the selected candidate timestamps.
- Vocal-stem waveform strip for the selected row with drag adjustment, plus visible ±0.1 s nudges and JIS-safe bracket keys.
- Plain-language reason text for every review row.
- Transactional save: the LRC, the first-save `.auto.lrc` and the edit log are staged and published together; a failed save leaves the files unchanged.
- Timing and alignment algorithm unchanged from v1.3.1 (39-song cache-only replay byte-identical).

### Install

In the existing v1.3.1 folder, overwrite the repository files with this release. The requirements and `setup.ps1` are unchanged; setup does not need to be rerun.
