# Benchmark: 01. 雨模様

Reference:

- Audio: `D:\MusicLibrary\Music\TUYU\ツユ - 雨模様\01. 雨模様.flac`
- Checked LRC: `D:\MusicLibrary\01. 雨模様.lrc`
- Generated checked copy: `scripts\01. 雨模様.lrc`
- Verifier: `scripts\evaluate_lrc.py`

## Acceptance Goal

For an un-timed lyric source, a backend should not become the default unless it
can plausibly reach the product target:

- 90-95% of timing entries usable without manual correction
- only a small number of lines needing review
- no known path that silently produces a low-accuracy LRC while reporting success

For this benchmark, a practical gate is:

- at least 95% of entries within +/-0.50s against the checked LRC
- no text or bilingual grouping mismatches

## Verified Template Mode

Using `--timing-source lyrics` with the checked LRC as input:

- entries: 49 / 49
- display lines: 98 / 98
- text mismatches: 0
- metadata equal: true
- max abs delta: 0 cs
- within +/-0.50s: 100%

This proves template preservation works, but it does not prove automatic
alignment.

## Failed Heuristic Baseline

The original energy/onset heuristic was tested against the checked LRC:

- entries: 49 / 49
- display lines: 98 / 98
- text mismatches: 0
- max timing error: about 10.56s
- all 49 timing entries differed from the checked LRC

Decision: heuristic timing is experimental only. It must not be the default.

## whisper.cpp Probe

Installed local backend:

- whisper.cpp release: v1.8.6 CUDA/cuBLAS 12.4 x64
- CLI: `tools\whisper.cpp\Release\whisper-cli.exe`
- ASR models:
  - `models\whisper.cpp\ggml-large-v3-turbo.bin`
  - `models\whisper.cpp\ggml-large-v3.bin`
- VAD model: `models\whisper.cpp\ggml-silero-v5.1.2.bin`

Runtime result:

- The 162.75s FLAC transcribed in about 19.6s on RTX 4070 Ti.
- Windows Unicode paths failed in `whisper-cli`; decoding the FLAC to an ASCII
  temp WAV fixes this.
- ASR transcription is usable as an anchor source, but it contains lyric errors
  and missed short repeated lines.

large-v3-turbo segment-to-lyric matching:

- ASR segments: 48
- reference timing entries: 49
- within +/-0.50s: about 63%
- within +/-1.00s: about 73%

Token-stream fuzzy matching:

- within +/-0.50s: about 63%
- within +/-1.00s: about 82%

full large-v3 segment-to-lyric matching:

- ASR segments: 47
- matched lyric entries: 46 / 49
- matched percent: 93.88%
- low-confidence entries: 7
- within +/-0.10s: 38.78%
- within +/-0.25s: 63.27%
- within +/-0.50s: 73.47%
- within +/-1.00s: 85.71%
- max abs timing delta: 3.21s
- mean abs timing delta: 0.46s

Decision: full large-v3 is a better ASR anchor than turbo and should be the
current whisper.cpp experimental default, but it still does not meet the 90-95%
timing target by itself.

Generic onset splitting after ASR grouping:

- within +/-0.50s: about 4%
- within +/-1.00s: about 31%
- failure mode: generic onset peaks often pick instruments or drums instead of
  sung consonant/vowel starts.

whisper.cpp `--vad` with Silero:

- detected only the final ~11s of this song
- missed most singing sections
- decision: generic speech VAD is not reliable for this music benchmark.

## Current Decision

The tool now preserves checked timestamps in default `auto` mode. For un-timed
lyrics, `auto` first searches the Music library for a same-stem checked LRC,
verifies that the sung-text order matches, and reuses those checked timings.
This is the preferred production path when a checked LRC already exists.

Verified checked-hint results:

- `scripts\rain_auto_checked_hint.lrc`: 49 / 49 entries, 0 text mismatches,
  max timing delta 0 cs.
- `scripts\utopia_auto_checked_hint.lrc`: 32 / 32 sung-text entries,
  0 text mismatches, max timing delta 0 cs.
- `scripts\oyasumi_auto_checked_hint.lrc`: 40 / 40 sung-text entries,
  0 text mismatches, max timing delta 0 cs.

If no checked LRC hint is available, `auto` uses the local `whisperx` hybrid
backend when whisper.cpp, large-v3, and `.venv-asr` WhisperX are present. Users
can still run:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource heuristic "D:\Music\Song.flac"
```

but that path is explicitly marked experimental.

## WhisperX Hybrid Probe

Backend:

- whisper.cpp full large-v3 creates ordered ASR anchors.
- WhisperX Japanese alignment refines the ASR segments.
- WhisperX runs with device `auto`; on the current RTX 4070 Ti machine this
  resolves to `cuda` after installing `torch==2.8.0+cu128`.
- A monotonic lyric matcher maps prepared lyric entries back onto the ASR
  anchors.
- A selective lyric forced-alignment pass corrects late short lines and a small
  set of line-spanning segment errors.

Verified result for `scripts\rain_audio_gpu10.lrc`:

- entries: 49 / 49
- display lines: 98 / 98
- text mismatches: 0
- max abs timing delta: 49 cs
- mean abs timing delta: 15.61 cs
- within +/-0.10s: 48.98%
- within +/-0.25s: 75.51%
- within +/-0.50s: 100%
- within +/-1.00s: 100%
- report device: `cuda`

Remaining entries outside +/-0.25s:

- entry 3 `梅雨の季節に おいでませ`: -0.42s
- entry 10 `雨模様`: -0.38s
- entry 13 `疼いてドクドク脈打って`: +0.29s
- entry 21 `夏よ早く おいでませ`: +0.49s
- entry 38 `駄目な心は溺れて`: -0.34s
- entry 39 `梅雨の匂いが`: +0.37s
- entry 40 `トゲみたいに襲って`: +0.36s
- entry 43 `最後にするよ`: +0.26s

Decision: this is the current best local audio-only backend. It remains
experimental and sits behind the checked-LRC hint in default `auto` mode because
the checked hint can be exact while the audio-only path still has song-specific
residual errors.

## Second Benchmark: 11.ユートピア

Reference:

- Audio: `D:\MusicLibrary\Music\Islet\magic\11.ユートピア.flac`
- Checked LRC: `D:\MusicLibrary\11.ユートピア.lrc`
- Untimed lyric source: `scripts\utopia_untimed.lyrics.txt`
- Generated LRC: `scripts\utopia_auto_whisperx_hybrid.lrc`

Policy:

- Non-lyric marker entries such as `♪` and `(Outro)` are skipped in un-timed
  audio alignment.
- Evaluation uses `scripts\evaluate_lrc.py --ignore-markers` so only sung text
  is scored.

Verified sung-text result:

- entries: 32 / 32
- display lines: 64 / 64
- text mismatches: 0
- max abs timing delta: 47 cs
- mean abs timing delta: 16.78 cs
- within +/-0.10s: 46.88%
- within +/-0.25s: 75.00%
- within +/-0.50s: 100%
- within +/-1.00s: 100%
- report device: `cuda`

## Third Benchmark: 01.おやすみモノクローム

Reference:

- Audio: `D:\MusicLibrary\Music\Dreamin' Her - Example Original Soundtrack\01.おやすみモノクローム.flac`
- Checked LRC: `D:\MusicLibrary\01.おやすみモノクローム.lrc`
- Untimed lyric source: `scripts\oyasumi_monochrome_untimed.lyrics.txt`
- Generated LRC: `scripts\oyasumi_audio_gpu10.lrc`

Verified sung-text result:

- entries: 40 / 40
- display lines: 80 / 80
- text mismatches: 0
- max abs timing delta: 50 cs
- mean abs timing delta: 14.07 cs
- within +/-0.10s: 60.00%
- within +/-0.25s: 82.50%
- within +/-0.50s: 100%
- within +/-1.00s: 100%
- report device: `cuda`

This result uses normal/SNS candidate fusion, selective same-segment forced
alignment, and a narrow acoustic late-onset refinement for cases where ASR and
forced alignment both start early.

Important finding:

- Normal whisper.cpp decoding collapsed into repeated tail text on this song.
- The hybrid backend now retries with whisper.cpp `-sns` only when the first
  candidate has low matching confidence, then keeps the better candidate.
- This preserves the `01. 雨模様` result, where normal decoding remains better
  for short repeated lines such as `空へ`.

## Next Backend Direction

The next useful backend is not a plain LLM. OpenClaw/llama.cpp is a text model
stack and can help with lyric cleanup or ASR-to-lyric matching, but the timestamp
source must come from an audio model or forced aligner.

Promising next steps:

- Implement a whisper.cpp JSON backend wrapper that always converts Unicode
  paths to temp WAVs. Done for the experimental `whispercpp` timing source.
- Add a monotonic ASR-to-lyric matcher with review flags for low-confidence
  lines. Done for the experimental `whispercpp` timing source.
- Add a music-aware vocal/onset refinement stage instead of generic onset peaks.
- Add more checked songs before claiming future-song 90-95% reliability.
- Evaluate MFA or another reference-transcript forced aligner for Japanese
  phoneme/word timing.
- Keep `scripts/evaluate_lrc.py` as the gate before changing the default.
