# LRC Timeline Aligner

Local Windows tooling for aligning prepared lyric/LRC text to a FLAC audio
timeline and writing same-folder `.lrc` files with an auditable
`.align-report.json`.

This is not an AI LRC generator. It does not invent lyrics from music, and it
is not a general LRC editor. The project is a timeline aligner / retimer for
prepared lyrics, checked LRC templates, and local audio alignment experiments.

## What It Does

LRC Timeline Aligner preserves the input lyric order and focuses on timestamp
reconstruction. It accepts a FLAC file plus prepared lyrics, then chooses the
best available local timing path:

- reuse checked timestamps from an existing LRC template;
- reuse a checked same-stem library LRC when the sung-text order matches;
- use the current local WhisperX hybrid backend when no checked timing source
  exists;
- write a same-name `.lrc` beside the FLAC;
- write a sibling `.align-report.json` so the timing path can be audited.

## Current Reliability Behavior

The current best local backend uses whisper.cpp ASR anchors plus WhisperX
Japanese forced alignment. The current local setup uses a CUDA/cuBLAS
whisper.cpp build and WhisperX with CUDA when available.

CUDA acceleration makes WhisperX-based timing practical for local batch
alignment. It does not guarantee perfect timestamps for every song.

These benchmark results are regression gates for the current local test set,
not a guarantee that every song will align perfectly. The current local gates
cover three checked songs and require 100% of sung lyric entries to land within
`+/-0.50s`; checked-LRC hint mode must match the checked reference exactly.

## Current Safety Behavior

This repository does not include copyrighted audio files, full song lyrics,
model weights, virtual environments, or generated benchmark outputs.

The tool is intentionally report-first:

- explicit `-TimingSource whisperx` failures fail loudly;
- explicit `whisperx` mode must not silently write heuristic drafts;
- heuristic timing is available only as an experimental draft path;
- generated LRC files are paired with `.align-report.json` audit data;
- non-lyric markers such as `(Intro)`, `(Interlude)`, `(Outro)`, and musical
  note markers are skipped during untimed audio alignment.

## Limitations

- Works best when prepared lyrics match the sung text order.
- Dense vocals, overlapping vocals, strong reverb, ad-libs, and mismatched
  lyrics still require report review.
- WhisperX and CUDA improve practicality, not absolute certainty.
- The current benchmarks are private regression checks, not public song
  fixtures.
- The heuristic path is a rough draft mode and should not be treated as
  production-quality timing.

## Lyric Input Contract

The lyric file is the source of truth for text order. The aligner must not
reorder, split, or rewrite the lyric content.

For simple lyrics, each non-empty line is one timing entry:

```text
first line to time
second line to time
third line to time
```

For bilingual or multi-display lines that should appear at the same timestamp,
put them in one timing entry with ` / `:

```text
Japanese sung line / English or translated line
next Japanese line / next translated line
```

The output becomes:

```text
[00:10.00]Japanese sung line
[00:10.00]English or translated line
[00:12.00]next Japanese line
[00:12.00]next translated line
```

If you already have an LRC-like lyric template, name it `Song.lyrics.lrc`.
Consecutive lines with the same timestamp are treated as one timing entry.

Do not use `Song.lrc` as the lyric source for a new run when it sits beside the
FLAC, because `Song.lrc` is the generated output path. A checked top-level
library LRC, such as `D:\MusicLibrary\Song.lrc`, is safe when the FLAC itself
lives deeper under `D:\MusicLibrary\Music\...`.

## Drag/Drop Usage

Put prepared lyrics next to the audio:

```text
Song.flac
Song.lyrics.txt
```

Then drag `Song.flac` onto `Align LRC.bat`.

The tool also checks the nearest parent folder named `Music` for a same-stem
checked LRC. For example:

```text
D:\MusicLibrary\Music\Album\Song.flac
D:\MusicLibrary\Song.lrc
```

If the checked LRC text order matches the prepared lyrics, the checked
timestamps are reused.

## PowerShell Usage

Preserve timestamps from a checked template:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource lyrics "D:\Music\Song.flac"
```

Run the current best local hybrid backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource whisperx "D:\Music\Song.flac"
```

Run the rough experimental heuristic:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource heuristic "D:\Music\Song.flac"
```

Write to a specific path:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource whisperx -Output ".\outputs\Song.lrc" "D:\Music\Song.flac"
```

When `-TimingSource whisperx` succeeds, the console should show the resolved
backend and device:

```text
Timing source: whisperx
Backend request: whisperx hybrid
WhisperX device request: auto
Resolved timing source: whisperx
Backend: whisperx
Strategy: whisperx-hybrid-experimental
WhisperX device: cuda
```

The sibling `.align-report.json` contains the same audit fields:
`requested_timing_source`, `resolved_timing_source`, `backend`, `strategy`,
`mode`, `heuristic_mode`, and `whisperx_device`.

## Local Setup Notes

The WhisperX backend expects local dependencies that are intentionally not
committed to this repository:

- whisper.cpp executable;
- whisper.cpp model weights;
- Python environment with WhisperX;
- PyTorch build appropriate for the local CPU/GPU.

Use `requirements-asr.txt` as a dependency pointer, but install PyTorch from
the official selector for your CUDA/CPU environment before installing WhisperX
when needed.

## Accuracy Gate

Compare a generated LRC against a checked reference:

```powershell
python .\scripts\evaluate_lrc.py "D:\Music\Reference.lrc" ".\outputs\Generated.lrc" --require-within-50cs 95
```

Run the private local regression gate:

```powershell
$env:LRC_TOOLS_MUSIC_DIR = "D:\MusicLibrary"
python .\scripts\run_benchmarks.py
```

Regenerate private outputs with the current pipeline before evaluating:

```powershell
$env:LRC_TOOLS_MUSIC_DIR = "D:\MusicLibrary"
python .\scripts\run_benchmarks.py --regenerate
```

Benchmark summaries in this repository describe private checked references.
They are useful regression notes, not bundled public fixtures.

## Roadmap

v0.2 focuses on report-first alignment:

- per-line confidence scoring;
- suspicious-line detection;
- suspicious reason fields in `.align-report.json`;
- local window re-alignment for low-confidence entries;
- optional anchor-line hints for manually fixed timestamps;
- clearer report schema documentation.

Example future report entry:

```json
{
  "entry_index": 12,
  "text": "...",
  "start": 24.31,
  "confidence": 0.72,
  "flags": ["low_lexical_match", "long_gap", "energy_mismatch"],
  "candidate_sources": ["whisperx", "energy_snap"],
  "review_required": true
}
```

v0.3 explores harder alignment problems:

- dual-backend comparison between WhisperX and another text/audio aligner;
- vocal-stem preprocessing for difficult songs;
- batch folder processing;
- HTML/CSV report summaries;
- synthetic public benchmark fixtures;
- tighter timing gates such as `+/-0.25s` line onset accuracy and
  consonant/sibilant onset checks.
