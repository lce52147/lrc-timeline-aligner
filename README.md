# LRC Timeline Aligner

Local Windows tooling for aligning prepared lyric/LRC text to a FLAC audio
timeline and writing same-folder `.lrc` files. Diagnostic reports are written
under the project `outputs/reports` directory, not into the music library.

This is not an AI LRC generator. It does not invent lyrics from music, and it
is not a general LRC editor. The project is a timeline aligner / retimer for
prepared lyrics, checked LRC templates, and local audio alignment experiments.

## What It Does

LRC Timeline Aligner preserves the input lyric order and focuses on timestamp
reconstruction. It accepts a FLAC file plus prepared lyrics, then chooses the
best available local timing path:

- reuse checked timestamps from an existing LRC template;
- reuse a checked same-stem library LRC when the sung-text order matches;
- run MMS/CTC and WhisperX hybrid candidates when no checked timing source
  exists, then select by report quality;
- write a same-name `.lrc` beside the FLAC;
- write reports and strict-review audit files under `outputs/reports` so the
  music folder stays clean.

## Current Reliability Behavior

The primary forced-alignment backend is MMS/CTC through
`torchaudio.pipelines.MMS_FA`. It keeps the prepared lyric order fixed,
romanizes sung Japanese text, and uses CTC/Viterbi alignment to place the whole
known lyric sequence on the audio timeline.

CTC output also applies a conservative acoustic backtrack for low-confidence
Japanese `r`-initial lines when the CTC start lands late but a strong local
onset exists just before it. The report records these changes as
`ctc_acoustic_backtracks`. This is a narrow refinement, not the final
multi-language consonant/sibilant onset model.

`auto` mode also runs the whisper.cpp + WhisperX hybrid candidate when it is
available. CTC is preferred on near-ties because it is constrained to the known
lyric order, but WhisperX can win when its report quality is clearly better.
An experimental `-VocalOnsetRefine` switch enables a conservative Demucs
vocal-onset tiebreak. It runs only when CTC and hybrid disagree substantially,
and changes a timestamp only when the isolated-vocal onset clearly supports the
CTC candidate. It is not enabled by default until it demonstrates a net gain on
the checked regression set.

`jactc` is an explicit experimental Japanese Wav2Vec2 CTC backend. It aligns
native Japanese tokens without romanization, but is deliberately excluded from
`auto` until it passes the checked-song regression set without collapse.

These benchmark results are regression gates for the current local test set,
not a guarantee that every song will align perfectly. The current local gates
cover three checked songs: rain and utopia require 100% of sung lyric entries
to land within `+/-0.50s`, and rain, utopia, and oyasumi all require 100%
within `+/-0.25s`.
Checked-LRC hint mode must match the checked reference exactly.

## Current Safety Behavior

This repository does not include copyrighted audio files, full song lyrics,
model weights, virtual environments, or generated benchmark outputs.

The tool is intentionally report-first:

- default `auto` preserves checked LRC hints first, otherwise runs CTC and
  WhisperX candidates and selects the safer backend by report quality;
- explicit `-TimingSource whisperx` failures fail loudly;
- explicit `-TimingSource ctc` uses MMS/CTC forced alignment over known lyrics;
- explicit `whisperx` mode must not silently write heuristic drafts;
- heuristic timing is available only as an experimental draft path;
- generated LRC files have corresponding audit data under `outputs/reports`;
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

The drag/drop batch file writes its LRC even when the report has lines that need
manual review; the terminal prints that warning and the report remains under
`outputs/reports`. To make a review-required line or less than 100% trusted
timing fail the command, set `LRC_TOOLS_STRICT_REVIEW=1` before dragging.

For a non-interactive smoke test of the same batch entry point, disable only the
final `pause`:

```cmd
set LRC_TOOLS_NO_PAUSE=1
"Align LRC.bat" "D:\Music\Song.flac"
```

Default `auto` behavior:

```text
checked LRC hint
  -> otherwise CTC candidate + WhisperX candidate
  -> backend-aware scorer selects ctc or whisperx
  -> .align-report.json records candidate_selection
```

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

Run the current primary local forced-alignment backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource ctc "D:\Music\Song.flac"
```

Run automatic backend selection:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource auto "D:\Music\Song.flac"
```

Run automatic backend selection as a production gate. This still writes draft
outputs, but exits non-zero if the report is not clean:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource auto -StrictReview "D:\Music\Song.flac"
```

When strict review fails, the tool also writes `Song.review-audit.md` with only
the review-required rows and candidate backend timestamps. It also writes
`Song.anchor-template.lrc`, which is a starter file for manual checking and is
not auto-applied. Passing an `.anchor-template.lrc` directly as `-AnchorHints`
is rejected so an unreviewed template cannot accidentally become trusted timing.

For a softer gate, fail only when review-required lines exist:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource auto -FailOnReviewRequired "D:\Music\Song.flac"
```

If a strict run writes `Song.anchor-template.lrc`, listen and verify the rows,
then copy or rename only the manually checked anchors into a timestamped
`Song.anchors.lrc` beside the FLAC. The next run will lock those line starts
after matching the sung text order. Keep the generated `# entry=N` comments
when possible; they bind each anchor to the original lyric entry and prevent
repeated lines from being applied to the wrong occurrence.

```text
# entry=4
[00:41.81]verified lyric line
# entry=6
[00:58.42]another verified lyric line
```

You can also pass an explicit partial anchor file:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 -TimingSource auto -AnchorHints "D:\Music\Song.anchors.lrc" "D:\Music\Song.flac"
```

Run the experimental WhisperX hybrid backend:

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

When `-TimingSource ctc` succeeds, the console should show the resolved
backend and device:

```text
Timing source: ctc
Backend request: MMS/CTC forced alignment
CTC device request: auto
Resolved timing source: ctc
Backend: ctc
Strategy: ctc-forced-align
CTC device: cuda
```

The report in `outputs/reports` contains the same audit fields:
`requested_timing_source`, `resolved_timing_source`, `backend`, `strategy`,
`mode`, `heuristic_mode`, `ctc_device`, and fallback-specific device fields
such as `whisperx_device`.

When `auto` has no checked LRC hint, the report also contains
`candidate_selection`. It records the selected backend, selected quality,
selection reason, and candidate summaries. CTC candidate summaries include
`ctc_score_min`, `ctc_score_mean`, `ctc_low_score_count`,
`ctc_very_low_score_count`, and `ctc_missing_count`; WhisperX summaries include
trusted/review percentages, collapse state, device, and suppress-NST mode.
CTC assignments include `ctc_token_spans`, the romanized character-level CTC
spans used to audit why a line start was placed where it was.
Hybrid reports may mark individual assignments as `timing_trusted` when CTC,
WhisperX, and raw ASR timestamps agree closely enough, or when a high-confidence
raw internal anchor explains a local WhisperX offset. The original text-match
score is preserved; `timing_trusted` only affects timing trust metrics.

## Report-First Line Decisions

Audio alignment is not treated as one backend producing one unquestioned
timestamp. For each lyric entry, the report records a decision object on the
assignment:

- `chosen_time` and `confidence`;
- `candidates` and `rejected_candidates` from the selected path, raw ASR,
  WhisperX forced-first timing, and available CTC leading-token peaks;
- `reasons`, `penalties`, `flags`, and `review_required`;
- `phonetic_anchor`, using the optional Japanese romaji/mora profile when it
  is available; and
- `split_suggestion` for long lines whose candidate evidence suggests more
  than one sung phrase.

The decision layer penalizes candidate spread, prior-line tail attachment,
long-line disagreement, and short-line onset uncertainty. A replacement time
is only adopted when it is inside neighboring lyric bounds and its candidate
evidence remains strong after those penalties. Otherwise the selected backend
time remains a draft and the line is marked for review. This is deliberately
more conservative than silently writing a clean-looking but disputed LRC.

The optional Japanese phonetic adapter uses `pykakasi` plus the CTC first-token
peaks as onset evidence. It improves auditability rather than claiming that
romanization alone proves an acoustic onset.

Export a readable per-line audit table from a generated LRC. The audit tool
finds its matching project report automatically. The audit includes review flags, backend candidate times, and
`timing_trusted` reasons/sources when hybrid consensus was used. When CTC
evidence is available, the audit also includes compact leading token spans such
as `s@00:40.86/0.066`.

```powershell
python .\scripts\export_alignment_audit.py "D:\Music\Song.lrc" --output "D:\Music\Song.audit.md"
python .\scripts\export_alignment_audit.py "D:\Music\Song.lrc" --format csv --output "D:\Music\Song.audit.csv"
```

For fast review of only the lines that must not be trusted without listening:

```powershell
python .\scripts\export_alignment_audit.py "D:\Music\Song.lrc" --review-only
```

For algorithm probes, preserve a manually checked same-name LRC by writing the
candidate under the project instead of beside the audio:

```powershell
python .\scripts\auto_lrc.py "D:\Music\Song.flac" --probe --timing-source auto --overwrite
```

`--probe` writes a hashed disposable LRC to `outputs/probes` and keeps reports
under `outputs/reports`. It cannot be combined with `--output`.

## Local Setup Notes

The CTC and WhisperX backends expect local dependencies that are intentionally not
committed to this repository:

- PyTorch / torchaudio build appropriate for the local CPU/GPU;
- `pykakasi` for Japanese lyric romanization;
- whisper.cpp executable;
- whisper.cpp model weights;
- Python environment with WhisperX;

Use `requirements-asr.txt` as a dependency pointer, but install PyTorch from
the official selector for your CUDA/CPU environment before installing optional
ASR/alignment packages when needed.

## Accuracy Gate

Compare a generated LRC against a checked reference:

```powershell
python .\scripts\evaluate_lrc.py "D:\Music\Reference.lrc" ".\outputs\Generated.lrc" --require-within-50cs 95
```

Run the private local regression gate:

```powershell
python .\scripts\check_public.py
python .\scripts\run_benchmarks.py --ctc-only --regenerate
python .\scripts\run_benchmarks.py --auto-selection-only --regenerate
```

`check_public.py` is public-safe and does not require audio, lyrics, model
weights, or GPU access. It runs the core logic tests, compiles the Python entry
points, and checks that media/model/generated artifacts are not tracked. GitHub
Actions runs the same public-safe check on push and pull request.

`--auto-selection-only` uses the internal `--no-checked-lrc-hint` test option so
the checked LRC shortcut does not hide the CTC/WhisperX selection behavior.

Run private checked-song regressions for local files that are not committed to
the repository. These cases compare regenerated output against an independently
stored checked LRC. Never use `Song.lrc` beside the FLAC as the reference: that
is the drag/drop output path. Put human-reviewed references in a separate local
directory with the explicit `.checked.lrc` suffix. The current local set uses
`10.方舟` as a difficult CTC weak-onset regression; `04.可惜夜` is included only
when `04.可惜夜.checked.lrc` exists in the reference directory:

```powershell
$env:LRC_TOOLS_MUSIC_DIR = "D:\Users\Administrator\Music"
$env:LRC_TOOLS_CHECKED_REFERENCE_DIR = "D:\Users\Administrator\Music\LRC tools checked references"
python .\scripts\run_benchmarks.py --local-regression-only --regenerate
```

Run a private risk-audit gate for a local difficult song without committing
audio, lyrics, or generated outputs:

```powershell
$env:LRC_TOOLS_PRIVATE_AUDIT_AUDIO = "D:\Music\Song.flac"
$env:LRC_TOOLS_PRIVATE_AUDIT_LYRICS = "D:\Music\Song.txt"
$env:LRC_TOOLS_PRIVATE_AUDIT_REVIEW_COUNT = "0"
$env:LRC_TOOLS_PRIVATE_AUDIT_TRUSTED_PERCENT = "100"
$env:LRC_TOOLS_PRIVATE_AUDIT_FUSION_COUNT = "3"
python .\scripts\run_benchmarks.py --private-audit-only --regenerate
```

Private audits run with `--strict-review` by default. Set
`LRC_TOOLS_PRIVATE_AUDIT_STRICT=0` only when intentionally auditing a known
failing draft.

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
