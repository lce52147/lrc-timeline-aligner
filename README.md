# LRC Timeline Aligner — v1.2

Local LRC timestamp generation for Windows. The tool keeps lyric order fixed, generates timing evidence from the audio, lets Central arbitration choose the final line starts, and writes a same-name `.lrc` plus an audit report.

## Current architecture

The production path has four layers:

1. CTC and Whisper-family producers generate candidate timings and evidence from the supplied audio and lyric text.
2. Central arbitration evaluates candidate identity, sequence, acoustic support, boundary ownership, and temporal coherence. Central is the only layer that selects the timestamp written to the LRC.
3. The optional R1/R2 arbiter can preserve or validate an incoming Central current. `gross-rescue` is the registered R1 rule. `reviewer-validity` is the validated R2 rule and accepts independent HUBP, WhisperX, and XLSR reviewer evidence only for registered soft evidence gaps.
4. The selected timestamps are serialized to LRC. Review/trust fields remain diagnostic metadata unless an explicit strict-review option asks for a non-zero exit.

Checked reference timestamps are never part of normal automatic inference. Explicit `lyrics` / `checked` modes remain available when assisted timing is intentionally requested.

## Basic use

With `Song.flac` and `Song.lyrics.txt` or `Song.lyrics.lrc` in the same folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 "D:\Music\Song.flac"
```

You can also drag a FLAC onto `Align LRC.bat`. A lyric file may be dragged by itself when the matching FLAC is beside it.

The generated LRC is written beside the audio. Reports go to `outputs\reports` by default.

Direct Python usage:

```powershell
python .\scripts\auto_lrc.py "D:\Music\Song.flac" --lyrics "D:\Music\Song.lyrics.txt" --timing-source auto
```

The wrapper refuses to use the same path as both lyric input and generated output, including when `-Overwrite` is supplied.

## Arbiter modes

`--arbiter` accepts:

- `off`: low-level Central-only mode. Central output is used without R1/R2 intervention.
- `gross-rescue`: validated R1 rule. It preserves a valid current timing when Central moves away without the registered large-shift evidence.
- `reviewer-validity`: validated R2 rule. Independent reviewer evidence may validate only the registered soft-current failures. Missing or unqualified reviewer evidence fails closed to Central.

R2 was the final validated system in the 2026-10-02 evaluation. `align-lrc.ps1` and drag/drop now default to `reviewer-validity`, so normal user-facing runs acquire reviewer evidence automatically before the final pass. The low-level `auto_lrc.py` parser still defaults to `off`; direct Python users must explicitly pass `--arbiter reviewer-validity` together with a reviewer sidecar, or use `scripts/r2_pipeline.py`.

Explicit R2 use is one command. When `-Arbiter reviewer-validity` is selected without `-ReviewerEvidence`, the wrapper first runs a Central-only baseline, acquires HUBP / WhisperX / XLSR reviewer observations one song at a time, builds the R2b evidence sidecar, and runs the final reviewer-validity pass:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 `
  -Arbiter reviewer-validity `
  "D:\Music\Song.flac"
```

Low-level Python use may still provide an already-built sidecar directly:

```powershell
python .\scripts\auto_lrc.py "D:\Music\Song.flac" `
  --lyrics "D:\Music\Song.lyrics.txt" `
  --arbiter reviewer-validity `
  --reviewer-evidence ".\outputs\reviewers\Song.json"
```

### How to read the console report

The user-facing R2 flow prints four stages: Central baseline, independent reviewer acquisition, final reviewer-validity output, and reviewer trust summary. Each stage shows `[1/4]` through `[4/4]` plus an elapsed-time line so a long model step is visibly still part of the same run.

`Trusted timing`, `Review required`, and `Low confidence` are internal workflow indicators, not an accuracy estimate. `Trusted timing` means the current evidence satisfied the tool's trust rules; `Review required` and `Low confidence` identify rows that deserve attention under those same rules. They do not mean that the remaining rows are guaranteed correct.

The lyric check is deliberately advisory. A controlled calibration used three songs with clean lyrics, one deleted line, and one adjacent-line swap (9 full R2 runs). Whole-song `provisional_unresolved` and `review_required` ratios did not reliably separate clean from damaged lyrics, so the wrapper does not emit an automatic "lyrics mismatch" verdict. It instead lists the five rows with the lowest `ctc_score` for manual inspection. Across the 6 damaged calibration variants, that lowest-five list covered the edited location within ±2 rows in 6/6 cases. This is a review clue, not an accuracy score or diagnosis.

If fewer than 2 of HUBP / WhisperX / XLSR provide R2-usable reviewer evidence and fewer than 50% of lyric rows have any R2-usable reviewer value, the console also warns that reviewer evidence is sparse. This commonly matters for non-Japanese songs because the shipped reviewer set contains Japanese-specific models; in that case R2 endorsement and reviewer trust labels have little effect, and percentages such as `Trusted timing` must not be read as accuracy.
Reviewer trust labeling is separate from R2b timestamp selection. The optional
`reviewer_layer` profiles use the shipped neural reviewer set HUBP, WhisperX
(WX), XLSR, and HUB. `strict` targets 2% and remains fail-closed to
`ALL_REVIEW`; `balanced` is the fixed four-reviewer `>=3` rule; `loose` is the
fixed four-reviewer `>=2` rule. The default remains `none`, which does not mark
any row reviewer-trusted. This does not change R2b timing logic or `--arbiter`
behavior.

The wrapper exposes these labels as `-ReviewerProfile none|strict|balanced|loose`.
HUB acquisition remains explicit with `-IncludeHubReviewer`; if HUB or any other
required reviewer is missing or unqualified, `balanced` and `loose` fail closed
to review. Non-Japanese rows also always fail closed to review. The selected
profile is written as auxiliary metadata to
`outputs/reviewer-work/<song>/reviewer-trust.json`; it does not alter the LRC
timestamps. The equivalent direct pipeline option is
`--reviewer-profile none|strict|balanced|loose`.

Source for every numeric trust-profile result in this section: **來自內部評估流程，參考答案與逐行資料不公開**. The public repository contains the production rules and tests, but not the private checked-reference corpus needed to reproduce these aggregate CV results.

The fixed-rule / point-estimate view on the 37-song pool is:

| Target | Production profile / fixed rule | CV coverage (95% song-bootstrap CI) | CV trusted error (95% song-bootstrap CI) |
|---:|---|---:|---:|
| 2% | `strict = ALL_REVIEW`; closest four-reviewer boundary candidate `HUBP+WX+XLSR/all` was not promoted | 12.8713% [8.7530, 17.0889] | 3.2967% [0.8127, 6.7583] |
| 5% | `balanced = HUBP+WX+XLSR+HUB/>=3` | 27.7935% [19.3035, 36.0784] | 4.3257% [2.4614, 6.5767] |
| 10% | `loose = HUBP+WX+XLSR+HUB/>=2` | 41.7256% [29.6884, 52.8856] | 6.4407% [3.9032, 9.1198] |

`balanced` therefore passes the 5% point-estimate selector, while the more
conservative Selector B (training-fold Wilson 95% upper bound <= target) does
not support it. Selector B on the second-round 28-rule family gives:

| Target | Selector B CV coverage (95% CI) | Selector B trusted error (95% CI) | Fold selection |
|---:|---:|---:|---|
| 2% | 0.0000% [0.0000, 0.0000] | N/A | `ALL_REVIEW` in 37/37 folds |
| 5% | 0.9901% [0.0000, 2.2580] | 42.8571% [33.3333, 50.0000] | `ALL_REVIEW` in 34/37 folds |
| 10% | 41.7256% [28.9923, 53.8987] | 6.4407% [4.0439, 9.5432] | `loose` rule in 37/37 folds |

The second-round family was designed after observing the first-round search.
That design choice has a material effect under Selector B: at 10%, changing
from the first-round 32-rule family to the second-round 28-rule family changes
coverage by -10.4668 percentage points and trusted error by -8.1934 points.
The 37-song pool therefore does not provide a truly independent validation set
for trust-profile selection.

For the fixed production rules, descriptive partition figures are:

| Profile | dev coverage / trusted error | former-acceptance coverage / trusted error |
|---|---:|---:|
| `balanced` | 325/1011 = 32.1464% / 14/325 = 4.3077% | 68/403 = 16.8734% / 3/68 = 4.4118% |
| `loose` | 489/1011 = 48.3680% / 34/489 = 6.9530% | 101/403 = 25.0620% / 4/101 = 3.9604% |

All trusted rows produced by these fixed rules in the analysis pool are
Japanese. HUB is unsupported for the tested English and mixed songs and mostly
unsupported outside Japanese; when any required reviewer is unavailable, the
trust layer fails closed to review. HubertFA was tested but is not shipped and
did not establish a deployable CV-safe profile. VON and REC were also tested
but are unused because their wrong-confirmation and shifted-time results did
not establish the required independence/discrimination.

These trust labels are auxiliary information, not a correctness guarantee.

## Timing-source modes

`--timing-source auto` is the normal independent path. It uses the available CTC / Whisper-family backends without taking timing authority from a checked LRC.

Other explicit modes include `lyrics`, `checked`, `ctc`, `jactc`, `whisperx`, `whispercpp`, `heuristic`, and `audio`. `jactc`, `whispercpp`, `heuristic`, and `audio` remain experimental comparison or fallback paths.

## Evaluation method

The manually aligned references are private and are not published with the repository because they contain manually checked lyric text/timestamps that are intentionally kept outside the public release. The public evaluator first normalizes the sung text with NFKC and whitespace removal, aligns equal lyric rows in sequence order, excludes non-lyric marker rows such as `♪`, and only then compares timestamps. Added or missing lyric rows are reported explicitly instead of shifting all later timing pairs.

Use the same public evaluator with your own checked reference:

```powershell
python .\scripts\evaluate_lrc.py "D:\Checked\Reference.lrc" ".\outputs\Generated.lrc" --json
```

The primary JSON fields are `timing_compared_entries`, `unmatched_reference_count`, `unmatched_generated_count`, `correct_le_30ms[_percent]`, `acceptable_30_to_50ms[_percent]`, `wrong_gt_50ms[_percent]`, `median_abs_delta_ms`, `mae_ms`, and `max_abs_delta_ms`. `aligned_pairs` and the `unmatched_*_entries` arrays show which text rows were paired or left unmatched. Legacy `within_10cs` / `within_25cs` / `within_50cs` / `within_100cs` fields remain for historical regressions; their names are centiseconds, so `within_10cs` means <=100 ms rather than the current <=30 ms correctness band.

The current reporting bands are:

- `<= 30 ms`
- `30-50 ms`
- `> 50 ms`

Source for every numeric corpus size, date, timing result, count, percentage, MAE, and comparison in the remainder of this evaluation section: **來自內部評估流程，參考答案與逐行資料不公開**. The command above reproduces the evaluation method on user-supplied references; the repository alone cannot reproduce the published aggregate values because the checked references are intentionally absent.

The frozen development split contains 27 songs / 1011 lyric entries. A former
acceptance split contains 10 songs / 403 lyric entries. Reference timestamps
are withheld from inference. On 2026-10-03 those 10 songs were explicitly
merged with dev for the reviewer-combination search, producing a 37-song pool
evaluated with leave-one-song-out cross-validation (LOSO). The former acceptance
subset is therefore only a historical/descriptive partition and is no longer
an independent validation set for reviewer-trust selection.

Release v1.2 measurements from a fresh clean-release replay on 2026-10-04:

| Split | <=50 ms | >1 s | MAE | trusted but >50 ms | review / unverified |
|---|---:|---:|---:|---:|---:|
| dev | 792/1011 (78.34%) | 36 | 180.74 ms | 82 | 573/1011 (56.68%) |
| former acceptance (historical partition) | 234/403 (58.06%) | 59 | 1596.50 ms | 46 | 230/403 (57.07%) |

The former-acceptance accuracy was 20.28 percentage points below dev in the
release-tree replay. These timing measurements come from the exact clean release
tree after deterministic vocal separation was enabled;
the same 10-song subset has since joined the 37-song reviewer-combination
analysis pool and is no longer independent. R2 also marks more rows as
review/unverified than the earlier S03 development result: 573/1011 (56.68%)
versus 385/1011 (38.08%). The R2 reviewer-validity path is deliberately
conservative about trust: missing or unqualified reviewer evidence fails closed,
and reviewer agreement only relaxes registered soft evidence gaps. Timing
accuracy and trusted/review status measure different things, so the higher
review rate does not by itself imply worse timing. The current system still has difficult
repeated passages, large timing misses, and songs where WhisperX is unavailable
or rejected by its own trust gate. These measurements describe the tested
corpus; they are not a guarantee for arbitrary music.

[`docs/existing-summary.md`](docs/existing-summary.md) contains the same publishable aggregate snapshot and split song names. It intentionally contains no lyric lines or reference timestamps.

## Environment rebuild

The repository does not include audio, model weights, private references, generated reports, virtual environments, or third-party binaries.

Recommended Windows setup:

```powershell
py -m venv .venv-asr
.\.venv-asr\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the PyTorch / torchaudio build matching your CUDA or CPU environment first.
python -m pip install -r .\requirements-asr.txt
python -m pip install -r .\requirements.txt
```

The 2026-10-02 validated Windows environment used Python 3.11.5, ffmpeg
9.0.1 (Gyan essentials build), PyTorch 2.8.0+cu128, torchaudio
2.8.0+cu128, WhisperX 3.8.6, Transformers 4.57.6, NumPy 2.4.4,
pykakasi 2.3.0, huggingface-hub 0.36.2, and the CUDA 12.8 PyTorch
runtime. The GPU validation host used an NVIDIA GeForce RTX 4070 Ti with
driver 610.88. Exact GPU hardware is not a correctness requirement, but
the CUDA/PyTorch pair must be compatible.

Source for the version and hardware numbers in the preceding paragraph is the validated local environment inventory. Reproduce the corresponding values on another machine with:

```powershell
python --version
ffmpeg -version
python -c "import importlib.metadata as m; print({p:m.version(p) for p in ['torch','torchaudio','whisperx','transformers','numpy','pykakasi','huggingface-hub']})"
nvidia-smi
```

External components used by the automatic path:

- Python 3 on Windows.
- `ffmpeg` and `ffprobe` on `PATH`.
- PyTorch / torchaudio matching the selected CUDA runtime or CPU.
- whisper.cpp built for Windows, with `whisper-cli.exe` at `tools\whisper.cpp\Release\whisper-cli.exe` unless an explicit path is supplied.
- whisper.cpp `ggml-large-v3.bin` at `models\whisper.cpp\ggml-large-v3.bin` unless an explicit model path is supplied.
- WhisperX in `.venv-asr`.
- The XLSR Japanese CTC model `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` available in the Hugging Face cache when XLSR reviewer evidence is used.
- The HUBP model `prj-beatrice/japanese-hubert-base-phoneme-ctc`, revision `1ec4eb3c45b2a1cafb7c477d447df34ca03070f2`, materialized at `models\hubert-phoneme-ctc` when R2 reviewer evidence is used.

The validated XLSR cache resolved to revision
`cf031e020336460d15a417eba710bbc5bb43be9a`. HUBP planning also requires
`pyopenjtalk-plus==0.4.1.post3`; it must be installed in the environment
that runs `build_hubp_plan.py`.

Source for the model revision identifiers and the pinned `pyopenjtalk-plus` version in this section is the public acquisition command below plus `requirements-asr.txt`; rerunning those commands resolves the same requested revisions when they remain available upstream.

The reviewer models can be acquired with huggingface-hub after the Python
environment is installed:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('prj-beatrice/japanese-hubert-base-phoneme-ctc', revision='1ec4eb3c45b2a1cafb7c477d447df34ca03070f2', local_dir='models/hubert-phoneme-ctc')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('jonatasgrosman/wav2vec2-large-xlsr-53-japanese', revision='cf031e020336460d15a417eba710bbc5bb43be9a')"
```

The local whisper.cpp executable used for validation does not expose a
version string and its checkout metadata is unavailable, so no exact
whisper.cpp revision is claimed here. Rebuild the current upstream
`whisper-cli` with CUDA support when desired, place it at the path above,
and obtain the `large-v3` GGML model using whisper.cpp's model download
script or an equivalent official model source. Model and binary files stay
outside Git.

`requirements.txt` and `requirements-asr.txt` are dependency pointers, not a lockfile. They do not install ffmpeg, whisper.cpp, model weights, or the correct CUDA-specific PyTorch build.

CUDA is optional for the base code paths but is the tested configuration for the reviewer generators used by R2. Model acquisition must be completed before using those reviewers.

## Output and review

A normal run writes:

- `Song.lrc`
- `outputs\reports\<song>--<id>.align-report.json`

Useful strict options include `--fail-on-review-required`, `--min-trusted-percent`, and `--strict-review`. The PowerShell wrapper exposes the same review gates.

Reports record the selected backend, Central timing state, trust/review counts, candidate provenance, and optional arbiter evidence. Review metadata does not silently prevent the LRC from being written.

## Public-safe validation

Run the checks that do not need private media or model files:

```powershell
python .\scripts\check_public.py
```

For local evaluation against a manually reviewed reference, the exact public command is:

```powershell
python .\scripts\evaluate_lrc.py "D:\Checked\Reference.lrc" ".\outputs\Generated.lrc" --json
```

The JSON output fields and pairing rules are documented in **Evaluation method** above. Legacy centisecond gates remain available for historical regressions. `50cs` means 500 ms; it is not the current 50 ms evaluation boundary.

## Repository boundaries

The repository intentionally excludes:

- audio and generated LRC files;
- model weights and third-party binaries;
- private reference answers;
- `_review`, `_local_archive`, virtual environments, `.c39rt1`, and generated validation/output trees;
- local checkpoint, handoff, changelog, and session notes.

Only publishable source, tests, setup files, and aggregate documentation belong on the release branch.
