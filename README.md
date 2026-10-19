# LRC Timeline Aligner — v1.4.0

Local LRC timestamp generation for Windows. The tool keeps lyric order fixed, generates timing evidence from the audio, lets Central arbitration choose the final line starts, and writes a same-name `.lrc` plus an audit report.

## What's new in v1.4

- Review page opens automatically after a batch alignment that produced review rows; `LRC_NO_REVIEW=1` disables it; install paths with spaces are supported.
- Candidates can be listened to and applied by click or digit key; each chip shows how many other candidates agree within ±0.15 s; overflow candidates sit in a collapsible list.
- A/B playback alternates the current and the selected candidate timestamps.
- Vocal-stem waveform strip for the selected row with drag adjustment, plus visible ±0.1 s nudges and JIS-safe bracket keys.
- Plain-language reason text for every review row.
- Transactional save: the LRC, the first-save `.auto.lrc` and the edit log are staged and published together; a failed save leaves the files unchanged.
- Timing and alignment algorithm unchanged from v1.3.1 (39-song cache-only replay byte-identical).

## What's new in v1.3

v1.3.1 is an installation and packaging update. It does not change the v1.3 timing or alignment policy. The setup path now builds the repository-local ASR environment, pins the validated dependencies and third-party binaries, can prefetch the production models, and finishes with a doctor check.

v1.3 adds a post-Central multimodel consensus layer and stronger review flagging on top of the v1.2 architecture.

- **Multimodel consensus (post-Central):** rows that Central leaves unresolved are re-aligned by MMS, MMS-1B (eng/jpn), three English CTC models and a Japanese CTC model. A new time is accepted only when at least three model families agree within 0.10 s (MMS required) and the occurrence/boundary guards pass.
- **Review flags:** `model-disagreement` and `silent-risk` mark lines where the models disagree or the timing pattern is risky, so large errors are surfaced for review instead of being written silently.
- **Reproducible reruns:** model outputs are cached by execution identity, so cache-only replays reproduce a run exactly.

Measured on the 39-song / 1487-line manually checked validation pool (cache-only replay of the v1.3.0 release):

| Measure | v1.3.0 |
|---|---|
| Lines within 0.2 s (PCO@0.2) | 1362 / 1487 (91.6%) |
| Lines off by more than 0.5 s | 59 |
| Lines off by more than 1 s | 40, all flagged for review |
| TRUSTED lines off by more than 1 s | 0 |
| Test suite | 1031 passed, 5 skipped |

These results describe this corpus and do not predict accuracy on arbitrary songs. The v1.2 sections below are kept for history.

## Current architecture

The production path has five layers:

1. CTC and Whisper-family producers generate candidate timings and evidence from the supplied audio and lyric text.
2. Central arbitration evaluates candidate identity, sequence, acoustic support, boundary ownership, and temporal coherence. Central is the only layer that selects the timestamp written to the LRC.
3. The optional R1/R2 arbiter can preserve or validate an incoming Central current. `gross-rescue` is the registered R1 rule. `reviewer-validity` is the validated R2 rule and accepts independent HUBP, WhisperX, and XLSR reviewer evidence only for registered soft evidence gaps.
4. Post-Central multimodel consensus (v1.3) may replace a still-unresolved Central time when at least three model families agree, and adds review flags where they disagree.
5. The selected timestamps are serialized to LRC. Review/trust fields remain diagnostic metadata unless an explicit strict-review option asks for a non-zero exit.

Checked reference timestamps are never part of normal automatic inference. Explicit `lyrics` / `checked` modes remain available when assisted timing is intentionally requested.

### Post-v1.2 selector changelog

The `reviewer-validity` selector now compares reviewer support before a reviewer-endorsed current timestamp can yield to a distant independent-consensus candidate. Candidate seed ranking also uses acoustic score and producer semantics before provenance identity. The R2 reviewer payload now applies the same generated-title-card filtering as the timing backend. On the 27-song development split, PCO@0.2 improved from 87.24% to 91.49%, rows over 500 ms fell from 69 to 49, and rows over 1 s fell from 36 to 30, with no newly created rows over either threshold. A descriptive evaluation of the full 37-song pool improved PCO@0.2 from 82.89% to 86.92%, reduced rows over 500 ms from 141 to 115 and rows over 1 s from 95 to 87, and created no new rows over either threshold. These are dev27 validation plus a full-pool descriptive evaluation; the former acceptance set is part of that pool, so the results have not been independently validated.

## Final timing self-check (F1–F4, post-Central)

After Central selects immutable lyric times, the final report audits every line and can **only reduce trust**, not replace, shift, or sort timestamps. Flagged lines receive timing_trusted=false, overall_trusted=false, review_required=true, and explicit self_check_flags. The trust label does not guarantee manually correct onset.

- F1: a 0.4-second RMS window in the isolated Demucs vocal stem is below the median active-vocal power minus 20 dB. No stem means F1 is skipped and a skip reason is recorded.
- F2: lyric is repeated and selected source depends on content identity.
- F3: three or more consecutively compressed lines (<0.45 × median interline interval), or any timestamp reversal.
- F4: final timestamp differs from original CTC first-token by >1 second.

On the **37-song / 1414-line manually checked BK research pool**, retrospective report replay flags 107 rows: 57/86 large (>1s) errors and 73/195 errors over 200ms; 34 flagged lines were actually within 200ms. This is **diagnostic recall, not timestamp accuracy improvement**. Historical-report replay is not an independent full new same-parent A/B. See _review/H2_BK37_OFFLINE_AUDIT_20261009.json.

## Known limitations（已知限制）

In v1.3, the remaining large errors are lines where every acoustic model fails together: very weak vocals, long instrumental gaps, and some repeated choruses. They are flagged for review in the report. The v1.2 notes below describe the earlier behavior.

Occurrence loss is a known v1.2 limitation: whole-song CTC may misidentify repeated sung passages and collapse or shift consecutive lyric rows. Verified examples: 04 影色舞, 03. Choir ‘S’ Choir, 01.ミレニアの水槽, and My Dearest. Manual listening is required; model confidence alone is insufficient.

The measurements below describe one 37-song / 1414-row analysis pool; the former acceptance subset has been merged into that pool, so there is no independent validation set. On the published selector, the full pool reaches 86.92% PCO@0.2 with 6.15% of rows over 1 s. The Japanese subset (30 songs / 1203 rows) reaches 90.86% PCO@0.2 with 3.74% over 1 s, while the English subset (7 songs / 211 rows) reaches 64.46% PCO@0.2 with 19.91% over 1 s. Median per-song PCO@0.2 is 92%; 22 songs reach at least 90%, and 2 songs are below 60% (`04 影色舞`, 38%; `03. Choir ‘S’ Choir`, 39%). These corpus results are descriptive and do not predict accuracy on arbitrary songs.

R2's shipped reviewer set is designed around Japanese models. English songs receive almost no usable reviewer evidence, so reviewer endorsement and trust labels provide little help there. Reviewer trust labeling defaults to `none`; the measured trusted rows are Japanese only, and the labels are auxiliary review information rather than a correctness guarantee.

Lyrics that omit, add, reorder, or otherwise disagree with the sung words can materially reduce alignment quality. The wrapper therefore shows an advisory lowest-confidence-row list. In a controlled 9-run calibration, whole-song ratios could not distinguish clean lyrics from one-line deletion or adjacent-line-swap variants, while the five lowest `ctc_score` rows covered the edited location within ±2 rows in all 6 damaged variants. This remains a manual inspection clue, not an automatic mismatch diagnosis.

Published forced-alignment results and this project can be compared only at the order-of-magnitude level: common literature evaluates English word onsets and may use noisy reference annotations, while this project evaluates manually checked lyric-line starts on a primarily Japanese corpus.

## Basic use

With `Song.flac` and `Song.lyrics.txt` or `Song.lyrics.lrc` in the same folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 "D:\Music\Song.flac"
```

You can also drag a FLAC onto `Align LRC.bat`. A lyric file may be dragged by itself when the matching FLAC is beside it.

The generated LRC is written beside the audio. Reports go to `outputs\reports` by default.

For a run with several songs, the wrapper prints the queue size and one elapsed/ETA line at the start and end of each song. The ETA average includes every completed attempt, including failures, because each attempt consumes runtime. After alignment it invokes scripts\lrc_batch_summary.py for the original input folder with the run start time, report directory, and repository work directory. A summary warning never replaces the alignment exit code.

To review generated reports, run Review LRC.bat with no arguments. It opens the reports in the local review server and keeps the console window open while the server is running; close the window to stop it. Pass report files or folders, or options such as --read-only, when you need a narrower review. The default server uses a free localhost port. If a manually selected port is occupied, omit --port or choose another one. Edits are saved by the review page and are consumed by the next alignment run.

Windows command lines have an 8191-character limit. Paths with spaces and special characters are passed as argument arrays by the PowerShell wrapper; for a large set of such paths, drop the whole report folder onto Review LRC.bat or process the audio in smaller batches instead of building one long command line.

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

The repository does not include audio, model weights, private references, generated reports, virtual environments, or third-party binaries. Windows with Python 3.11 is required; `setup.ps1` does not install system Python.

From an extracted release folder, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

For a first install that downloads all production/reviewer model assets up front:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Prefetch
```

`setup.ps1` creates or reuses `.venv-asr`, selects the CUDA or CPU PyTorch 2.8.0 build, installs the pinned ASR lock plus the public requirements, installs repository-local ffmpeg/ffprobe when PATH does not provide them, installs the pinned whisper.cpp binary and `ggml-large-v3.bin`, optionally prefetches the Hugging Face and torchaudio models, then runs `scripts\doctor.py`. Use `-Force` to rebuild the venv, or `-Cuda` / `-Cpu` to override GPU detection.

`align-lrc.ps1` prefers `.venv-asr\Scripts\python.exe` and automatically adds the repository-local ffmpeg directory to PATH. Normal use therefore does not depend on another Python installation or a third-party ffmpeg copy elsewhere on the machine.

The v1.3.1 clean-install work used Python 3.11.5 on an NVIDIA GeForce RTX 4070 Ti with 12,282 MiB VRAM. Peak inference VRAM was not instrumented, so 12 GB is the validated host capacity rather than a measured minimum. The completed prefetch plus two-song validation folder occupied 37.12 GiB excluding the copied test songs. The third clean `-Prefetch` attempt took 2224.481 seconds (37 min 4 s) to reach the doctor gate. That attempt downloaded and installed the full environment but the doctor gate failed on cache-dependent model identity; the source fix is covered by the installation tests, and the deployment report keeps the clean gate marked failed because the three-round cap prohibited a fourth rebuild.

Pinned external runtime identities used by this candidate:

- ffmpeg 8.1.2 (Gyan essentials build).
- whisper.cpp Windows CUDA build `b5454`.
- `ggml-large-v3.bin`: 3,095,033,483 bytes, SHA256 `64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2`.
- PyTorch / torchaudio 2.8.0+cu128 on the validated GPU host.
- WhisperX 3.8.6, Transformers 4.57.6, NumPy 2.4.4, huggingface-hub 0.36.2, and pyopenjtalk-plus 0.4.1.post3.

The first clean run is network- and disk-heavy. Observed large transfers included the 3.5 GB PyTorch CUDA wheel, a 2.88 GB `ggml-large-v3` model, two roughly 1.18 GB torchaudio checkpoints, the whisper.cpp CUDA archive, ffmpeg, and the prefetched Hugging Face model snapshots. Later runs reuse the repository-local venv and the configured model caches.

After setup, place `Song.flac` and `Song.lyrics.txt` (or `.lyrics.lrc`) together and drag the FLAC onto `Align LRC.bat`, or run the PowerShell wrapper directly.

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
