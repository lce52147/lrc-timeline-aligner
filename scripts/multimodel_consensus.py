"""Pure multi-model consensus windowing and guard logic.

The module deliberately has no audio, model, filesystem, or production-state
dependencies. Runtime orchestration lives in multimodel_align.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import re
from statistics import median
from typing import Mapping, Sequence
from functools import lru_cache

import runtime_paths


TAU_SECONDS = 0.10
MIN_SHIFT_SECONDS = 0.50
COLLISION_SECONDS = 0.20
PROTOCOL_REVISION = "multimodel-align-v3-g8-g9"
MODEL_SPECS = (
    "mms",
    "mms1b:eng:roman",
    "mms1b:jpn:native",
    "en-ctc:facebook/wav2vec2-large-960h-lv60-self:roman",
    "en-ctc:jonatasgrosman/wav2vec2-large-xlsr-53-english:roman",
    "en-ctc:torchaudio:roman",
    "ja-ctc:jonatasgrosman/wav2vec2-large-xlsr-53-japanese:native",
)

# Opt-in experiment models (never in MODEL_SPECS; see extra_model_specs()).
KARAOKE_JA_REPO = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"
KARAOKE_JA_SPEC = f"en-ctc:{KARAOKE_JA_REPO}:roman"
EXTRA_MODELS_ENV = "LRC_MULTIMODEL_EXTRA_MODELS"

HF_MODEL_REVISIONS = {
    "facebook/mms-1b-all": "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
    KARAOKE_JA_REPO: "2ab2b5f46539ee284703c281f286b01d2410ee12",
    "facebook/wav2vec2-large-960h-lv60-self": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": "569a6236e92bd5f7652a0420bfe9bb94c5664080",
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese": "cf031e020336460d15a417eba710bbc5bb43be9a",
}


def extra_model_specs(value: str | None = None) -> tuple[str, ...]:
    """Opt-in model specs from LRC_MULTIMODEL_EXTRA_MODELS (comma/semicolon separated).

    Only ``en-ctc:<pinned repo>:roman|native`` and ``ja-ctc:...`` forms are accepted,
    so an experiment can never pull an unpinned model. Specs already in MODEL_SPECS
    are ignored. Invalid entries fail loudly instead of being skipped.
    """
    raw = os.environ.get(EXTRA_MODELS_ENV, "") if value is None else value
    specs: list[str] = []
    for item in re.split(r"[,;]", raw or ""):
        spec = item.strip()
        if not spec or spec in MODEL_SPECS or spec in specs:
            continue
        parts = spec.split(":", 2)
        if len(parts) != 3 or parts[0] not in {"en-ctc", "ja-ctc"} or parts[2] not in {"roman", "native"}:
            raise RuntimeError(f"{EXTRA_MODELS_ENV}: unsupported model spec: {spec}")
        if parts[1] == "torchaudio" or parts[1] not in HF_MODEL_REVISIONS:
            raise RuntimeError(f"{EXTRA_MODELS_ENV}: unpinned model repo: {parts[1]}")
        specs.append(spec)
    return tuple(specs)


def active_model_specs(value: str | None = None) -> tuple[str, ...]:
    """MODEL_SPECS plus any opt-in extra specs; the cache identity uses this list."""
    return tuple(MODEL_SPECS) + extra_model_specs(value)


_WEIGHT_NAME = re.compile(
    r"^(?:pytorch_model|model|adapter(?:\.[^/]+)?)-?\d*(?:-of-\d+)?\.(?:bin|safetensors)$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_snapshot(repo: str, revision: str | None = None) -> Path | None:
    roots = runtime_paths.hf_hub_cache_candidates()
    repo_dir_name = "models--" + repo.replace("/", "--")
    for root in dict.fromkeys(roots):
        repo_dir = root / repo_dir_name
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        if revision:
            candidate = snapshots / revision
            if candidate.is_dir():
                return candidate
            continue
        ref = repo_dir / "refs" / "main"
        if ref.is_file():
            ref_revision = ref.read_text(encoding="utf-8").strip()
            candidate = snapshots / ref_revision
            if candidate.is_dir():
                return candidate
        found = sorted(item for item in snapshots.iterdir() if item.is_dir())
        if len(found) == 1:
            return found[0]
    return None


def _snapshot_weights(snapshot: Path) -> tuple[Path, ...]:
    files = sorted(
        item for item in snapshot.iterdir()
        if item.is_file() and _WEIGHT_NAME.fullmatch(item.name)
    )
    if any(item.name.startswith("model") and item.suffix == ".safetensors" for item in files):
        files = [item for item in files if not item.name.startswith("pytorch_model")]
    return tuple(files)


def _snapshot_blob(snapshot: Path) -> str:
    files = _snapshot_weights(snapshot)
    if not files:
        return "UNVERIFIED"
    blobs: list[str] = []
    for item in files:
        resolved = item.resolve()
        blob = resolved.name
        if not re.fullmatch(r"[0-9a-f]{64}", blob):
            blob = _sha256_file(resolved)
        blobs.append(f"{item.name}:{blob}")
    return hashlib.sha256("\n".join(blobs).encode("utf-8")).hexdigest()


def hf_model_load_options(repo: str) -> dict[str, object]:
    revision = HF_MODEL_REVISIONS.get(repo)
    if revision is None:
        raise RuntimeError(f"unpinned Hugging Face model: {repo}")
    snapshot = _hf_snapshot(repo, revision)
    if snapshot is None:
        raise RuntimeError(f"pinned Hugging Face snapshot unavailable: {repo}@{revision}")
    weights = _snapshot_weights(snapshot)
    if not weights:
        raise RuntimeError(f"pinned Hugging Face weights unavailable: {repo}@{revision}")
    return {
        "revision": revision,
        "local_files_only": True,
        "use_safetensors": any(
            item.name.startswith("model") and item.suffix == ".safetensors"
            for item in weights
        ),
    }


def _torch_blob(filename: str) -> str:
    roots = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        root = Path(torch_home)
        roots.extend((root / "hub" / "checkpoints", root / "checkpoints"))
    roots.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    for root in dict.fromkeys(roots):
        path = root / filename
        if path.is_file():
            return _sha256_file(path)
    return "UNVERIFIED"


@lru_cache(maxsize=None)
def model_identity(spec: str) -> dict[str, str]:
    """Return local revision/blob evidence; unresolved evidence stays explicit."""
    value = str(spec)
    if value == "mms":
        return {
            "model_revision": "torchaudio:MMS_FA:model.pt",
            "model_blob_sha256": _torch_blob("model.pt"),
        }
    if value.startswith("en-ctc:torchaudio:"):
        return {
            "model_revision": "torchaudio:WAV2VEC2_ASR_LARGE_960H:wav2vec2_fairseq_large_ls960_asr_ls960.pth",
            "model_blob_sha256": _torch_blob("wav2vec2_fairseq_large_ls960_asr_ls960.pth"),
        }
    if value.startswith("mms1b:"):
        repo = "facebook/mms-1b-all"
    elif value.startswith("en-ctc:") or value.startswith("ja-ctc:"):
        parts = value.split(":", 2)
        repo = parts[1]
    else:
        return {"model_revision": "UNVERIFIED", "model_blob_sha256": "UNVERIFIED"}
    snapshot = _hf_snapshot(repo, HF_MODEL_REVISIONS.get(repo))
    if snapshot is None:
        return {"model_revision": "UNVERIFIED", "model_blob_sha256": "UNVERIFIED"}
    return {
        "model_revision": snapshot.name,
        "model_blob_sha256": _snapshot_blob(snapshot),
    }


@dataclass(frozen=True)
class ConsensusOverride:
    entry_index: int
    seconds: float
    support_families: tuple[str, ...]
    model_times: tuple[tuple[str, float], ...]
    near_written_families: tuple[str, ...]
    guards: tuple[tuple[str, bool], ...]


def model_family(model: str) -> str:
    """Map model variants to the family that counts as one vote."""
    value = str(model).lower()
    if value == "mms" or value.startswith("mms:"):
        return "mms"
    if value.startswith("mms1b") or "mms-1b" in value:
        return "mms1b"
    if value.startswith("ja-ctc") or "japanese" in value or "ja-xlsr" in value:
        return "ja-ctc"
    if (
        value.startswith("en-ctc")
        or "english" in value
        or "en-large" in value
        or "torchaudio" in value
    ):
        return "en-ctc"
    return value


def build_windows(
    written_times: Sequence[float],
    statuses: Sequence[str],
    texts: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build the exact bounded windows required by the consensus spec."""
    if len(written_times) != len(statuses):
        raise ValueError("written_times/statuses cardinality mismatch")
    if texts is not None and len(texts) != len(written_times):
        raise ValueError("texts cardinality mismatch")
    selected = [
        index for index, status in enumerate(statuses) if status == "selected_valid"
    ]
    windows: list[dict[str, object]] = []
    for index, status in enumerate(statuses):
        if status != "provisional_unresolved":
            continue
        prior = [other for other in selected if other < index]
        if prior:
            start_index = prior[-1]
            start = float(written_times[start_index])
        else:
            start_index = index
            start = max(0.0, float(written_times[index]) - 8.0)
        following = [
            other for other in selected
            if other > index and float(written_times[other]) > start + 0.5
        ]
        end = float(written_times[following[0]]) if following else start + 30.0
        actual_start = max(0.0, start - 0.3)
        actual_end = end - 0.05
        if actual_end <= actual_start:
            actual_end = actual_start + 0.5
        window_texts = list(texts[start_index:index + 1]) if texts is not None else []
        windows.append({
            "entry_index": index,
            "start": round(actual_start, 6),
            "end": round(actual_end, 6),
            "text_start_index": start_index,
            "target_offset": index - start_index,
            "texts": window_texts,
        })
    return tuple(windows)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _family_points(
    model_times: Mapping[str, object],
) -> list[tuple[float, str, str]]:
    points: list[tuple[float, str, str]] = []
    for model, value in model_times.items():
        if not _finite(value):
            continue
        point = float(value)
        if point < 0:
            continue
        points.append((point, model_family(str(model)), str(model)))
    return sorted(points, key=lambda item: (item[0], item[1], item[2]))


def _best_cluster(
    model_times: Mapping[str, object],
    tau: float,
) -> tuple[float, tuple[str, ...], tuple[tuple[str, float], ...]] | None:
    points = _family_points(model_times)
    if not points:
        return None
    candidates: list[
        tuple[int, float, float, tuple[str, ...], tuple[tuple[str, float], ...]]
    ] = []
    for left_time, _, _ in points:
        cluster = [
            point for point in points
            if 0.0 <= point[0] - left_time <= tau + 1e-9
        ]
        by_family: dict[str, list[tuple[float, str]]] = {}
        for point, family, model in cluster:
            by_family.setdefault(family, []).append((point, model))
        representatives = {
            family: median(value for value, _ in values)
            for family, values in by_family.items()
        }
        support = tuple(sorted(representatives))
        family_times = tuple((family, representatives[family]) for family in support)
        center = float(median(representatives.values()))
        span = max(representatives.values()) - min(representatives.values())
        candidates.append((len(support), span, center, support, family_times))
    chosen = min(
        candidates,
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    return chosen[2], chosen[3], chosen[4]


def model_disagreement_review(
    written_times: Sequence[float],
    statuses: Sequence[str],
    model_outputs: Mapping[int | str, Mapping[str, object]],
    windows: Sequence[Mapping[str, object]],
    *,
    tau: float = TAU_SECONDS,
    min_disagreement: float = 1.0,
) -> dict[int, dict[str, object]]:
    """Find review-only model disagreements without proposing a new time."""
    if len(written_times) != len(statuses):
        raise ValueError("written_times/statuses cardinality mismatch")
    windows_by_entry: dict[int, tuple[float, float]] = {}
    for window in windows:
        try:
            index = int(window["entry_index"])
            start = float(window["start"])
            end = float(window["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(statuses) and math.isfinite(start) and math.isfinite(end) and start <= end:
            windows_by_entry[index] = (start, end)

    reviewed: dict[int, dict[str, object]] = {}
    for raw_index, values in model_outputs.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(statuses)) or index not in windows_by_entry:
            continue
        if not isinstance(values, Mapping):
            continue
        start, end = windows_by_entry[index]
        written = float(written_times[index])
        triggers: list[str] = []
        mms_time = values.get("mms")
        if _finite(mms_time) and start <= float(mms_time) <= end:
            if abs(float(mms_time) - written) > min_disagreement:
                triggers.append("mms")
        cluster = _best_cluster(values, tau)
        cluster_time = None
        cluster_families: tuple[str, ...] = ()
        if cluster is not None:
            cluster_time, cluster_families, _ = cluster
            if (
                len(cluster_families) >= 3
                and start <= cluster_time <= end
                and abs(cluster_time - written) > min_disagreement
            ):
                triggers.append("cluster")
        if triggers:
            reviewed[index] = {
                "entry_index": index,
                "reason": "model-disagreement",
                "written_time": written,
                "mms_time": float(mms_time) if _finite(mms_time) else None,
                "best_cluster_time": cluster_time,
                "cluster_families": list(cluster_families),
                "triggers": triggers,
            }
    return reviewed


def _raw_order_violation(
    index: int,
    seconds: float,
    raw_anchors: Sequence[tuple[int, float]],
) -> bool:
    return any(
        (other > index and value <= seconds)
        or (other < index and value >= seconds)
        for other, value in raw_anchors
    )


def choose_consensus_overrides(
    written_times: Sequence[float],
    statuses: Sequence[str],
    model_outputs: Mapping[int | str, Mapping[str, object]],
    raw_anchors: Sequence[tuple[int, float]] = (),
    *,
    tau: float = TAU_SECONDS,
    min_shift: float = MIN_SHIFT_SECONDS,
    collision: float = COLLISION_SECONDS,
) -> dict[int, ConsensusOverride]:
    """Return only gated provisional-row rewrites."""
    if len(written_times) != len(statuses):
        raise ValueError("written_times/statuses cardinality mismatch")
    proposed: dict[int, ConsensusOverride] = {}
    for raw_index, values in model_outputs.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(statuses)):
            continue
        if statuses[index] != "provisional_unresolved":
            continue
        cluster = _best_cluster(values, tau)
        if cluster is None:
            continue
        consensus, support, _ = cluster
        near = tuple(sorted({
            model_family(model)
            for model, value in values.items()
            if _finite(value) and abs(float(value) - float(written_times[index])) <= tau + 1e-9
        }))
        clean_support = tuple(family for family in support if family not in near)
        g2 = not _raw_order_violation(index, consensus, raw_anchors)
        g10 = len(clean_support) >= 3
        g8 = "mms" in clean_support
        g4 = abs(consensus - float(written_times[index])) > min_shift
        g5_initial = len(near) < 3
        if not (g2 and g10 and g8 and g4 and g5_initial):
            continue
        proposed[index] = ConsensusOverride(
            entry_index=index,
            seconds=consensus,
            support_families=support,
            model_times=tuple(sorted(
                (str(model), float(value))
                for model, value in values.items()
                if _finite(value)
            )),
            near_written_families=near,
            guards=(
                ("G2_raw_order", g2),
                ("G10_clean_family_support", g10),
                ("G4_shift", g4),
                ("G5_near_written", g5_initial),
                ("G8_mms_support", g8),
            ),
        )

    active = set(proposed)
    while active:
        rewritten = {
            index: (
                proposed[index].seconds
                if index in active
                else float(written_times[index])
            )
            for index in range(len(written_times))
        }
        revoke: set[int] = set()
        active_list = sorted(active)
        for index in active_list:
            for other in range(len(written_times)):
                if other == index:
                    continue
                if abs(rewritten[index] - rewritten[other]) <= collision + 1e-9:
                    if other in active:
                        if other > index:
                            revoke.update((index, other))
                    else:
                        revoke.add(index)
                elif other in active and other > index:
                    if rewritten[index] > rewritten[other] + 1e-9:
                        revoke.update((index, other))
        if not revoke:
            break
        active.difference_update(revoke)
    return {
        index: replace(
            proposed[index],
            guards=proposed[index].guards + (("G9_monotonic_order", True),),
        )
        for index in sorted(active)
    }


def consensus_report(
    overrides: Mapping[int, ConsensusOverride],
) -> list[dict[str, object]]:
    return [
        {
            "entry_index": item.entry_index,
            "time": round(item.seconds, 6),
            "source": "multi-model-consensus",
            "support_families": list(item.support_families),
            "model_times": dict(item.model_times),
            "near_written_families": list(item.near_written_families),
            "guards": dict(item.guards),
        }
        for item in overrides.values()
    ]
