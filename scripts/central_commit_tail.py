"""Extracted from auto_lrc.py (split phase 2): commit tail: install an evaluated Central selection into the production report."""

from __future__ import annotations

import copy
from pathlib import Path
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY, _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY, _validate_backend_provenance_projection, apply_central_summary_to_report, assert_report_review_contract, finalize_review_summary, summarize_central_timing_state, thaw_generation_diagnostics, timing_report_evaluation_revision
from timing_core import BackendSelection, EvaluatedBackendBundle, EvaluatedBackendTiming
from lyrics_io import LRC_ALGORITHM_REVISION, _BACKEND_PROVENANCE_PROJECTION_KEY, ratio_percent
from runtime_base import LrcError, _path_sha256, _timing_content_digest, validated_loaded_source_sha256
from central_commit import commit_final_timing_state
import argparse
from timing_core import FinalTimingState
from backend_bundle import _backend_projection_error_code, _backend_projection_field, _capture_evaluated_backend_bundle, _install_peer_review_only_conflicts, _peer_review_only_conflict_projection, _postcentral_rebind_existing_peer_conflicts, _public_backend_capability_summary, _public_local_capability_summary, _rebind_peer_review_only_conflicts_to_final_selection, _simulate_cross_backend_peer_graph, central_backend_summary_payload, central_candidate_selection_payload, central_selection_reason, select_evaluated_backend
import lrc_stage_timings


_BACKEND_PROVENANCE_PROJECTION_SCHEMA_VERSION = 1


def _postcentral_existing_conflict_projection(
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object] | None:
    """Replay existing peer conflicts only, after neutral selection is frozen."""
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(seed, dict):
        return None
    if not (
        isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return None
    current_onsets = report.get(_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY)
    if not isinstance(current_onsets, tuple):
        current_onsets = ()
    rebound = _postcentral_rebind_existing_peer_conflicts(
        seed, state, bundles, current_onsets=current_onsets
    )
    details = rebound.get("cross_backend_material_conflicts")
    if not isinstance(details, dict):
        return None
    # Structural safety: this stage may only remove old peer rows/candidates.
    old_details = seed.get("cross_backend_material_conflicts")
    if not isinstance(old_details, dict):
        return None
    old_entries = {
        int(item["entry"])
        for item in old_details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    new_entries = {
        int(item["entry"])
        for item in details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    if not new_entries.issubset(old_entries):
        return None

    def cluster_member_ids(row: object) -> set[str]:
        if not isinstance(row, dict):
            return set()
        output: set[str] = set()
        clusters = row.get("clusters")
        if not isinstance(clusters, list):
            return output
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            ids = cluster.get("member_candidate_ids")
            if isinstance(ids, list):
                output.update(item for item in ids if isinstance(item, str))
        return output

    old_rows = {
        int(item["entry"]): item
        for item in old_details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for row in details.get("entries", []):
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            return None
        old_row = old_rows.get(int(row["entry"]))
        if old_row is None or not cluster_member_ids(row).issubset(
            cluster_member_ids(old_row)
        ):
            return None
    return rebound


def _build_backend_provenance_projection_seed(
    audio_path: Path,
    args: argparse.Namespace,
    generation_candidates: list[dict[str, object]],
    preview_records: list[dict[str, object]],
    selection: BackendSelection,
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]] | None = None,
) -> dict[str, object]:
    # Preview records are the last in-memory point where each backend report,
    # Central scaffold, and runtime sidecar coexist.  Capture immutable bundles
    # here before winner selection discards nonselected report objects.
    for record in preview_records:
        evaluated = record.get("evaluated")
        report = record.get("report")
        if (
            isinstance(evaluated, EvaluatedBackendTiming)
            and isinstance(report, dict)
            and not isinstance(record.get("bundle"), EvaluatedBackendBundle)
        ):
            record["bundle"] = _capture_evaluated_backend_bundle(
                evaluated,
                report,
                audio_path=audio_path,
                args=args,
            )
    bundles = tuple(
        record["bundle"]
        for record in preview_records
        if isinstance(record.get("bundle"), EvaluatedBackendBundle)
    )
    peer_simulation = _simulate_cross_backend_peer_graph(
        bundles,
        selected_backend=selection.selected.backend,
        timing_source_policy=selection.selected.timing_source_policy,
    )
    evaluated_reports = {
        id(record.get("evaluated")): record.get("report")
        for record in preview_records
        if record.get("evaluated") is not None
    }
    candidates: list[dict[str, object]] = []
    for evaluated in selection.candidates:
        report = evaluated_reports.get(id(evaluated))
        summary = _public_backend_capability_summary(
            audio_path, args, evaluated.backend, report if isinstance(report, dict) else None
        )
        summary.update({
            "usable": True,
            "error": False,
            "selection_status": "selected" if evaluated is selection.selected else "nonselected",
            "evaluation": central_backend_summary_payload(evaluated),
            "selection_reason": central_selection_reason(selection),
        })
        candidates.append(summary)

    generation: list[dict[str, object]] = []
    for candidate in generation_candidates:
        backend = str(candidate.get("backend") or "unknown")
        error = candidate.get("error")
        report = candidate.get("report")
        capability_report = report if isinstance(report, dict) else None
        if (
            backend == "whisperx"
            and error
            and whisperx_evidence_candidates
            and isinstance(whisperx_evidence_candidates[0][1], dict)
        ):
            capability_report = whisperx_evidence_candidates[0][1]
        generation.append({
            "backend": backend,
            "usable": not bool(error),
            "error": bool(error),
            "error_code": _backend_projection_error_code(error),
            "capability": _public_backend_capability_summary(
                audio_path, args, backend, capability_report
            ),
            "evidence_candidate_count": (
                len(whisperx_evidence_candidates)
                if backend == "whisperx" and whisperx_evidence_candidates
                else 0
            ),
        })

    try:
        source_audio_sha256 = _path_sha256(audio_path)
    except (OSError, LrcError):
        source_audio_sha256 = None
    local_summary = _public_local_capability_summary(audio_path, args)
    cross_backend_material_conflicts = _peer_review_only_conflict_projection(
        peer_simulation
    )
    return {
        "schema_version": _BACKEND_PROVENANCE_PROJECTION_SCHEMA_VERSION,
        "source_audio": {
            "sha256": _backend_projection_field(source_audio_sha256),
            "source": "input-audio",
        },
        "candidates": candidates,
        "generation_candidates": generation,
        "selection": central_candidate_selection_payload(selection),
        "producer_capabilities": [local_summary] if local_summary is not None else [],
        "algorithm_source_sha256": validated_loaded_source_sha256(),
        "algorithm_revision": LRC_ALGORITHM_REVISION,
        "peer_simulation": peer_simulation,
        "cross_backend_material_conflicts": cross_backend_material_conflicts,
    }


def _finalize_backend_provenance_projection(
    report: dict[str, object],
    seed: dict[str, object],
    selection: BackendSelection,
) -> dict[str, object]:
    projection = copy.deepcopy(seed)
    selected_capability_revision = None
    for candidate in projection.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        evaluation = candidate.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("evaluation_revision") == selection.selected.evaluation_revision:
            capability = candidate.get("capability")
            if isinstance(capability, dict):
                field = capability.get("capability_revision")
                if isinstance(field, dict):
                    selected_capability_revision = field.get("value")
            break

    trusted_rows: list[dict[str, object]] = []
    assignments = report.get("assignments")
    central_state = report.get("central_timing_state")
    if (
        not isinstance(assignments, list)
        or not isinstance(central_state, list)
        or len(assignments) != len(central_state)
    ):
        raise LrcError("Backend provenance row count does not match assignments")
    if isinstance(assignments, list) and isinstance(central_state, list):
        for assignment, state in zip(assignments, central_state, strict=True):
            if not isinstance(assignment, dict) or not isinstance(state, dict):
                continue
            if not bool(assignment.get("overall_trusted")):
                continue
            provenance = assignment.get("candidate_provenance")
            evidence = assignment.get("normalized_timing_evidence")
            direct = provenance.get("direct_onset_evidence") if isinstance(provenance, dict) else None
            acoustic = provenance.get("acoustic_evidence") if isinstance(provenance, dict) else None
            source_revision = None
            if isinstance(direct, dict):
                source_revision = direct.get("source_revision")
            if source_revision is None and isinstance(acoustic, dict):
                source_revision = acoustic.get("source_revision")
            trusted_rows.append({
                "entry": assignment.get("entry"),
                "candidate_id": state.get("candidate_id"),
                "candidate_source": state.get("candidate_source"),
                "candidate_generation_revision": (
                    provenance.get("generation_revision") if isinstance(provenance, dict) else None
                ),
                "candidate_evidence_revision": (
                    evidence.get("candidate_revision") if isinstance(evidence, dict) else None
                ),
                "timing_evidence_revision": assignment.get("timing_evidence_revision"),
                "timestamp_revision": assignment.get("timestamp_revision"),
                "source_revision": source_revision,
                "occurrence_evidence": (
                    provenance.get("occurrence_evidence")
                    if isinstance(provenance, dict) else None
                ),
                "selected_backend_capability_revision": selected_capability_revision,
            })

    projection["selected_backend"] = selection.selected.backend
    projection["selected_backend_capability_revision"] = selected_capability_revision
    # The central evaluator owns this immutable report projection revision.
    # Later public metric refreshes may rebuild diagnostic booleans, so the
    # binding must remain the evaluator revision rather than a post-commit
    # re-projection of the mutable report dictionary.
    projection["report_revision_binding"] = selection.selected.report_projection_revision
    projection["central_state_binding"] = _timing_content_digest(
        report.get("central_timing_state", [])
    )
    projection["selection_binding"] = _timing_content_digest(
        report.get("candidate_selection", {})
    )
    projection["trusted_row_evidence"] = trusted_rows
    projection["projection_status"] = "RETRIEVED"
    projection.pop("projection_revision", None)
    projection["projection_revision"] = _timing_content_digest(projection)
    return projection


def validate_backend_selection(selection: BackendSelection) -> None:
    """Verify that selection provenance is the canonical immutable central result."""
    if not selection.candidates:
        raise LrcError("Backend selection has no centrally evaluated candidates")
    if not any(item is selection.selected for item in selection.candidates):
        raise LrcError("Selected backend is not an identity member of its candidate set")
    canonical = select_evaluated_backend(selection.candidates)
    if (
        len(canonical.candidates) != len(selection.candidates)
        or any(
            expected is not actual
            for expected, actual in zip(canonical.candidates, selection.candidates)
        )
        or canonical.selected is not selection.selected
        or canonical.selection_revision != selection.selection_revision
    ):
        raise LrcError("Backend selection provenance is not the canonical central selection")


def _apply_evaluated_backend_projection(
    selection: BackendSelection,
    report: dict[str, object],
) -> FinalTimingState:
    validate_backend_selection(selection)
    evaluated = selection.selected
    expected_selection_payload = central_candidate_selection_payload(selection)
    existing_selection = report.get("candidate_selection")
    if existing_selection is not None and existing_selection != expected_selection_payload:
        raise LrcError("Candidate selection provenance changed after central selection")
    current_report_revision = timing_report_evaluation_revision(report)
    if current_report_revision != evaluated.report_projection_revision:
        raise LrcError("Report evaluation input changed after central preview")
    diagnostics = thaw_generation_diagnostics(evaluated)
    expected_input_revision = _timing_content_digest({
        "backend": evaluated.backend,
        "report_projection_revision": current_report_revision,
        "generation_diagnostics_semantic_revision": (
            evaluated.generation_diagnostics_semantic_revision
        ),
        "candidate_sets": [
            [candidate.candidate_id for candidate in candidates]
            for candidates in evaluated.candidate_sets
        ],
        **(
            {"arbiter_mode": evaluated.arbiter_mode}
            if evaluated.arbiter_mode != "off"
            else {}
        ),
    })
    if expected_input_revision != evaluated.input_revision:
        raise LrcError("Backend timing input changed after central preview")
    projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(projection_seed, dict):
        review_projection = projection_seed.get("cross_backend_material_conflicts")
        if not (
            isinstance(review_projection, dict)
            and review_projection.get("postcentral_existing_conflict_evidence_replayed") is True
        ):
            projection_seed = _rebind_peer_review_only_conflicts_to_final_selection(
                projection_seed, evaluated.state
            )
        report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = projection_seed
    _install_peer_review_only_conflicts(
        report,
        projection_seed if isinstance(projection_seed, dict) else None,
    )
    committed = commit_final_timing_state(
        evaluated.state,
        report,
        diagnostics,
        finding_states=evaluated.finding_states,
    )
    if committed is not evaluated.state:
        raise LrcError("Central timing commit did not reuse the selected preview state")
    report["candidate_selection"] = expected_selection_payload
    apply_central_summary_to_report(
        report,
        summarize_central_timing_state(report, committed),
    )
    # The evaluated summary may predate final suspicious-row REVIEW changes.
    # Sync only aggregate review fields here: private replays must preserve
    # their immutable Central row timing decisions until the final-output audit.
    committed_assignments = report.get("assignments", [])
    final_trusted_count = sum(
        row.get("verification_status") == "trusted"
        for row in committed_assignments if isinstance(row, dict)
    )
    final_unverified_count = sum(
        row.get("verification_status") == "unverified"
        for row in committed_assignments if isinstance(row, dict)
    )
    total = len(committed_assignments)
    report["unverified_count"] = final_unverified_count
    report["unverified_percent"] = ratio_percent(final_unverified_count, total)
    report["low_confidence_count"] = total - final_trusted_count
    report["low_confidence_percent"] = ratio_percent(total - final_trusted_count, total)
    for key in ("trusted", "overall_trusted", "authority_trusted", "matched"):
        report[key + "_entries"] = final_trusted_count
        report[key + "_percent"] = ratio_percent(final_trusted_count, total)
    finalize_review_summary(report)
    if isinstance(projection_seed, dict):
        report[_BACKEND_PROVENANCE_PROJECTION_KEY] = _finalize_backend_provenance_projection(
            report,
            projection_seed,
            selection,
        )
        _validate_backend_provenance_projection(report)
    assert_report_review_contract(report)
    return committed


def commit_evaluated_backend(
    selection: BackendSelection,
    report: dict[str, object],
) -> FinalTimingState:
    """Install one already-evaluated Central selection into the production report."""
    return _apply_evaluated_backend_projection(selection, report)


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
