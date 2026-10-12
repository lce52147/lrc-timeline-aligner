"""P3: trust refresh on a private Central shadow must rebind provenance."""
from __future__ import annotations
from pathlib import Path
from unittest import TestCase, mock
import auto_lrc
import test_core_logic


class PrivateShadowProvenanceTests(TestCase):
    def test_neutral_shadow_trust_refresh_rebinds_immutable_provenance(self):
        fixture = test_core_logic.CentralTimingStateTests()
        candidate = fixture.candidate(12.0, fixture.coherent_spans(12.0), current=True)
        report = fixture.report(1)
        evaluated = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), report, {})
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        report[auto_lrc._RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = {
            "schema_version": 1,
            "algorithm_source_sha256": auto_lrc.validated_loaded_source_sha256(),
            "candidates": [],
            "cross_backend_material_conflicts": None,
        }
        real_refresh = auto_lrc.update_report_confidence_metrics

        def simulated_postcommit_trust_change(shadow):
            real_refresh(shadow)
            assignment = shadow["assignments"][0]
            central = shadow["central_timing_state"][0]
            assignment["timing_trusted"] = False
            assignment["timing_audit_trusted"] = False
            assignment["overall_trusted"] = False
            assignment["review_required"] = True
            assignment["verification_status"] = "review-required"
            assignment["timestamp_status"] = "provisional_unresolved"
            assignment["provisional_candidate_id"] = central["candidate_id"]
            assignment["selected_candidate_id"] = None
            central["timing_trusted"] = False
            central["timestamp_status"] = "provisional_unresolved"
            shadow["review_required"] = True
            shadow["review_required_count"] = 1
            shadow["review_required_percent"] = 100.0

        def verify_probe(_audio, _entries, shadow, _duration, _args, _state):
            auto_lrc._validate_backend_provenance_projection(shadow)
            return {"probe_revision": "synthetic", "entries": []}, {}

        with (
            mock.patch.object(auto_lrc, "update_report_confidence_metrics",
                              side_effect=simulated_postcommit_trust_change),
            mock.patch.object(auto_lrc, "_build_unresolved_evidence_requirements_projection",
                              return_value={"status": "RETRIEVED"}),
            mock.patch.object(auto_lrc, "run_r2t_unified_neutral_conflict_probe",
                              side_effect=verify_probe),
            mock.patch.object(auto_lrc, "_neutral_probe_trigger_rows", return_value=[]),
        ):
            auto_lrc._precommit_neutral_boundary_arbitration(
                Path("unused.flac"), [auto_lrc.LyricEntry(["synthetic"])],
                report, 25.0, auto_lrc.build_parser().parse_args(["unused.flac"]), selection,
            )


if __name__ == "__main__":
    import unittest
    unittest.main()