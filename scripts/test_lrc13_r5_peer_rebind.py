from __future__ import annotations

import unittest

import auto_lrc
from test_core_logic import BackendPeerSimulationTests


class PeerRebindMaterialityTests(unittest.TestCase):
    def test_final_near_distant_cluster_keeps_peer_baseline_conflict(self) -> None:
        def cluster(
            relation: str,
            review: bool,
            revision: str,
            anchor_time: float,
        ) -> dict[str, object]:
            return {
                "cluster_revision": revision,
                "anchor_time": anchor_time,
                "relation": relation,
                "review_required": review,
                "explicit_falsification": False,
                "member_candidate_ids": [revision],
                "member_generation_revisions": [revision],
                "support_families": ["ctc"],
            }

        peer = {
            "status": "SUCCESS",
            "authority": False,
            "simulation_revision": "peer-revision",
            "bundles": [
                {
                    "valid": True,
                    "scaffold_lyric_valid": True,
                    "bundle_revision": "bundle-a",
                }
            ],
            "context_validation": {"valid_context_count": 1},
            "rows": [
                {
                    "entry": 1,
                    "candidate_id": "peer-baseline",
                    "hypothetical_centiseconds": 100,
                    "material_revision": "mat-1",
                    "cluster_revision": "clusters-1",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, "base-1", 1.0),
                        cluster("distant", True, "distant-1", 2.0),
                    ],
                }
            ],
        }
        projection = auto_lrc._peer_review_only_conflict_projection(peer)

        helper = BackendPeerSimulationTests()
        final_candidate = helper.candidate(
            0, 2.0, current=True, source="ctc-current", direct=True
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(final_candidate,),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )

        rebound = auto_lrc._rebind_peer_review_only_conflicts_to_final_selection(
            {"cross_backend_material_conflicts": projection},
            state,
        )
        details = rebound["cross_backend_material_conflicts"]
        self.assertEqual(details["entries"][0]["entry"], 1)
        self.assertEqual(details["entries"][0]["peer_selected_delta_centiseconds"], 100)
        self.assertTrue(details["entries"][0]["peer_selected_baseline_conflict"])


if __name__ == "__main__":
    unittest.main()
