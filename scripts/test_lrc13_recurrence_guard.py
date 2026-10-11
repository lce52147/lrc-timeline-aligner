"""P3 recurrence trust must be backed by an independent lineage."""
from __future__ import annotations

from dataclasses import replace
import unittest

import auto_lrc


class GlobalRecurrenceCorroborationTests(unittest.TestCase):
    def candidate(self, source: str, time: float) -> auto_lrc.TimingCandidate:
        recurrence = "recurrence" in source
        artifact = {"source": source, "time": time}
        producer = "global-recurrence-search" if recurrence else "consonant-onset-detector"
        kind = "recurrence-onset" if recurrence else "vocal-onset"
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="repeat lyric",
            source=source,
            raw_time=time,
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.95,
            acoustic_onset_time=time,
            acoustic_source_artifact=artifact,
            acoustic_evidence_producer=producer,
            acoustic_evidence_kind=kind,
            acoustic_evidence_independent=True,
            direct_onset_time=time,
            direct_onset_source_artifact=artifact,
            direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind=kind,
            direct_onset_evidence_independent=True,
            independent_content_identity=recurrence,
            source_artifact=artifact,
        )

    def audited(self, witness: auto_lrc.TimingCandidate | None = None, *,
                witness_valid: bool = True) -> auto_lrc.TimingInvariantAudit:
        recurrence = self.candidate("global-acoustic-recurrence", 10.00)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(recurrence,),
            previous_candidate=None, selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        if witness is not None:
            evidence = auto_lrc.evaluate_timing_candidate(witness)
            self.assertTrue(evidence.valid)
            if not witness_valid:
                evidence = replace(evidence, valid=False, rejection_reasons=("unverified",))
            decision = replace(decision, rejected_candidates=(evidence,))
        return auto_lrc.audit_final_timing_decisions((decision,))[0]

    def test_selected_global_recurrence_without_independent_witness_requires_review(self) -> None:
        audit = self.audited()
        self.assertFalse(audit.timing_trusted)
        self.assertIn("recurrence-uncorroborated", audit.failures)
        self.assertTrue(auto_lrc._timing_failure_requires_review("recurrence-uncorroborated"))

    def test_same_family_recurrence_is_not_independent_corroboration(self) -> None:
        audit = self.audited(self.candidate("mutual-acoustic-recurrence", 10.06))
        self.assertIn("recurrence-uncorroborated", audit.failures)

    def test_distant_other_family_is_not_corroboration(self) -> None:
        audit = self.audited(self.candidate("ctc-vocal-onset-confirmed", 10.45))
        self.assertIn("recurrence-uncorroborated", audit.failures)

    def test_invalid_other_family_is_not_corroboration(self) -> None:
        audit = self.audited(self.candidate("ctc-vocal-onset-confirmed", 10.06),
                             witness_valid=False)
        self.assertIn("recurrence-uncorroborated", audit.failures)

    def test_valid_close_other_family_preserves_recurrence_trust(self) -> None:
        audit = self.audited(self.candidate("ctc-vocal-onset-confirmed", 10.06))
        self.assertTrue(audit.timing_trusted)
        self.assertNotIn("recurrence-uncorroborated", audit.failures)


if __name__ == "__main__":
    unittest.main()
