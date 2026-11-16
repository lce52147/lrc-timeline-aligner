"""A journal that cannot be deleted after a finished publication never rolls the published pair back."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lrc_publication

_real_unlink = Path.unlink


class JournalLockTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lrc = self.root / "song.lrc"
        self.report = self.root / "song.align-report.json"
        self.lrc.write_text("old", encoding="utf-8")
        self.report.write_text('{"old": true}', encoding="utf-8")
        self.journal = lrc_publication.journal_path(self.report)
        self.sleeps = []
        patcher = mock.patch.object(lrc_publication.time, "sleep", side_effect=self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _publish(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            lrc_publication.publish_pair(self.lrc, self.report, lambda p: p.write_text("new", encoding="utf-8"),
                                         lambda p: p.write_text('{"new": true}', encoding="utf-8"))
        return err.getvalue()

    def _locked_journal(self, failures):
        """Path.unlink raising PermissionError for the journal ``failures`` times (None: always)."""
        state = {"left": failures}

        def unlink(path, *args, **kwargs):
            if Path(path).name.endswith(".txn.json") and (state["left"] is None or state["left"] > 0):
                if state["left"] is not None:
                    state["left"] -= 1
                raise PermissionError(13, "locked by another process")
            return _real_unlink(path, *args, **kwargs)

        return mock.patch.object(Path, "unlink", autospec=True, side_effect=unlink)

    def _leftovers(self):
        return sorted(path.name for path in self.root.iterdir() if path.name not in (self.lrc.name, self.report.name))

    def test_a_locked_journal_that_frees_up_is_retried_and_removed(self):
        with self._locked_journal(2):
            warning = self._publish()
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self._leftovers(), [])
        self.assertEqual(warning, "")
        self.assertEqual(self.sleeps, list(lrc_publication.JOURNAL_REMOVE_DELAYS[:2]))

    def test_a_journal_that_stays_locked_keeps_the_new_pair_and_is_marked_committed(self):
        with self._locked_journal(None):
            warning = self._publish()
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertIn("journal cannot be deleted", warning)
        self.assertEqual(self._leftovers(), [self.journal.name])
        self.assertTrue(json.loads(self.journal.read_text(encoding="utf-8"))["committed"])
        self.assertEqual(self.sleeps, list(lrc_publication.JOURNAL_REMOVE_DELAYS))

    def test_next_start_removes_a_committed_journal_without_rolling_back(self):
        with self._locked_journal(None):
            self._publish()
        self.assertTrue(lrc_publication.recover_pair(self.lrc, self.report))
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self._leftovers(), [])
        self.assertFalse(lrc_publication.recover_pair(self.lrc, self.report))

    def test_recovery_with_the_journal_still_locked_does_not_fail_or_roll_back(self):
        with self._locked_journal(None):
            self._publish()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertTrue(lrc_publication.recover_pair(self.lrc, self.report))
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("journal cannot be deleted", err.getvalue())
        # a later publication replaces the leftover journal and succeeds
        self._publish()
        self.assertEqual(self._leftovers(), [])

    def test_a_publication_that_stopped_before_the_commit_mark_still_rolls_back(self):
        def crash(src, dst):
            if Path(dst) == self.report:
                raise OSError("crash between the two replaces")
            Path(src).replace(dst)

        with self.assertRaises(OSError):
            lrc_publication.publish_pair(self.lrc, self.report, lambda p: p.write_text("new", encoding="utf-8"),
                                         lambda p: p.write_text('{"new": true}', encoding="utf-8"), replace=crash)
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "old")
        self.assertIn("old", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self._leftovers(), [])

    def test_a_crash_after_the_replaces_but_before_the_mark_is_rolled_back_on_next_start(self):
        # the process dies right after the two replaces: a plain journal (no committed flag) is still on disk
        manifest = {"schema": 1, "lrc": str(self.lrc.resolve()), "report": str(self.report.resolve()),
                    "backups": {"lrc": str(self.root / "song.lrc.x.backup"), "report": str(self.root / "song.align-report.json.x.backup")},
                    "stages": {}}
        (self.root / "song.lrc.x.backup").write_text("old", encoding="utf-8")
        (self.root / "song.align-report.json.x.backup").write_text('{"old": true}', encoding="utf-8")
        self.lrc.write_text("new", encoding="utf-8")
        self.report.write_text('{"new": true}', encoding="utf-8")
        self.journal.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(lrc_publication.recover_pair(self.lrc, self.report))
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "old")
        self.assertEqual(self._leftovers(), [])

    def _journal_held_open_after_creation(self):
        """os.replace onto the journal works once (its creation) and then fails: another process holds it open without delete sharing."""
        real_replace = lrc_publication.os.replace
        state = {"journal_replaces": 0}

        def replace(src, dst, *args, **kwargs):
            if Path(dst) == self.journal:
                state["journal_replaces"] += 1
                if state["journal_replaces"] > 1:
                    raise PermissionError(32, "The process cannot access the file because it is being used by another process")
            return real_replace(src, dst, *args, **kwargs)

        return mock.patch.object(lrc_publication.os, "replace", side_effect=replace)

    def test_a_journal_held_open_from_its_creation_still_publishes_and_leaves_a_commit_marker(self):
        with self._journal_held_open_after_creation(), self._locked_journal(None):
            warning = self._publish()
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertIn("journal cannot be deleted", warning)
        marker = lrc_publication.committed_marker_path(self.journal)
        self.assertEqual(self._leftovers(), sorted([self.journal.name, marker.name]))
        self.assertFalse(json.loads(self.journal.read_text(encoding="utf-8")).get("committed"))
        # the next start reads the marker: the pair is kept, journal and marker go
        self.assertTrue(lrc_publication.recover_pair(self.lrc, self.report))
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertIn("new", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self._leftovers(), [])
        self.assertFalse(lrc_publication.recover_pair(self.lrc, self.report))

    def test_a_journal_held_open_that_frees_up_is_deleted_with_its_marker(self):
        with self._journal_held_open_after_creation():
            warning = self._publish()
        self.assertEqual(warning, "")
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "new")
        self.assertEqual(self._leftovers(), [])

    def test_a_stray_commit_marker_never_makes_the_next_publication_look_committed(self):
        marker = lrc_publication.committed_marker_path(self.journal)
        marker.write_text("committed", encoding="utf-8")
        self.assertFalse(lrc_publication.recover_pair(self.lrc, self.report))
        self.assertFalse(marker.exists())
        marker.write_text("committed", encoding="utf-8")
        def crash(src, dst):
            if Path(dst) == self.report:
                raise OSError("crash between the two replaces")
            Path(src).replace(dst)

        with self.assertRaises(OSError):
            lrc_publication.publish_pair(self.lrc, self.report, lambda p: p.write_text("new", encoding="utf-8"),
                                         lambda p: p.write_text('{"new": true}', encoding="utf-8"), replace=crash)
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "old")
        self.assertEqual(self._leftovers(), [])

    def test_when_neither_the_journal_nor_a_marker_can_be_written_the_publication_rolls_back(self):
        marker = lrc_publication.committed_marker_path(self.journal)
        real_write = Path.write_text

        def write_text(path, *args, **kwargs):
            if Path(path) == marker:
                raise PermissionError(13, "refused")
            return real_write(path, *args, **kwargs)

        with self._journal_held_open_after_creation(), mock.patch.object(Path, "write_text", autospec=True, side_effect=write_text):
            with self.assertRaises(OSError):
                self._publish()
        self.assertEqual(self.lrc.read_text(encoding="utf-8"), "old")
        self.assertIn("old", self.report.read_text(encoding="utf-8"))
        self.assertEqual(self._leftovers(), [])


if __name__ == "__main__":
    unittest.main()
