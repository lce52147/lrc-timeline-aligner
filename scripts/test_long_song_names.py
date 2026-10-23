import sys
import unittest
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r2_pipeline

WINDOWS_MAX_PATH = 260
STAGE_SUFFIX = len(".0123456789ab.stage")


class LongSongNameTests(unittest.TestCase):
    def test_baseline_report_stage_fits_windows_max_path(self) -> None:
        # The 2026-10-22 failure: this song's baseline staging path was over 260 characters.
        base = PureWindowsPath(r"D:\Users\Administrator\Documents\LRC tools\outputs\reviewer-work")
        audio = Path("/music/Rosa Walton&Hallie Coggins-I Really Want to Stay at Your House.flac")
        run_dir = r2_pipeline.reviewer_work_directory(Path("/work"), audio)
        self.assertLessEqual(len(run_dir.name), r2_pipeline._WORK_NAME_MAX + len("--") + 12)
        self.assertTrue(run_dir.name.endswith("--" + run_dir.name[-12:]))
        report_name = f"{audio.stem}--0123456789.align-report.json"
        staged = base / run_dir.name / "baseline-reports" / report_name
        self.assertLess(len(str(staged)) + STAGE_SUFFIX, WINDOWS_MAX_PATH)

    def test_short_names_keep_their_work_folder(self) -> None:
        audio = Path("/music/02. 明弦音.flac")
        self.assertTrue(r2_pipeline.reviewer_work_directory(Path("/work"), audio).name.startswith("02. 明弦音--"))

    def test_same_prefix_songs_stay_separate(self) -> None:
        first = Path("/music/" + "x" * 40 + " A.flac")
        second = Path("/music/" + "x" * 40 + " B.flac")
        self.assertNotEqual(
            r2_pipeline.reviewer_work_directory(Path("/work"), first),
            r2_pipeline.reviewer_work_directory(Path("/work"), second),
        )

    def test_publication_stage_names_are_short(self) -> None:
        import tempfile
        import lrc_publication
        seen = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def write_lrc(stage: Path) -> None:
                seen.append(stage.name)
                stage.write_text("[00:01.00]a\n", encoding="utf-8")
            def write_report(stage: Path) -> None:
                seen.append(stage.name)
                stage.write_text("{}", encoding="utf-8")
            lrc_publication.publish_pair(root / "song.lrc", root / "song.align-report.json", write_lrc, write_report)
        self.assertEqual([name.rsplit(".", 2)[1] for name in seen], [name.rsplit(".", 2)[1][:12] for name in seen])
        self.assertTrue(all(len(name.rsplit(".", 2)[1]) == 12 for name in seen))


if __name__ == "__main__":
    unittest.main()
