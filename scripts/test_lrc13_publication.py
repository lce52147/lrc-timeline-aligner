import unittest
import lrc_publication

class PublicationTests(unittest.TestCase):
    def test_module_has_recovery(self):
        self.assertTrue(callable(lrc_publication.recover_pair))
    def test_pair_success(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),
                                         lambda p:p.write_text('{"new":true}'))
            self.assertEqual(a.read_text(),"new")
            self.assertIn("new",b.read_text())
    def test_writer_failure_does_not_change_old_pair(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            a.write_text("old")
            b.write_text('{"old":true}')
            def broken(path):
                raise OSError("staging failed")
            with self.assertRaises(OSError):
                lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),broken)
            self.assertEqual(a.read_text(),"old")
            self.assertIn("old",b.read_text())
    def test_second_rename_failure_rolls_back(self):
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            a.write_text("old")
            b.write_text('{"old":true}')
            def rename(src,dst):
                if Path(dst)==b:
                    raise OSError("rename failed")
                os.replace(src,dst)
            with self.assertRaises(OSError):
                lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),
                                             lambda p:p.write_text('{"new":true}'),replace=rename)
            self.assertEqual(a.read_text(),"old")
            self.assertIn("old",b.read_text())
    def test_first_rename_failure_rolls_back(self):
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            a.write_text("old")
            b.write_text('{"old":true}')
            def rename(src,dst):
                if Path(dst)==a:
                    raise OSError("rename failed")
                os.replace(src,dst)
            with self.assertRaises(OSError):
                lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),
                                             lambda p:p.write_text('{"new":true}'),replace=rename)
            self.assertEqual(a.read_text(),"old")
            self.assertIn("old",b.read_text())
    def test_initial_second_failure_removes_new_files(self):
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            def rename(src,dst):
                if Path(dst)==b:
                    raise OSError("rename failed")
                os.replace(src,dst)
            with self.assertRaises(OSError):
                lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),
                                             lambda p:p.write_text('{"new":true}'),replace=rename)
            self.assertFalse(a.exists())
            self.assertFalse(b.exists())
    def test_interrupted_publish_recovers_next_invocation(self):
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"song.lrc"
            b=Path(directory)/"song.json"
            a.write_text("old")
            b.write_text('{"old":true}')
            def interrupt(src,dst):
                if Path(dst)==b:
                    raise KeyboardInterrupt("interrupted")
                os.replace(src,dst)
            with self.assertRaises(KeyboardInterrupt):
                lrc_publication.publish_pair(a,b,lambda p:p.write_text("new"),
                                             lambda p:p.write_text('{"new":true}'),replace=interrupt)
            self.assertEqual(a.read_text(),"new")
            lrc_publication.recover_pair(a,b)
            self.assertEqual(a.read_text(),"old")
            self.assertIn("old",b.read_text())
