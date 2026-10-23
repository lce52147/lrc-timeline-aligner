import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import file_digest


class FileDigestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_DIR": str(self.root / "cache")})
        self.env.start()
        file_digest._MEMO.clear()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_matches_hashlib_and_rehashes_after_rewrite(self):
        path = self.root / "a.bin"
        path.write_bytes(b"first")
        self.assertEqual(file_digest.sha256(path), hashlib.sha256(b"first").hexdigest())
        path.write_bytes(b"other")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertEqual(file_digest.sha256(path), hashlib.sha256(b"other").hexdigest())

    def test_unchanged_file_is_read_once(self):
        path = self.root / "a.bin"
        path.write_bytes(b"data")
        with mock.patch.object(file_digest, "hash_bytes", wraps=file_digest.hash_bytes) as spy:
            file_digest.sha256(path)
            file_digest.sha256(path)
        self.assertEqual(spy.call_count, 1)

    def test_large_file_digest_survives_a_new_process(self):
        path = self.root / "model.bin"
        path.write_bytes(b"weights")
        with mock.patch.object(file_digest, "DISK_MIN_BYTES", 1):
            first = file_digest.sha256(path)
            file_digest._MEMO.clear()
            with mock.patch.object(file_digest, "hash_bytes", side_effect=AssertionError("re-read")):
                self.assertEqual(file_digest.sha256(path), first)
        self.assertTrue(file_digest.memo_path("file-sha256.json").is_file())

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            file_digest.sha256(self.root / "missing.bin")


if __name__ == "__main__":
    unittest.main()
