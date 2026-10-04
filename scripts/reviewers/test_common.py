from __future__ import annotations

import unittest

from common import select_songs


class ReviewerCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "songs": [
                {"id": "song-a", "title": "A"},
                {"id": "song-b", "title": "B"},
                {"id": "song-c", "title": "C"},
            ]
        }

    def test_select_songs_defaults_to_all_in_input_order(self) -> None:
        selected = select_songs(self.payload, [])
        self.assertEqual([song["id"] for song in selected], ["song-a", "song-b", "song-c"])

    def test_select_songs_accepts_arbitrary_subset_in_requested_order(self) -> None:
        selected = select_songs(self.payload, ["song-c", "song-a"])
        self.assertEqual([song["id"] for song in selected], ["song-c", "song-a"])

    def test_select_songs_rejects_unknown_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown song id"):
            select_songs(self.payload, ["song-z"])


if __name__ == "__main__":
    unittest.main()
