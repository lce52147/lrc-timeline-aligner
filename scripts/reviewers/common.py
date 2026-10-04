from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_songs(payload: dict[str, Any], song_ids: Sequence[str]) -> list[dict[str, Any]]:
    songs = payload.get("songs")
    if not isinstance(songs, list) or not all(isinstance(song, dict) for song in songs):
        raise ValueError("producer payload has invalid songs")
    typed_songs: list[dict[str, Any]] = songs
    if not song_ids:
        return typed_songs

    by_id = {str(song.get("id")): song for song in typed_songs}
    missing = [song_id for song_id in song_ids if song_id not in by_id]
    if missing:
        raise ValueError(f"unknown song id: {missing[0]}")
    return [by_id[song_id] for song_id in song_ids]
