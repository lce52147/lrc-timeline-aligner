"""Write a reviewed LRC into an audio file's lyrics tag (FLAC Vorbis ``LYRICS``, M4A ``©lyr``).

Only the tag is edited in place (mutagen); audio is never re-encoded or remuxed. Every write is
guarded: the previous value goes to a JSONL log, a temp copy is taken first, and the result is
verified (tag text equals the LRC, decoded audio and other tags unchanged) or rolled back.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import runtime_paths

SUPPORTED = {".flac": "LYRICS", ".m4a": "©lyr"}
LOG_NAME = "export-log.jsonl"
CACHE_NAME = ".export-cache"
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


class ExportError(Exception):
    pass


def _mutagen():
    try:
        import mutagen.flac
        import mutagen.mp4
    except ImportError as exc:
        raise ExportError("缺少 mutagen 套件，請重新執行 setup.ps1 安裝（pip install mutagen）") from exc
    return mutagen.flac, mutagen.mp4


def lrc_text(data):
    """LRC bytes -> the tag text: UTF-8 (BOM dropped), CRLF line endings, no trailing line break."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExportError("LRC 不是 UTF-8，無法寫入標籤") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    return text.replace("\n", "\r\n")


def _line_count(text):
    return len(re.split(r"\r\n|\r|\n", text)) if text else 0


def tag_name(path):
    suffix = pathlib.Path(path).suffix.casefold()
    if suffix not in SUPPORTED:
        raise ExportError(f"不支援的音檔格式 {suffix or '(無副檔名)'}，只能匯出到 .flac 或 .m4a")
    return SUPPORTED[suffix]


def read_lyrics(path):
    """Embedded lyrics text of a FLAC/M4A, or None."""
    name = tag_name(path)
    flac, mp4 = _mutagen()
    try:
        if name == "LYRICS":
            values = flac.FLAC(str(path)).get("lyrics")
        else:
            tags = mp4.MP4(str(path)).tags
            values = tags.get(name) if tags else None
    except Exception as exc:
        raise ExportError(f"無法讀取標籤：{exc}") from exc
    if not values:
        return None
    return "\r\n".join(str(item) for item in values)


def info(path):
    """What the confirm dialog shows: target name, tag, whether lyrics are already embedded."""
    path = pathlib.Path(path)
    name = tag_name(path)
    existing = read_lyrics(path)
    return {"name": path.name, "tag": name, "has_lyrics": bool(existing), "lines": _line_count(existing or "")}


def candidates(audio):
    """The song's own audio files: same folder and stem, supported extension (the given file first)."""
    audio = pathlib.Path(audio)
    found = []
    for ext in runtime_paths.AUDIO_EXTENSIONS:
        for entry in audio.parent.glob("*") if audio.parent.is_dir() else []:
            if entry.suffix.casefold() == ext and entry.stem.casefold() == audio.stem.casefold() and entry.is_file():
                found.append(entry)
    found.sort(key=lambda item: (item != audio, item.name.casefold()))
    return found


def _decoded_md5(path):
    try:
        result = subprocess.run(
            [runtime_paths.ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "md5", "-"],
            capture_output=True, text=True, timeout=600, check=False, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExportError(f"無法用 ffmpeg 驗證音訊：{exc}") from exc
    line = result.stdout.strip()
    if result.returncode != 0 or not line.startswith("MD5="):
        raise ExportError(f"ffmpeg 驗證失敗：{(result.stderr or line).strip()[:200]}")
    return line[4:]


def _flac_payload_digest(path):
    """sha256 of a FLAC's bytes from the first audio frame to EOF, or None if the metadata cannot be walked."""
    with open(path, "rb") as handle:
        if handle.read(4) != b"fLaC":
            return None
        while True:
            head = handle.read(4)
            if len(head) < 4:
                return None
            last = head[0] & 0x80
            handle.seek(int.from_bytes(head[1:4], "big"), os.SEEK_CUR)
            if last:
                break
        digest = hashlib.sha256()
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _m4a_payload_digest(path):
    """sha256 of the (single) top-level ``mdat`` payload of an MP4, or None if it cannot be found."""
    size_total = os.path.getsize(path)
    found = None
    with open(path, "rb") as handle:
        offset = 0
        while offset + 8 <= size_total:
            handle.seek(offset)
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header = 16
            elif size == 0:
                size = size_total - offset
            if size < header or offset + size > size_total:
                return None
            if kind == b"mdat":
                if found is not None:
                    return None
                found = (offset + header, offset + size)
            offset += size
        if found is None:
            return None
        handle.seek(found[0])
        remaining = found[1] - found[0]
        digest = hashlib.sha256()
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                return None
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _payload_digest(path, name):
    try:
        return _flac_payload_digest(path) if name == "LYRICS" else _m4a_payload_digest(path)
    except (OSError, struct.error):
        return None


def _snapshot(path, name):
    """Everything that must survive the write: audio payload, STREAMINFO md5, other tags, picture sizes.

    The audio is checked byte-for-byte (FLAC frames / MP4 ``mdat``); ffmpeg decoding is only the fallback
    for files whose container cannot be walked.
    """
    flac, mp4 = _mutagen()
    digest = _payload_digest(path, name)
    snap = {"audio_payload": digest} if digest else {"audio_md5": _decoded_md5(path)}
    if name == "LYRICS":
        handle = flac.FLAC(str(path))
        snap["streaminfo_md5"] = handle.info.md5_signature
        snap["tags"] = {key: list(handle[key]) for key in handle.keys() if key.casefold() != "lyrics"}
        snap["pictures"] = [(pic.type, pic.mime, len(pic.data)) for pic in handle.pictures]
    else:
        handle = mp4.MP4(str(path))
        tags = handle.tags or {}
        snap["tags"] = {key: repr(tags[key]) for key in tags.keys() if key != name}
    return snap


def _write_tag(path, name, text):
    flac, mp4 = _mutagen()
    if name == "LYRICS":
        handle = flac.FLAC(str(path))
        handle["LYRICS"] = [text]
        handle.save()
    else:
        handle = mp4.MP4(str(path))
        if handle.tags is None:
            handle.add_tags()
        handle.tags[name] = [text]
        handle.save()


def _log(log_path, record):
    with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export(audio, lrc_bytes, work_dir):
    """Embed ``lrc_bytes`` into ``audio``. ``work_dir`` holds export-log.jsonl and the temp copy.

    Returns {"name", "tag", "lines"}; raises ExportError (file restored) when anything fails.
    """
    audio = pathlib.Path(audio)
    work_dir = pathlib.Path(work_dir)
    name = tag_name(audio)
    if not audio.is_file():
        raise ExportError("找不到音檔")
    if not os.access(audio, os.W_OK):
        raise ExportError("音檔是唯讀，沒有寫入任何東西；請先取消檔案的唯讀屬性再匯出")
    text = lrc_text(lrc_bytes)
    if not text:
        raise ExportError("LRC 是空的，不寫入")
    before = _snapshot(audio, name)
    old = read_lyrics(audio)
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(work_dir / LOG_NAME, {
        "utc": datetime.now(timezone.utc).isoformat(), "file": str(audio), "tag": name,
        "old": old if old else "none", "new_lines": _line_count(text)})
    cache = work_dir / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    backup = cache / (uuid.uuid4().hex + audio.suffix)
    shutil.copy2(audio, backup)
    try:
        try:
            _write_tag(audio, name, text)
            if read_lyrics(audio) != text:
                raise ExportError("寫入後讀回的歌詞和 LRC 不一致")
            after = _snapshot(audio, name)
            if "audio_payload" in before and "audio_payload" not in after:
                raise ExportError("寫入後無法讀出音訊資料")
            for key in ("audio_payload", "audio_md5"):
                if key in before and after.get(key) != before[key]:
                    raise ExportError("寫入後音訊內容改變")
            for key in ("streaminfo_md5", "tags", "pictures"):
                if after.get(key) != before.get(key):
                    raise ExportError(f"寫入後 {key} 改變")
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError(f"寫入標籤失敗：{exc}") from exc
    except ExportError as exc:
        try:
            shutil.copyfile(backup, audio)
        except OSError as restore_exc:
            raise ExportError(f"{exc}；且還原失敗：{restore_exc}（備份在 {backup}）") from exc
        backup.unlink(missing_ok=True)
        raise ExportError(f"{exc}（已還原原檔）") from exc
    backup.unlink(missing_ok=True)
    return {"name": audio.name, "tag": name, "lines": _line_count(text)}


def _recycle(path):
    """Send ``path`` to the Windows Recycle Bin (no dialogs). False when that is not possible."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    op = SHFILEOPSTRUCTW()
    op.wFunc = 3  # FO_DELETE
    op.pFrom = str(path) + "\0"
    op.fFlags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # ALLOWUNDO | NOCONFIRMATION | SILENT | NOERRORUI
    return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)) == 0 and not os.path.exists(path)


def archive_lrc(lrc_path, work_dir, audio_name, setting=None):
    """After a verified export: move (or recycle) the song's ``.lrc`` per ``LRC_EXPORT_ARCHIVE_DIR``.

    Never raises and never touches anything but ``lrc_path`` itself. Returns {"mode", "to"?, "error"?};
    mode ``off`` means the file was left in place. The outcome goes to export-log.jsonl.
    """
    mode, folder = setting if setting is not None else runtime_paths.export_archive_setting()
    result = {"mode": mode}
    lrc_path = pathlib.Path(lrc_path)
    if mode == "off":
        return result
    try:
        if lrc_path.suffix.casefold() != ".lrc" or not lrc_path.is_file():
            raise OSError("not an existing .lrc file")
        if mode == "delete":
            if not _recycle(lrc_path):
                raise OSError("could not send the file to the Recycle Bin")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / lrc_path.name
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            count = 0
            while target.exists():
                suffix = "_" + stamp + ("" if count == 0 else f"-{count}")
                target = folder / (lrc_path.stem + suffix + lrc_path.suffix)
                count += 1
            shutil.move(str(lrc_path), str(target))
            result["to"] = str(target)
    except OSError as exc:
        result["mode"] = "failed"
        result["error"] = str(exc)
    try:
        _log(pathlib.Path(work_dir) / LOG_NAME, {
            "utc": datetime.now(timezone.utc).isoformat(), "file": audio_name, "lrc": str(lrc_path),
            "archive": result["mode"], **{k: v for k, v in result.items() if k in ("to", "error")}})
    except OSError:
        pass
    return result
