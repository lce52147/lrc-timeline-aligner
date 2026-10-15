"""Capture Central inputs for all 39 songs (cache-only), two songs at a time.

usage (repo root): python -X utf8 _review/opus3_central_capture_all_20261015.py <out root> [workers]
Skips songs whose central_capture.pkl already exists.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def one(index: int) -> str:
    out = root / str(index)
    if (out / "central_capture.pkl").is_file():
        return f"{index} skip"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "capture.log").open("wb") as log:
        code = subprocess.run(
            [sys.executable, "-X", "utf8", str(V / "opus3_central_capture_20261015.py"), str(index), str(out)],
            cwd=V.parent, stdout=log, stderr=subprocess.STDOUT,
        ).returncode
    return f"{index} exit {code}"


with ThreadPoolExecutor(workers) as pool:
    for line in pool.map(one, range(1, 40)):
        print(line, flush=True)
