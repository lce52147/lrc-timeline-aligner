from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REVIEWERS = SCRIPTS / "reviewers"
for path in (SCRIPTS, REVIEWERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EXPECTED_PACKAGES = {
    "whisperx": ("whisperx", "3.8.6"),
    "demucs": ("demucs", "4.0.1"),
    "numpy": ("numpy", "2.4.4"),
    "pykakasi": ("pykakasi", "2.3.0"),
    "fugashi": ("fugashi", "1.5.2"),
    "unidic-lite": ("unidic_lite", "1.0.8"),
    "transformers": ("transformers", "4.57.6"),
    "huggingface-hub": ("huggingface_hub", "0.36.2"),
    "pyopenjtalk-plus": ("pyopenjtalk", "0.4.1.post3"),
}
WHISPER_MODEL_SHA256 = "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
WHISPER_MODEL_SIZE = 3_095_033_483
HUBP_SHA256 = "f958c23c16a54d9cca9fcc6e453e4f9054b47bcfe3c1d9313cbeb17975483caf"
KNOWN_HUBERT_REPO = "TKU410410103/hubert-base-japanese-asr"
KNOWN_HUBERT_REVISION = "9206637a0f07d78aa9101ff9439a9d7fe1a43778"
KNOWN_HUBERT_SHA256 = "d6f5034777c0542db6c587d57db497206d0500f7eca8fb8bc7962301f45d54d9"
EXPECTED_MODEL_IDENTITIES = {
    "mms": (
        "torchaudio:MMS_FA:model.pt",
        "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680",
    ),
    "mms1b:eng:roman": (
        "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
        "57f4356d7adc674e7fb86dcf4601e0dfd3b327d5fd000af99de2f6c231279d13",
    ),
    "mms1b:jpn:native": (
        "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
        "57f4356d7adc674e7fb86dcf4601e0dfd3b327d5fd000af99de2f6c231279d13",
    ),
    "en-ctc:facebook/wav2vec2-large-960h-lv60-self:roman": (
        "54074b1c16f4de6a5ad59affb4caa8f2ea03a119",
        "ae4aefbb0d05afffb4e020aca0fbd0d8e0460ee95cc91ec19752415f5ea0796e",
    ),
    "en-ctc:jonatasgrosman/wav2vec2-large-xlsr-53-english:roman": (
        "569a6236e92bd5f7652a0420bfe9bb94c5664080",
        "aa4e8474448561d1c8bc9b1c880cb9f5b22e54c5ce8af9a6ba91170641d2d204",
    ),
    "en-ctc:torchaudio:roman": (
        "torchaudio:WAV2VEC2_ASR_LARGE_960H:wav2vec2_fairseq_large_ls960_asr_ls960.pth",
        "f45e55ccb71e0f0c7dc52d519c29524213a2d41da7761f71089f11bf3ea40702",
    ),
    "ja-ctc:jonatasgrosman/wav2vec2-large-xlsr-53-japanese:native": (
        "cf031e020336460d15a417eba710bbc5bb43be9a",
        "f9eff8ae91f4da69acfa1234ba7423d0bd33ce731367c6365ae74a8ea305df3b",
    ),
}
TORCH_CHECKPOINTS = {
    "model.pt": "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680",
    "wav2vec2_fairseq_large_ls960_asr_ls960.pth": "f45e55ccb71e0f0c7dc52d519c29524213a2d41da7761f71089f11bf3ea40702",
    "wav2vec2_fairseq_base_ls960_asr_ls960.pth": "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persistent_hf_environment(scope: str) -> dict[str, str | None]:
    names = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
    values = {name: None for name in names}
    if os.name != "nt":
        return values
    import winreg

    if scope == "User":
        root, key = winreg.HKEY_CURRENT_USER, "Environment"
    elif scope == "Machine":
        root, key = winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    else:
        raise ValueError(scope)
    try:
        with winreg.OpenKey(root, key) as handle:
            for name in names:
                try:
                    values[name] = str(winreg.QueryValueEx(handle, name)[0])
                except FileNotFoundError:
                    pass
    except FileNotFoundError:
        pass
    return values


def _hf_hub_path(values: dict[str, str | None]) -> Path:
    if values.get("HF_HUB_CACHE"):
        raw = str(values["HF_HUB_CACHE"])
    elif values.get("HF_HOME"):
        raw = str(Path(str(values["HF_HOME"])) / "hub")
    else:
        raw = str(Path.home() / ".cache" / "huggingface" / "hub")
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def check_hf_cache_environment() -> str:
    names = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
    scopes = {
        "Process": {name: os.environ.get(name) for name in names},
        "User": _persistent_hf_environment("User"),
        "Machine": _persistent_hf_environment("Machine"),
    }
    report: dict[str, object] = {}
    mismatches: list[str] = []
    for scope, values in scopes.items():
        hub = _hf_hub_path(values)
        transformers_cache = values.get("TRANSFORMERS_CACHE")
        report[scope] = {**values, "resolved_hub": str(hub)}
        if transformers_cache:
            legacy = Path(os.path.expandvars(transformers_cache)).expanduser().resolve()
            if legacy != hub:
                mismatches.append(f"{scope}:{legacy} != {hub}")
    if mismatches:
        raise RuntimeError("TRANSFORMERS_CACHE differs from hub: " + "; ".join(mismatches))
    return json.dumps(report, sort_keys=True)


class Doctor:
    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []

    def check(self, name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.results.append({"name": name, "status": "FAIL", "detail": detail})
            print(f"FAIL {name}: {detail}", flush=True)
            return
        self.results.append({"name": name, "status": "PASS", "detail": detail})
        print(f"PASS {name}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.results)


def _run(command: list[str], timeout: float = 30.0) -> str:
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"exit={proc.returncode}: {output[-800:]}")
    return output


def check_python() -> str:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"requires Python 3.11, got {sys.version.split()[0]}")
    return f"{sys.version.split()[0]} executable={Path(sys.executable).resolve()}"


def check_package(dist: str, module: str, expected: str) -> str:
    actual = importlib.metadata.version(dist)
    if actual != expected:
        raise RuntimeError(f"{dist} {actual} != {expected}")
    importlib.import_module(module)
    return actual


def check_torch(expect_cuda: str) -> str:
    import torch
    import torchaudio

    expected_suffix = "+cu128" if expect_cuda == "yes" else "+cpu" if expect_cuda == "no" else ""
    if not torch.__version__.startswith("2.8.0"):
        raise RuntimeError(f"torch version changed: {torch.__version__}")
    if not torchaudio.__version__.startswith("2.8.0"):
        raise RuntimeError(f"torchaudio version changed: {torchaudio.__version__}")
    if expected_suffix and not torch.__version__.endswith(expected_suffix):
        raise RuntimeError(f"torch build {torch.__version__} does not match expected {expected_suffix}")
    if expect_cuda == "yes" and not torch.cuda.is_available():
        raise RuntimeError("CUDA host selected but torch.cuda.is_available() is false")
    return (
        f"torch={torch.__version__} torchaudio={torchaudio.__version__} "
        f"cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}"
    )


def check_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH")
    output = _run([path, "-version"], timeout=20.0)
    first = output.splitlines()[0] if output else ""
    return f"{Path(path).resolve()} | {first}"


def check_whisper_cli() -> str:
    path = ROOT / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe"
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    output = _run([str(path), "--help"], timeout=30.0)
    return f"{path} sha256={sha256_file(path)} help={output.splitlines()[0] if output else 'ok'}"


def check_whisper_model() -> str:
    path = ROOT / "models" / "whisper.cpp" / "ggml-large-v3.bin"
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    if path.stat().st_size != WHISPER_MODEL_SIZE:
        raise RuntimeError(f"size {path.stat().st_size} != {WHISPER_MODEL_SIZE}")
    actual = sha256_file(path)
    if actual != WHISPER_MODEL_SHA256:
        raise RuntimeError(f"sha256 {actual} != {WHISPER_MODEL_SHA256}")
    return f"size={path.stat().st_size} sha256={actual}"


def check_demucs() -> str:
    output = _run([sys.executable, "-m", "demucs", "--help"], timeout=30.0)
    return output.splitlines()[0] if output else "help ok"


def _torch_checkpoint_dir() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home) / "hub" / "checkpoints"
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def check_torch_checkpoints() -> str:
    root = _torch_checkpoint_dir()
    for name, expected in TORCH_CHECKPOINTS.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name} sha256 {actual} != {expected}")
    return str(root)


def check_multimodel_identities() -> str:
    import multimodel_consensus as mm

    mm.model_identity.cache_clear()
    for spec in mm.MODEL_SPECS:
        identity = mm.model_identity(spec)
        revision = identity.get("model_revision")
        blob = identity.get("model_blob_sha256")
        if revision == "UNVERIFIED" or blob == "UNVERIFIED":
            raise RuntimeError(f"{spec}: identity unresolved")
        expected = EXPECTED_MODEL_IDENTITIES.get(spec)
        if expected is None:
            raise RuntimeError(f"{spec}: no expected identity")
        if (revision, blob) != expected:
            raise RuntimeError(f"{spec}: {(revision, blob)} != {expected}")
    return f"{len(mm.MODEL_SPECS)}/{len(mm.MODEL_SPECS)} verified"


def check_reviewer_models() -> str:
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCTC, AutoProcessor
    import japanese_ctc_align as jactc
    import run_xlsr

    revision, _config_sha, _weight_sha = run_xlsr.local_model_identity()
    if revision != "cf031e020336460d15a417eba710bbc5bb43be9a":
        raise RuntimeError(f"XLSR revision moved: {revision}")
    processor = AutoProcessor.from_pretrained(
        jactc.MODEL_NAME, revision=revision, local_files_only=True
    )
    model = AutoModelForCTC.from_pretrained(
        jactc.MODEL_NAME, revision=revision, local_files_only=True
    )
    del processor, model
    gc.collect()

    hubp = ROOT / "models" / "hubert-phoneme-ctc"
    hubp_weight = hubp / "model.safetensors"
    if not hubp_weight.is_file() or sha256_file(hubp_weight) != HUBP_SHA256:
        raise RuntimeError("HuBERT phoneme reviewer model missing or wrong hash")
    processor = AutoProcessor.from_pretrained(hubp, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(hubp, local_files_only=True, use_safetensors=True)
    del processor, model
    gc.collect()

    known = try_to_load_from_cache(
        KNOWN_HUBERT_REPO, "model.safetensors", revision=KNOWN_HUBERT_REVISION
    )
    if not isinstance(known, str) or sha256_file(Path(known)) != KNOWN_HUBERT_SHA256:
        raise RuntimeError("known-lyric HuBERT model missing or wrong hash")
    return f"XLSR={revision}; HUBP={HUBP_SHA256[:12]}; known-HuBERT={KNOWN_HUBERT_REVISION[:12]}"


def check_whisperx_alignment_identity() -> str:
    from whisperx_refine import model_provenance

    japanese = model_provenance("ja")
    if not japanese:
        raise RuntimeError("Japanese WhisperX alignment identity unavailable")
    if japanese.get("align_model_revision") != "cf031e020336460d15a417eba710bbc5bb43be9a":
        raise RuntimeError(f"Japanese WhisperX revision mismatch: {japanese}")
    return json.dumps(japanese, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the LRC Tools v1.3.1 runtime.")
    parser.add_argument("--prefetch", action="store_true", help="Require every prefetched model.")
    parser.add_argument("--expect-cuda", choices=("yes", "no", "auto"), default="auto")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    doctor = Doctor()
    doctor.check("python", check_python)
    doctor.check("hf-cache-environment", check_hf_cache_environment)
    for dist, (module, expected) in EXPECTED_PACKAGES.items():
        doctor.check(f"package:{dist}", lambda d=dist, m=module, e=expected: check_package(d, m, e))
    doctor.check("torch", lambda: check_torch(args.expect_cuda))
    doctor.check("ffmpeg", lambda: check_command("ffmpeg"))
    doctor.check("ffprobe", lambda: check_command("ffprobe"))
    doctor.check("whisper-cli", check_whisper_cli)
    doctor.check("whisper-model", check_whisper_model)
    doctor.check("demucs", check_demucs)
    if args.prefetch:
        doctor.check("torchaudio-checkpoints", check_torch_checkpoints)
        doctor.check("multimodel-identities", check_multimodel_identities)
        doctor.check("reviewer-models", check_reviewer_models)
        doctor.check("whisperx-align-ja", check_whisperx_alignment_identity)

    payload = {
        "schema": 1,
        "python_executable": str(Path(sys.executable).resolve()),
        "expect_cuda": args.expect_cuda,
        "prefetch_required": bool(args.prefetch),
        "results": doctor.results,
        "ok": not doctor.failed,
    }
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DOCTOR PASS" if not doctor.failed else "DOCTOR FAIL", flush=True)
    return 1 if doctor.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
