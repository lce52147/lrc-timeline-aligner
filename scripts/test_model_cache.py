"""Tests for ``scripts/model_cache.py`` and ``migrate_model_cache.py`` (LRC 1.5 W3)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402
import migrate_model_cache  # noqa: E402
import model_cache as mc  # noqa: E402

GOLDEN = SCRIPTS / "test_fixtures" / "model_cache_golden_v2.json"
_CLEAN_ENV = {"LRC_MODEL_CACHE": "1", "LRC_MODEL_CACHE_ONLY": "0"}


def _sample_identity(**overrides: object) -> dict[str, object]:
    identity = mc.identity_raw_asr(
        stem_sha256="a" * 64, cli_sha256="b" * 64, model_sha256="c" * 64,
        language="ja", suppress_nst=False,
    )
    identity.update(overrides)
    return identity


class GoldenKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(self.fixture["key_revision"], mc.MODEL_CACHE_KEY_REVISION)

    def test_all_kinds_present(self) -> None:
        self.assertEqual(set(self.fixture["cases"]), set(mc.ALL_KINDS))

    def test_identity_keyed_kinds(self) -> None:
        for kind in sorted(mc.IDENTITY_KEYED_KINDS | mc.WHOLE_IDENTITY_KEYED_KINDS):
            case = self.fixture["cases"][kind]
            self.assertEqual(mc.key_from_identity(kind, case["identity"]), case["expected_key"], kind)

    def test_raw_asr_key_differs_from_legacy_path_bound_key(self) -> None:
        case = self.fixture["cases"]["raw-asr"]
        self.assertNotEqual(case["expected_key"], case["legacy_v1_key"])

    def test_multimodel_key_unchanged(self) -> None:
        case = self.fixture["cases"]["multimodel-consensus"]
        inputs = case["key_inputs"]
        self.assertEqual(
            auto_lrc._model_cache_key(
                inputs["kind"], inputs["stem_sha256"], inputs["model_revision"],
                inputs["model_blob_sha256"], inputs["helper_protocol_revision"], inputs["request"],
            ),
            case["expected_key"],
        )

    def test_local_whisper_key_unchanged(self) -> None:
        case = self.fixture["cases"]["local-whisper"]
        inputs = case["key_inputs"]
        got = auto_lrc._local_whisper_cache_key(
            SimpleNamespace(execution_revision=inputs["request_revision"]),
            SimpleNamespace(capability_revision=inputs["capability_revision"], protocol_revision=inputs["protocol_revision"]),
        )
        self.assertEqual(got, case["expected_key"])

    def test_crossline_key_unchanged(self) -> None:
        case = self.fixture["cases"]["crossline-joint-ctc"]
        inputs = case["key_inputs"]
        with mock.patch.object(auto_lrc, "_ctc_model_blob_sha256", return_value=inputs["model_blob_sha256"]):
            got = auto_lrc._crossline_retry_helper_cache_key(
                SimpleNamespace(
                    alignment_audio_revision=inputs["alignment_audio_sha256"],
                    transcript_digest=inputs["transcript_digest"],
                    window_start=inputs["window_start"], window_end=inputs["window_end"],
                ),
                SimpleNamespace(
                    helper_sha256=inputs["helper_sha256"],
                    helper_protocol_revision=inputs["helper_protocol_revision"],
                    model_identity=inputs["model_identity"], device=inputs["device"],
                ),
            )
        self.assertEqual(got, case["expected_key"])

    def test_externally_keyed_kinds_reject_key_from_identity(self) -> None:
        for kind in mc.EXTERNALLY_KEYED_KINDS:
            with self.assertRaises(ValueError):
                mc.key_from_identity(kind, self.fixture["cases"][kind]["identity"])

    def test_key_revision_decoupled_from_envelope_schema(self) -> None:
        self.assertEqual(mc.MODEL_CACHE_SCHEMA, 2)
        self.assertEqual(mc.MODEL_CACHE_KEY_REVISION, 1)
        self.assertEqual(auto_lrc.MODEL_CACHE_SCHEMA, 2)


class PortableIdentityTests(unittest.TestCase):
    def test_paths_are_rejected(self) -> None:
        for bad in ("D:\\x\\whisper-cli.exe", "C:/models/x.bin", "/home/user/x", "\\\\server\\share"):
            with self.assertRaises(mc.NonPortableIdentity, msg=bad):
                mc.assert_portable_identity(_sample_identity(model_revision=bad))
            with self.assertRaises(mc.NonPortableIdentity, msg=bad):
                mc.assert_portable_identity(_sample_identity(request={"cli": bad}))

    def test_project_root_text_is_rejected(self) -> None:
        with self.assertRaises(mc.NonPortableIdentity):
            mc.assert_portable_identity(_sample_identity(note=f"root={auto_lrc.PROJECT_ROOT}"))

    def test_clean_identity_passes(self) -> None:
        mc.assert_portable_identity(_sample_identity())

    def test_lyric_text_fields_may_look_like_paths(self) -> None:
        identity = _sample_identity(request={"transcript": [{"text": "A:/ not a path"}], "lines": ["/slash"]})
        mc.assert_portable_identity(identity)
        identity = mc.identity_whole_ctc(
            stem_sha256="a" * 64, model_revision="r", model_blob_sha256="b" * 64,
            helper_protocol_revision="p", helper_sha256="h" * 64,
            entries=["A:/weird lyric", "\\\\double"], duration=1.0, device="cuda", source="vocal-stem",
        )
        mc.assert_portable_identity(identity)

    def test_all_identity_builders_are_portable_with_weird_paths(self) -> None:
        weird = "Z:\\weird path\\"
        builders = [
            mc.identity_raw_asr(stem_sha256="a" * 64, cli_sha256="b" * 64, model_sha256="c" * 64, language="ja", suppress_nst=True),
            mc.identity_whisperx_alignment(stem_sha256="a" * 64, model_revision="r", model_blob_sha256="b" * 64,
                                           helper_protocol_revision="p", helper_sha256="h" * 64,
                                           transcript=[{"text": weird}], language="ja", device="cuda"),
            mc.identity_whole_ctc(stem_sha256="a" * 64, model_revision="r", model_blob_sha256="b" * 64,
                                  helper_protocol_revision="p", helper_sha256="h" * 64,
                                  entries=[weird], duration=2.0, device="cuda", source="vocal-stem"),
            mc.identity_local_window_ctc(stem_sha256="a" * 64, model_revision="r", model_blob_sha256="b" * 64,
                                         helper_protocol_revision="p", helper_sha256="h" * 64,
                                         request={"lyric_texts": [weird], "start": 0.0, "end": 1.0, "device": "cuda"}),
            mc.identity_multimodel_consensus(model_revision="r", model_blob_sha256="b" * 64, protocol_revision="p", models=["m"]),
            mc.identity_local_whisper(audio_sha256="a" * 64, whisper_model_sha256="c" * 64, protocol_revision="p", request_revision="rev"),
            mc.identity_crossline_joint_ctc(alignment_audio_sha256="a" * 64, model_identity="m", model_blob_sha256="b" * 64,
                                            helper_protocol_revision="p", transcript_digest="t", window_start=0.0,
                                            window_end=1.0, context_entry_indexes=[0]),
            mc.identity_r2_reviewer_sidecar(payload_sha256="a" * 64, reviewers_sha256="b" * 64, whisper_cli_sha256="c" * 64,
                                            whisper_model_sha256="d" * 64, device="cuda", include_hub=False),
        ]
        self.assertEqual(len(builders), len(mc.ALL_KINDS))
        for identity in builders:
            mc.assert_portable_identity(identity)

    def test_production_raw_asr_identity_has_no_cli_path(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            cli = root / "Z weird" / "whisper-cli.exe"
            cli.parent.mkdir()
            cli.write_bytes(b"cli")
            model = root / "ggml.bin"
            model.write_bytes(b"model")
            wav = root / "input.wav"
            wav.write_bytes(b"pcm")
            args = argparse.Namespace(whisper_language="ja")
            key, identity = auto_lrc._whispercpp_cache_identity(wav, args, suppress_nst=False, cli=cli, model=model)
        mc.assert_portable_identity(identity)
        self.assertNotIn("cli", identity["request"])
        self.assertEqual(identity["request"]["cli_sha256"], identity["model_revision"].split(":", 1)[1])
        self.assertEqual(key, mc.key_from_identity("raw-asr", identity))


class PolicyTests(unittest.TestCase):
    def test_policy_from_env_and_args(self) -> None:
        with mock.patch.dict(os.environ, _CLEAN_ENV):
            self.assertEqual(mc.policy_from_env_and_args(None).mode, "read-write")
            self.assertEqual(mc.policy_from_env_and_args(SimpleNamespace(model_cache=False)).mode, "off")
            self.assertEqual(mc.policy_from_env_and_args(SimpleNamespace(model_cache_only=True)).mode, "cache-only")
        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE": "0"}):
            self.assertEqual(mc.policy_from_env_and_args(SimpleNamespace(model_cache=True)).mode, "off")
        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE": "1", "LRC_MODEL_CACHE_ONLY": "1"}):
            self.assertEqual(mc.policy_from_env_and_args(SimpleNamespace(model_cache_only=False)).mode, "cache-only")

    def test_cache_only_read_miss_raises_without_env(self) -> None:
        identity = _sample_identity()
        key = mc.key_from_identity("raw-asr", identity)
        with TemporaryDirectory() as td, mock.patch.dict(os.environ, _CLEAN_ENV):
            policy = mc.policy_from_env_and_args(argparse.Namespace(model_cache=True, model_cache_only=True))
            with self.assertRaises(mc.ModelCacheMiss):
                mc.read("raw-asr", key, identity, policy, Path(td))
            self.assertFalse(mc.write("raw-asr", key, identity, {"x": 1}, policy, Path(td)))
            self.assertEqual(list(Path(td).rglob("*.json")), [])
            # non-cache-only: miss is None, write succeeds, read round-trips
            rw = mc.policy_from_env_and_args(argparse.Namespace(model_cache=True, model_cache_only=False))
            self.assertIsNone(mc.read("raw-asr", key, identity, rw, Path(td)))
            self.assertTrue(mc.write("raw-asr", key, identity, {"x": 1}, rw, Path(td)))
            self.assertEqual(mc.read("raw-asr", key, identity, rw, Path(td)), {"x": 1})
            self.assertEqual(mc.read("raw-asr", key, identity, policy, Path(td)), {"x": 1})

    def test_schema_1_envelope_is_a_miss_and_left_untouched(self) -> None:
        identity = _sample_identity()
        key = mc.key_from_identity("raw-asr", identity)
        with TemporaryDirectory() as td, mock.patch.dict(os.environ, _CLEAN_ENV):
            path = mc.cache_path("raw-asr", key, Path(td))
            path.parent.mkdir(parents=True)
            text = json.dumps({"schema": 1, "kind": "raw-asr", "key": key, "identity": identity, "payload": [1]})
            path.write_text(text, encoding="utf-8")
            rw = mc.ModelCachePolicy(enabled=True, read_only=False)
            self.assertIsNone(mc.read("raw-asr", key, identity, rw, Path(td)))
            with self.assertRaises(mc.ModelCacheMiss):
                mc.read("raw-asr", key, identity, mc.ModelCachePolicy(True, True), Path(td))
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_non_portable_identity_rejected_on_write(self) -> None:
        identity = _sample_identity(model_revision="D:\\cli.exe")
        with TemporaryDirectory() as td:
            with self.assertRaises(mc.NonPortableIdentity):
                mc.write("raw-asr", "k", identity, {}, mc.ModelCachePolicy(), Path(td))

    def test_crossline_cache_only_miss_raises_from_args_only(self) -> None:
        # Condition: LRC_MODEL_CACHE_ONLY unset; only args.model_cache_only=True.
        with TemporaryDirectory() as td, mock.patch.dict(os.environ, _CLEAN_ENV), \
                mock.patch.object(auto_lrc, "DEFAULT_MODEL_CACHE_DIR", Path(td)), \
                mock.patch.object(auto_lrc, "_ctc_model_blob_sha256", return_value="b" * 64):
            policy = mc.policy_from_env_and_args(argparse.Namespace(model_cache=True, model_cache_only=True))
            request = SimpleNamespace(
                alignment_audio_revision="a" * 64, transcript_digest="t", window_start=0.0, window_end=1.0,
                context_entry_indexes=(0,), retry_epoch_revision="e", request_revision="r", entry_index=0,
            )
            capability = SimpleNamespace(helper_sha256="h" * 64, helper_protocol_revision="p",
                                         model_identity="m", device="cuda")
            with self.assertRaises(mc.ModelCacheMiss):
                auto_lrc.invoke_crossline_retry(request, capability, [], helper_cache={}, cache_policy=policy)

    def test_segment_skeleton_cache_only_miss_raises_from_args_only(self) -> None:
        import segment_skeleton_production as ssp

        with TemporaryDirectory() as td, mock.patch.dict(os.environ, _CLEAN_ENV), \
                mock.patch.object(auto_lrc, "DEFAULT_MODEL_CACHE_DIR", Path(td)), \
                mock.patch.object(auto_lrc, "_ctc_model_blob_sha256", return_value="b" * 64):
            root = Path(td)
            stem = root / "stem.wav"
            stem.write_bytes(b"pcm")
            helper = root / "ctc_align.py"
            helper.write_text("# helper", encoding="utf-8")
            transcript = root / "t.json"
            transcript.write_text(json.dumps(["la"]), encoding="utf-8")
            command = ["python", str(helper), "--audio", str(stem), "--transcript", str(transcript),
                       "--output", str(root / "o.json"), "--device", "cuda", "--start", "0.0", "--end", "1.0"]
            runner = mock.Mock(side_effect=AssertionError("helper launched in cache-only mode"))
            cached_helper = ssp.build_cached_helper(
                audio_api=auto_lrc, ctc_helper=helper, device="cuda", helper_runner=runner,
                cache_args=argparse.Namespace(model_cache=True, model_cache_only=True),
            )
            with self.assertRaises(mc.ModelCacheMiss):
                cached_helper(command, timeout=1.0)
            runner.assert_not_called()


class MigrationTests(unittest.TestCase):
    def _legacy(self, kind: str, key: str, identity: dict, payload: object) -> dict:
        return {"schema": 1, "kind": kind, "key": key, "identity": identity, "payload": payload}

    def test_migration_recomputes_only_raw_asr(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
        raw_identity = json.loads(json.dumps(golden["raw-asr"]["identity"]))
        legacy_request = dict(raw_identity["request"])
        legacy_request.pop("cli_sha256")
        legacy_request["cli"] = "D:\\lrc\\tools\\whisper.cpp\\Release\\whisper-cli.exe"
        raw_identity["request"] = legacy_request
        lw_identity = dict(golden["local-whisper"]["identity"])
        lw_identity["model_revision"] = "D:\\lrc\\models\\whisper.cpp\\ggml-large-v3.bin"
        mm_identity = golden["multimodel-consensus"]["identity"]
        payloads = {
            "raw-asr": [{"start": 1.0, "end": 2.0, "text": "ら"}],
            "local-whisper": {"status": "ok", "value": [1, 2, 3]},
            "multimodel-consensus": {"rows": [{"x": 1.5}]},
        }
        with TemporaryDirectory() as td:
            src = Path(td) / "src"
            dst = Path(td) / "dst"
            for kind, identity, legacy_key in (
                ("raw-asr", raw_identity, golden["raw-asr"]["legacy_v1_key"]),
                ("local-whisper", lw_identity, golden["local-whisper"]["expected_key"]),
                ("multimodel-consensus", mm_identity, golden["multimodel-consensus"]["expected_key"]),
            ):
                path = mc.cache_path(kind, legacy_key, src)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._legacy(kind, legacy_key, identity, payloads[kind]), sort_keys=True), encoding="utf-8")
            src_snapshot = {p: p.read_bytes() for p in src.rglob("*.json")}

            summary = migrate_model_cache.migrate_directory(src, dst)

            self.assertEqual(summary["total_key_collisions"], 0)
            self.assertEqual(summary["total_skipped"], 0)
            self.assertEqual(summary["total_written"], 3)
            self.assertTrue(summary["kinds"]["raw-asr"]["key_recomputed"])
            self.assertFalse(summary["kinds"]["local-whisper"]["key_recomputed"])
            self.assertEqual({p: p.read_bytes() for p in src.rglob("*.json")}, src_snapshot)
            self.assertTrue((dst / "migration_report.json").is_file())

            raw_path = mc.cache_path("raw-asr", golden["raw-asr"]["expected_key"], dst)
            self.assertTrue(raw_path.is_file())
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema"], 2)
            self.assertEqual(raw["identity"], golden["raw-asr"]["identity"])
            self.assertEqual(raw["payload"], payloads["raw-asr"])

            lw = json.loads(mc.cache_path("local-whisper", golden["local-whisper"]["expected_key"], dst).read_text(encoding="utf-8"))
            self.assertEqual(lw["identity"]["model_revision"], golden["local-whisper"]["identity"]["model_revision"])
            self.assertEqual(lw["payload"], payloads["local-whisper"])
            mm = json.loads(mc.cache_path("multimodel-consensus", golden["multimodel-consensus"]["expected_key"], dst).read_text(encoding="utf-8"))
            self.assertEqual(mm["identity"], mm_identity)
            for kind, path in (("raw-asr", raw_path),):
                mc.assert_portable_identity(json.loads(path.read_text(encoding="utf-8"))["identity"])
            for envelope in (lw, mm):
                mc.assert_portable_identity(envelope["identity"])
            # payload JSON bytes identical
            for kind, expected_key in (("raw-asr", golden["raw-asr"]["expected_key"]),
                                       ("local-whisper", golden["local-whisper"]["expected_key"]),
                                       ("multimodel-consensus", golden["multimodel-consensus"]["expected_key"])):
                got = json.loads(mc.cache_path(kind, expected_key, dst).read_text(encoding="utf-8"))["payload"]
                self.assertEqual(json.dumps(got, sort_keys=True), json.dumps(payloads[kind], sort_keys=True))
            # new envelopes are readable through the public API
            policy = mc.ModelCachePolicy(True, True)
            self.assertEqual(mc.read("raw-asr", golden["raw-asr"]["expected_key"], golden["raw-asr"]["identity"], policy, dst), payloads["raw-asr"])

            # second run: existing dst files are skipped and counted
            again = migrate_model_cache.migrate_directory(src, dst)
            self.assertEqual(again["total_written"], 0)
            self.assertEqual(again["total_skipped"], 3)
            self.assertEqual(again["kinds"]["raw-asr"]["skip_reasons"], {"dst-exists": 1})

    def test_dry_run_writes_nothing(self) -> None:
        identity = _sample_identity()
        with TemporaryDirectory() as td:
            src = Path(td) / "src"
            path = mc.cache_path("raw-asr", "legacy", src)
            path.parent.mkdir(parents=True)
            identity["request"] = {**identity["request"]}
            identity["request"].pop("cli_sha256")
            identity["request"]["cli"] = "C:/x.exe"
            path.write_text(json.dumps(self._legacy("raw-asr", "legacy", identity, [])), encoding="utf-8")
            summary = migrate_model_cache.migrate_directory(src, Path(td) / "dst", dry_run=True)
            self.assertEqual(summary["total_written"], 1)
            self.assertFalse((Path(td) / "dst").exists())


if __name__ == "__main__":
    unittest.main()
