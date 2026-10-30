"""Hot update of the review server: real server processes on a copy of the scripts, code edited while they run."""
from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
HIDDEN = 0x08000000 if os.name == "nt" else 0


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Fixture:
    def __init__(self):
        base = Path(os.environ.get("LRC_HOT_TEST_ROOT", "F:/_RuntimePrompt/LRC15/BE/tests"))
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="lrc-hot-test-", dir=str(base)))
        self.scripts = self.root / "scripts"
        shutil.copytree(SCRIPTS, self.scripts, ignore=shutil.ignore_patterns("test_*", "__pycache__", "*.pyc"))
        (self.root / "outputs" / "reports").mkdir(parents=True)
        self.port = free_port()
        self.proc = None
        self.token = ""

    def start(self, extra_env=None):
        env = dict(os.environ, PYTHONUTF8="1", LRC_HOT_UPDATE_INTERVAL="0.3", LRC_NO_REVIEW="1", **(extra_env or {}))
        self.proc = subprocess.Popen([sys.executable, str(self.scripts / "review_server.py"), "--report-dir", str(self.root / "outputs" / "reports"),
                                      "--port", str(self.port), "--no-browser"], cwd=str(self.root), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, creationflags=HIDDEN)
        info = json.loads(self.proc.stdout.readline().decode("utf-8"))
        self.token = info["token"]
        return info

    def get(self, path, headers=None, timeout=10):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Host": "127.0.0.1:%d" % self.port, **(headers or {})})
            response = conn.getresponse()
            body = response.read()
            return response.status, (json.loads(body) if body[:1] in (b"{", b"[") else body)
        finally:
            conn.close()

    def admin(self):
        try:
            status, body = self.get("/api/admin/version")
            return body if status == 200 else None
        except OSError:
            return None

    def version(self):
        return self.get("/api/version", {"X-Review-Token": self.token})

    def wait_for(self, predicate, seconds=40):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(0.2)
        return None

    def log(self):
        path = self.root / "outputs" / "review-server.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def edit(self, name, text):
        path = self.scripts / name
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")

    def cli(self, *args, timeout=240):
        return subprocess.run([sys.executable, str(self.scripts / "review_hot_update.py"), *args], capture_output=True, text=True, encoding="utf-8",
                              timeout=timeout, creationflags=HIDDEN)

    def close(self):
        pids = {self.proc.pid} if self.proc else set()
        info = self.admin()
        if info:
            pids.add(info["pid"])
        for pid in pids:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=HIDDEN)
        shutil.rmtree(self.root, ignore_errors=True)


class HotUpdateTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)
        self.fx.start()
        self.first = self.fx.admin()
        self.assertIsNotNone(self.first)

    def test_page_is_served_with_its_asset_version(self):
        status, page = self.fx.get("/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"__ASSET_VERSION__", page)
        self.assertIn(self.first["asset_version"].encode(), page)

    def test_admin_endpoint_rejects_browser_origin_and_other_host(self):
        self.assertEqual(self.fx.get("/api/admin/version", {"Origin": "http://evil.example"})[0], 403)
        conn = http.client.HTTPConnection("127.0.0.1", self.fx.port, timeout=5)
        conn.request("GET", "/api/admin/version", headers={"Host": "evil.example"})
        self.assertEqual(conn.getresponse().status, 403)
        conn.close()

    def test_asset_change_is_reported_without_a_restart(self):
        self.fx.edit("review_page/index.html", "\n<!-- changed -->\n")
        info = self.fx.wait_for(lambda: (self.fx.admin() or {}).get("asset_version") != self.first["asset_version"] and self.fx.admin())
        self.assertTrue(info)
        self.assertEqual(info["pid"], self.first["pid"])
        self.assertNotEqual(self.fx.version()[1]["asset_version"], self.first["asset_version"])
        time.sleep(1)
        self.assertEqual(self.fx.admin()["pid"], self.first["pid"])

    def test_code_change_while_idle_reexecs_on_same_port_with_same_token(self):
        self.fx.edit("review_server.py", "\n# hot update test marker\n")
        info = self.fx.wait_for(lambda: (self.fx.admin() or {}).get("pid") not in (None, self.first["pid"]) and self.fx.admin())
        self.assertTrue(info, self.fx.log())
        self.assertNotEqual(info["code_version"], self.first["code_version"])
        status, payload = self.fx.version()  # old token still valid: the page reconnects on the same URL
        self.assertEqual(status, 200)
        self.assertEqual(payload["code_version"], info["code_version"])
        log = self.fx.log()
        self.assertIn("result=ok", log)
        self.assertIn("idle_wait_ms=", log)

    def test_change_during_an_inflight_request_waits_then_updates(self):
        # a POST whose body never arrives is a request in flight (a save under way looks the same to the server)
        sock = socket.create_connection(("127.0.0.1", self.fx.port), timeout=5)
        sock.sendall(("POST /api/song/0/save HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nX-Review-Token: %s\r\nOrigin: http://127.0.0.1:%d\r\n"
                      "Content-Type: application/json\r\nContent-Length: 20\r\n\r\n{" % (self.fx.port, self.fx.token, self.fx.port)).encode())
        time.sleep(0.5)
        self.fx.edit("review_server.py", "\n# marker while busy\n")
        time.sleep(4)
        self.assertEqual(self.fx.admin()["pid"], self.first["pid"], "must not restart while a request is in flight")
        self.assertFalse(self.fx.admin()["idle"])
        sock.sendall(b'"a":1}' + b" " * 13)  # finish the body: the request completes (invalid) and the server becomes idle
        sock.recv(4096)
        sock.close()
        info = self.fx.wait_for(lambda: (self.fx.admin() or {}).get("pid") not in (None, self.first["pid"]) and self.fx.admin())
        self.assertTrue(info, self.fx.log())

    def test_failed_reexec_keeps_the_old_server(self):
        self.fx.edit("review_server.py", "\ndef broken(:\n")
        notice = self.fx.wait_for(lambda: (self.fx.admin() or {}).get("update_failed"))
        self.assertTrue(notice, self.fx.log())
        info = self.fx.admin()
        self.assertEqual(info["pid"], self.first["pid"])
        self.assertEqual(self.fx.version()[0], 200)
        self.assertIn("result=failed", self.fx.log())
        self.assertEqual(self.fx.version()[1]["update_failed"], "自動更新失敗，請手動重開")
        time.sleep(2)
        self.assertEqual(self.fx.log().count("result=failed"), 1, "a failed signature is not retried")

    def test_failure_after_the_port_was_freed_rebinds_the_old_server(self):
        # the replacement loads, the old server frees the port, the replacement then dies before binding: the old one must come back
        self.fx.edit("review_hot_update.py", "\nimport os as _o\nif _o.environ.get('LRC_HOT_HANDOFF'):\n    _real = replacement_bound\n"
                                             "    def replacement_bound():\n        _o._exit(3)\n")
        info = self.fx.wait_for(lambda: (self.fx.admin() or {}).get("update_failed"), 60)
        self.assertTrue(info, self.fx.log())
        self.assertEqual(self.fx.admin()["pid"], self.first["pid"])
        self.assertEqual(self.fx.version()[0], 200)

    def test_cli_pushes_the_update_into_the_running_server(self):
        listed = self.fx.cli("--list")
        self.assertIn("pid %d port %d" % (self.first["pid"], self.fx.port), listed.stdout)
        self.assertIn("already on the current code", self.fx.cli().stdout)
        self.fx.edit("review_server.py", "\n# cli marker\n")
        result = self.fx.cli("--wait", "60")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # the server's own watcher may have swapped itself before the CLI looked (test interval 0.3 s): both outcomes mean the new code runs
        self.assertTrue("hot-updated in place" in result.stdout or "already on the current code" in result.stdout, result.stdout)
        self.assertNotEqual(self.fx.admin()["pid"], self.first["pid"])
        self.assertEqual(self.fx.version()[0], 200)


class LegacyServerTests(unittest.TestCase):
    def test_cli_restarts_a_server_that_has_no_hot_update_on_the_same_port(self):
        fx = Fixture()
        self.addCleanup(fx.close)
        fx.start({"LRC_NO_HOT_UPDATE": "1"})
        self.assertIsNone(fx.admin())
        old_pid = fx.proc.pid
        result = fx.cli("--wait", "120")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("restarted", result.stdout)
        info = fx.wait_for(fx.admin)
        self.assertTrue(info)
        self.assertNotEqual(info["pid"], old_pid)
        self.assertEqual(info["port"], fx.port)

    def test_cli_reports_not_running(self):
        fx = Fixture()
        self.addCleanup(fx.close)
        self.assertIn("not running", fx.cli().stdout)


if __name__ == "__main__":
    unittest.main()
