"""Hot update of a running Review LRC server: the server side (watcher + in-place hand-over) and the CLI that pushes a formal ff into it.

Server side: HotUpdater records the code version at start, watches it every few seconds off the store lock and, when the code changed and
the server is idle, starts a replacement process on the same port with the same token and arguments. The replacement loads its own store
first, then the old server frees the port, the replacement binds it and the old process exits. When the replacement fails at any step the
old server binds its port again and keeps serving; a red notice goes to the page.

CLI (``python scripts\\review_hot_update.py``): finds the running servers of this install, asks each one (loopback only) to update
itself when idle and waits for the new process on the same port. A server that predates hot update is restarted from its command line.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HANDOFF_ENV = "LRC_HOT_HANDOFF"
TOKEN_ENV = "LRC_REVIEW_TOKEN"
HIDDEN_FLAGS = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
PAGE_FILE = SCRIPTS_DIR / "review_page" / "index.html"
LIGHT_PATHS = ("/api/version", "/api/status", "/api/jobs", "/api/admin/version", "/api/admin/hot-update")
LOG_NAME = "review-server.log"


def _sha(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        try:
            digest.update(str(path.name).encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        except OSError:
            digest.update(b"missing\0")
    return digest.hexdigest()[:12]


def code_files():
    """The scripts of this install that the running process has imported (plus review_server.py itself)."""
    files = {SCRIPTS_DIR / "review_server.py", SCRIPTS_DIR / "review_hot_update.py"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            try:
                path = pathlib.Path(name).resolve()
            except OSError:
                continue
            if path.parent == SCRIPTS_DIR and path.suffix == ".py":
                files.add(path)
    return files


def code_signature(extra=()):
    return _sha(set(code_files()) | set(extra))


def asset_signature():
    return _sha([PAGE_FILE])


def git_head(root):
    """Short git HEAD of the install, read from .git without running git; '' when unknown."""
    try:
        git = pathlib.Path(root) / ".git"
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            try:
                head = (git / ref).read_text(encoding="utf-8").strip()
            except OSError:
                for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref):
                        head = line.split()[0]
                        break
        return head[:8]
    except OSError:
        return ""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HotUpdater:
    """Watches the code of this install and swaps the running server for a fresh process when it is safe."""

    def __init__(self, server, argv=None, interval=5.0, quiet_s=1.0, ready_timeout=180.0, bound_timeout=20.0, log_dir=None, start=True):
        self.server = server
        self.root = SCRIPTS_DIR.parent
        self.argv = list(argv if argv is not None else sys.argv[1:])
        self.interval, self.quiet_s = interval, quiet_s
        self.ready_timeout, self.bound_timeout = ready_timeout, bound_timeout
        self.log_dir = pathlib.Path(log_dir) if log_dir else None
        self.started_code = code_signature()
        self.started_head = git_head(self.root)
        self.started_asset = asset_signature()
        self.updating = False
        self.handing_over = False
        self.failed = ""
        self.failed_signature = ""
        self.requested = threading.Event()
        self.resume = threading.Event()
        self.resumed_server = None
        self._inflight = 0
        self._idle_since = time.monotonic()
        self._count_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hot-update", daemon=True)
        if start:
            self._thread.start()

    # -- request accounting (called by the handler) --------------------------------------------------
    def enter(self):
        with self._count_lock:
            self._inflight += 1

    def leave(self):
        with self._count_lock:
            self._inflight -= 1
            if self._inflight <= 0:
                self._idle_since = time.monotonic()

    @property
    def inflight(self):
        return self._inflight

    # -- state -----------------------------------------------------------------------------------------
    def info(self):
        return {"pid": os.getpid(), "port": self.server.server_port, "code_version": self.started_code, "head": self.started_head,
                "asset_version": asset_signature(), "updating": self.updating, "update_failed": self.failed,
                "code_current": code_signature() == self.started_code, "idle": self.idle_reason() is None}

    def version_label(self, code=None, head=None):
        return "%s@%s" % (head if head is not None else self.started_head or "nogit", code if code is not None else self.started_code)

    def idle_reason(self):
        """None when the server may be replaced now, else why not."""
        store = self.server.store
        if self._inflight > 0:
            return "request in flight"
        if time.monotonic() - self._idle_since < self.quiet_s:
            return "recent request"
        if store.export_status.get("active"):
            return "export active"
        if (store.reload_progress or {}).get("active"):
            return "reload active"
        jobs = getattr(self.server, "jobs", None)
        if jobs is not None:
            with jobs.lock:
                if jobs.current is not None or any(item.get("status") in ("queued", "running") for job in jobs.jobs for item in job.get("items", [])):
                    return "alignment job active"
        return None

    def log(self, line):
        target = self.log_dir or self.server.store.report_dir.parent
        try:
            with open(pathlib.Path(target) / LOG_NAME, "a", encoding="utf-8", newline="\n") as handle:
                handle.write("%s hot-update %s\n" % (_now(), line))
        except OSError:
            pass

    # -- watcher ----------------------------------------------------------------------------------------
    def request(self):
        """Ask for an immediate check (POST /api/admin/hot-update); returns the state the caller should expect."""
        if self.updating:
            return "updating"
        current = code_signature()
        if current == self.started_code:
            return "current"
        if current == self.failed_signature:
            return "failed"
        self.requested.set()
        return "scheduled"

    def _run(self):
        while not self._stop.is_set():
            self.requested.wait(self.interval)
            self.requested.clear()
            if self._stop.is_set() or self.updating:
                continue
            try:
                current = code_signature()
                if current != self.started_code and current != self.failed_signature:  # a signature that already failed is not retried until it changes
                    self.apply()
            except Exception as exc:  # the watcher must never take the server down
                self._fail("watcher error: %s" % exc)

    def stop(self):
        self._stop.set()
        self.requested.set()

    def _fail(self, reason):
        self.failed = "自動更新失敗，請手動重開"
        self.failed_signature = code_signature()
        self.updating = False
        self.log("old=%s result=failed reason=%s" % (self.version_label(), reason))

    # -- the update itself --------------------------------------------------------------------------------
    def apply(self, wait_limit=None):
        """Wait until idle, then replace this process. Returns normally only when the update did not happen."""
        old = self.version_label()
        waited = time.monotonic()
        store = self.server.store
        while True:
            if self._stop.is_set():
                return False
            if code_signature() == self.started_code:
                return False  # the change was reverted meanwhile
            if self.idle_reason() is None and store.lock.acquire(timeout=0):
                break
            if wait_limit is not None and time.monotonic() - waited > wait_limit:
                return False
            time.sleep(0.2)
        idle_ms = round((time.monotonic() - waited) * 1000)
        self.resume.clear()
        self.resumed_server = None
        new_code, new_head = code_signature(), git_head(self.root)
        self.updating = True
        self.failed = ""
        child = None
        try:
            if self._inflight > 0:
                raise RuntimeError("request arrived while taking the store lock")
            handoff = pathlib.Path(tempfile.mkdtemp(prefix="lrc-hot-"))
            child = self._spawn(handoff)
            if not self._wait_file(handoff / "ready", child, self.ready_timeout):
                raise RuntimeError("replacement did not become ready (exit=%s)" % child.poll())
            self.handing_over = True
            self.server.shutdown()      # the serve loop in the main thread returns and waits for self.resume
            self.server.server_close()  # frees the port for the replacement
            (handoff / "go").write_text("go", encoding="utf-8")
            if not self._wait_file(handoff / "bound", child, self.bound_timeout):
                raise RuntimeError("replacement did not bind the port (exit=%s)" % child.poll())
            self.log("old=%s new=%s idle_wait_ms=%d result=ok pid=%d->%d" % (old, self.version_label(new_code, new_head), idle_ms, os.getpid(), child.pid))
            os._exit(0)
        except Exception as exc:
            if child is not None and child.poll() is None:
                child.kill()
            if self.handing_over:
                try:
                    self.resumed_server = self.server = self._rebind()
                finally:
                    self.handing_over = False
            self._fail("%s (old=%s new=%s idle_wait_ms=%d)" % (exc, old, self.version_label(new_code, new_head), idle_ms))
            self.resume.set()
            return False
        finally:
            try:
                store.lock.release()
            except RuntimeError:
                pass

    def _spawn(self, handoff):
        argv = list(self.argv)
        # the replacement must bind the port this server is on, not the one that was asked for on the command line
        stripped, skip = [], False
        for item in argv:
            if skip:
                skip = False
            elif item == "--port":
                skip = True
            elif not item.startswith("--port="):
                stripped.append(item)
        cmd = [sys.executable, str(SCRIPTS_DIR / "review_server.py"), *stripped, "--port", str(self.server.server_port), "--no-browser"]
        env = dict(os.environ)
        env[HANDOFF_ENV] = str(handoff)
        env[TOKEN_ENV] = self.server.token
        env["PYTHONUTF8"] = "1"
        return subprocess.Popen(cmd, cwd=str(self.root), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=HIDDEN_FLAGS if os.name == "nt" else 0, close_fds=True)

    @staticmethod
    def _wait_file(path, child, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if path.exists():
                return True
            if child.poll() is not None:
                return path.exists()
            time.sleep(0.05)
        return False

    def _rebind(self):
        """The replacement failed after this server freed its port: bind it again and keep serving with the same state."""
        import review_server
        port = self.server.server_port
        end = time.monotonic() + 10
        fresh = None
        while fresh is None:
            try:
                fresh = review_server.ReviewHTTPServer(("127.0.0.1", port), self.server.RequestHandlerClass)
            except OSError:
                if time.monotonic() > end:
                    fresh = review_server.bind_server(port)
                else:
                    time.sleep(0.2)
        for name in ("store", "jobs", "token", "port_notice", "hot"):
            setattr(fresh, name, getattr(self.server, name, None))
        return fresh


def replacement_handshake(port):
    """Run by the replacement process once its store is loaded: tell the old server, wait for the port to be freed.
    Returns True when this process may serve, False when it must exit. A no-op (True) when not started by a hot update."""
    directory = os.environ.get(HANDOFF_ENV)
    if not directory:
        return True
    handoff = pathlib.Path(directory)
    try:
        (handoff / "ready").write_text("ready", encoding="utf-8")
    except OSError:
        return False
    end = time.monotonic() + 60
    while time.monotonic() < end:
        if (handoff / "go").exists():
            return True
        time.sleep(0.05)
    return False


def replacement_bound():
    directory = os.environ.get(HANDOFF_ENV)
    if directory:
        try:
            (pathlib.Path(directory) / "bound").write_text("bound", encoding="utf-8")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------

def _argv_of(command_line):
    """Split a Windows command line like CommandLineToArgvW does."""
    if os.name != "nt":
        import shlex
        return shlex.split(command_line)
    shell32 = ctypes.windll.shell32
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    count = ctypes.c_int(0)
    pointer = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(pointer)


def _powershell(script):
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, creationflags=0x08000000 if os.name == "nt" else 0)
    return result.stdout


def _review_port(port):
    """True when a Review LRC server answers on 127.0.0.1:port (it says so in its Server header, even to a request without a token)."""
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
    try:
        conn.request("GET", "/api/version", headers={"Host": "127.0.0.1:%d" % port})
        return (conn.getresponse().getheader("Server") or "").startswith("LRCReview")
    except OSError:
        return False
    finally:
        conn.close()


def _port_arg(argv):
    for index, item in enumerate(argv):
        if item == "--port" and index + 1 < len(argv) and argv[index + 1].isdigit():
            return int(argv[index + 1])
        if item.startswith("--port=") and item[7:].isdigit():
            return int(item[7:])
    return None


def find_servers(scripts_dir=SCRIPTS_DIR, port_range=range(58400, 58421)):
    """Running review_server.py processes of this install: [{pid, ppid, port, argv, launcher_argv}].
    The port is the --port argument, else the one a server answers on that no other known server owns (netstat cannot name the owner here)."""
    text = _powershell("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'review_server\.py' } | "
                       "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    if not text.strip():
        return []
    rows = json.loads(text)
    rows = rows if isinstance(rows, list) else [rows]
    by_pid = {row["ProcessId"]: row for row in rows}
    target = str(pathlib.Path(scripts_dir).resolve() / "review_server.py").casefold()
    parents = {row["ParentProcessId"] for row in rows}
    servers = []
    for row in rows:
        pid = row["ProcessId"]
        if pid in parents:
            continue  # the venv launcher in front of the real interpreter
        argv = _argv_of(row["CommandLine"])
        script = next((item for item in argv if item.lower().endswith("review_server.py")), "")
        if not script or str(pathlib.Path(script).resolve()).casefold() != target:
            continue
        parent = by_pid.get(row["ParentProcessId"])
        servers.append({"pid": pid, "ppid": row["ParentProcessId"], "port": _port_arg(argv), "argv": argv,
                        "launcher_argv": _argv_of(parent["CommandLine"]) if parent else None})
    open_ports = [port for port in set(port_range) | {server["port"] for server in servers if server["port"]} if _review_port(port)]
    claimed = set()
    for server in servers:
        if server["port"]:
            claimed.add(server["port"])
            continue
        for port in sorted(open_ports):
            info = _get(port, "/api/admin/version")[1]
            if info.get("pid") == server["pid"]:
                server["port"] = port
                claimed.add(port)
                break
    loose = [server for server in servers if not server["port"]]
    free = [port for port in sorted(open_ports) if port not in claimed and _get(port, "/api/admin/version")[0] != 200]
    if len(loose) == 1 and len(free) == 1:
        loose[0]["port"] = free[0]
    return [server for server in servers if server["port"]]


def _get(port, path, method="GET", data=None, timeout=10):
    request = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), method=method, data=data,
                                     headers={"Host": "127.0.0.1:%d" % port, **({"Content-Type": "application/json"} if data is not None else {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


def _newest_mtime(directory):
    newest = 0.0
    for dirpath, _dirs, names in os.walk(directory):
        for name in names:
            try:
                newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                pass
    return newest


def _cpu_seconds(pid):
    out = _powershell("(Get-Process -Id %d -ErrorAction SilentlyContinue).TotalProcessorTime.TotalSeconds" % pid).strip()
    try:
        return float(out)
    except ValueError:
        return None


def legacy_idle(server, report_dir, quiet_s=20.0):
    """A server without the admin endpoint cannot be asked: no report file written for quiet_s, no child process, no CPU burn."""
    if time.time() - _newest_mtime(report_dir) < quiet_s:
        return False
    children = _powershell("(Get-CimInstance Win32_Process -Filter 'ParentProcessId=%d' | Measure-Object).Count" % server["pid"]).strip()
    if children not in ("0", ""):
        return False
    first = _cpu_seconds(server["pid"])
    time.sleep(3)
    second = _cpu_seconds(server["pid"])
    return first is not None and second is not None and second - first < 0.3


def legacy_restart(server, wait_s, report_dir, say):
    port = server["port"]
    end = time.monotonic() + wait_s
    while not legacy_idle(server, report_dir):
        if time.monotonic() > end:
            say("port %d: never idle within %ds; left running" % (port, wait_s))
            return False
        time.sleep(5)
    argv = server["argv"]
    launcher = server.get("launcher_argv")
    exe = launcher[0] if launcher and launcher[0].lower().endswith("python.exe") and any(a.lower().endswith("review_server.py") for a in launcher) else argv[0]
    rest = argv[1:]
    if "--port" not in rest and not any(a.startswith("--port=") for a in rest):
        rest = rest + ["--port", str(port)]
    for pid in {server["pid"], server["ppid"] if launcher else None} - {None}:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)
    for _ in range(100):
        if _get(port, "/api/version")[0] == 0:
            break
        time.sleep(0.1)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    try:
        process = subprocess.Popen([exe, *rest], cwd=str(SCRIPTS_DIR.parent), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, creationflags=HIDDEN_FLAGS, close_fds=True)
    except OSError as exc:
        say("port %d: relaunch failed: %s" % (port, exc))
        return False
    end = time.monotonic() + 240
    while time.monotonic() < end:
        status, _ = _get(port, "/api/admin/version")
        if status == 200:
            say("port %d: restarted (old pid %d -> new pid %d)" % (port, server["pid"], process.pid))
            return True
        if process.poll() is not None:
            break
        time.sleep(1)
    say("port %d: relaunch did not answer (exit=%s)" % (port, process.poll()))
    return False


def update_server(server, wait_s, report_dir, say):
    port = server["port"]
    status, info = _get(port, "/api/admin/version")
    if status == 200:
        old_pid = info.get("pid")
        status, answer = _get(port, "/api/admin/hot-update", "POST", b"{}")
        state = answer.get("state") if status == 200 else "error"
        if state == "current":
            say("port %d pid %s: already on the current code (%s)" % (port, old_pid, info.get("code_version")))
            return True
        if state not in ("scheduled", "updating"):
            say("port %d pid %s: hot-update request failed (%s)" % (port, old_pid, status))
            return False
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            time.sleep(1)
            status, now = _get(port, "/api/admin/version")
            if status == 200 and now.get("pid") != old_pid:
                say("port %d: hot-updated in place (pid %s -> %s, %s -> %s)" % (port, old_pid, now.get("pid"), info.get("code_version"), now.get("code_version")))
                return True
            if status == 200 and now.get("update_failed"):
                say("port %d pid %s: hot update failed, old server kept running" % (port, old_pid))
                return False
        say("port %d pid %s: not idle within %ds, update still pending (the server applies it when idle)" % (port, old_pid, wait_s))
        return False
    if status in (404, 403, 405):
        say("port %d pid %d: server predates hot update; restarting it from its command line" % (port, server["pid"]))
        return legacy_restart(server, wait_s, report_dir, say)
    say("port %d pid %d: no answer (%s)" % (port, server["pid"], status))
    return False


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Push the current code of this install into its running Review LRC server(s).")
    parser.add_argument("--wait", type=int, default=900, help="seconds to wait for a busy server to go idle (default 900)")
    parser.add_argument("--list", action="store_true", help="only list the running servers")
    parser.add_argument("--report-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    say = lambda text: print(text, flush=True)
    servers = find_servers()
    if not servers:
        say("not running, next launch uses the new version")
        return 0
    ok = True
    for server in servers:
        status, info = _get(server["port"], "/api/admin/version")
        say("running: pid %d port %d %s" % (server["pid"], server["port"], ("version %s@%s" % (info.get("head") or "nogit", info.get("code_version"))) if status == 200 else "version unknown (predates hot update)"))
        if args.list:
            continue
        report_dir = args.report_dir
        if report_dir is None:
            tokens = server["argv"]
            report_dir = pathlib.Path(tokens[tokens.index("--report-dir") + 1]) if "--report-dir" in tokens else SCRIPTS_DIR.parent / "outputs" / "reports"
        ok = update_server(server, args.wait, report_dir, say) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
