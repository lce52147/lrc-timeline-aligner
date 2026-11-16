"""Samples the visible top-level windows (ctypes EnumWindows) so a test run can prove it showed nothing on the screen.

``WindowWatcher`` is used by the hot-update tests and as a command line observer:
``python test_window_watch.py watch --seconds 30 [--root-pid N]`` prints every window that becomes visible, one JSON line each.
A window is attributable to a process tree when its owner process, or the parent of its owner (a conhost.exe made for a console
of the tree), is the root or a descendant of it.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import unittest
from ctypes import wintypes

IS_WINDOWS = os.name == "nt"
CONSOLE_CLASSES = ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "PseudoConsoleWindow")
SAMPLE_SECONDS = 0.05


if IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32
    _ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.InternalGetWindowText.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


def process_table():
    """{pid: (parent pid, exe name)} of every process now."""
    table = {}
    if not IS_WINDOWS:
        return table
    snap = _kernel32.CreateToolhelp32Snapshot(0x2, 0)
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = _kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            table[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            ok = _kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _kernel32.CloseHandle(snap)
    return table


def visible_windows():
    """[{hwnd, pid, cls, title, rect}] of every visible top-level window."""
    found = []
    if not IS_WINDOWS:
        return found

    def each(hwnd, _lparam):
        if _user32.IsWindowVisible(hwnd):
            pid = wintypes.DWORD()
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            cls = ctypes.create_unicode_buffer(256)
            _user32.GetClassNameW(hwnd, cls, 256)
            title = ctypes.create_unicode_buffer(256)
            _user32.InternalGetWindowText(hwnd, title, 256)  # reads the stored title: GetWindowText would send WM_GETTEXT to a thread that may not pump messages
            rect = wintypes.RECT()
            _user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append({"hwnd": int(hwnd or 0), "pid": pid.value, "cls": cls.value, "title": title.value,
                          "rect": [rect.left, rect.top, rect.right, rect.bottom]})
        return True

    callback = _ENUM_PROC(each)
    _user32.EnumWindows(callback, 0)
    return found


class WindowWatcher:
    """Records every top-level window that becomes visible while it runs (50 ms samples), with whether the tree of ``root_pid`` owns it."""

    def __init__(self, root_pid=None, seconds=SAMPLE_SECONDS):
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.seconds = seconds
        self.events = []
        self._parent = {}
        self._seen = set()
        self._recorded = {}
        self._stop = threading.Event()
        self._thread = None

    def _attributable(self, window, table):
        self._parent.update({pid: parent for pid, (parent, _name) in table.items()})
        owners = [window["pid"], self._parent.get(window["pid"], 0)]  # a console's conhost.exe is a child of the process it serves
        for pid in owners:
            seen = set()
            while pid and pid not in seen:
                if pid == self.root_pid:
                    return True
                seen.add(pid)
                pid = self._parent.get(pid, 0)
        return False

    def sample(self):
        table = process_table()
        for window in visible_windows():
            key = (window["hwnd"], window["pid"])
            if key in self._seen:
                known = self._recorded.get(key)
                if known:  # a window caught while it was still being built: keep its settled title and rectangle
                    known.update(title=window["title"], rect=window["rect"], cls=window["cls"])
                continue
            self._seen.add(key)
            if not self._baseline_done:
                continue
            window["attributable"] = self._attributable(window, table)
            window["name"] = table.get(window["pid"], (0, ""))[1]
            window["t"] = round(time.time(), 3)
            self.events.append(window)
            self._recorded[key] = window

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.seconds)

    def start(self):
        self._baseline_done = False
        self.sample()
        self._baseline_done = True
        self._thread = threading.Thread(target=self._run, name="window-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)
        self.sample()
        return self.events

    def ours(self):
        return [event for event in self.events if event["attributable"]]

    def consoles(self):
        return [event for event in self.events if event["cls"] in CONSOLE_CLASSES]


def create_offscreen_tool_window(title="window-watch-control", show=True):
    """A visible top-level window nobody sees: tool window (no taskbar button), no activation, far off the screen. Returns (hwnd, destroy)."""
    wndclass = "WindowWatchControl%d" % os.getpid()
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    _user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    _user32.DefWindowProcW.restype = ctypes.c_ssize_t
    proc = WNDPROC(lambda hwnd, msg, wparam, lparam: _user32.DefWindowProcW(hwnd, msg, wparam, lparam))

    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HANDLE), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                    ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

    cls = WNDCLASS()
    cls.lpfnWndProc = proc
    cls.hInstance = _kernel32.GetModuleHandleW(None)
    cls.lpszClassName = wndclass
    _user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    _user32.CreateWindowExW.restype = wintypes.HWND
    _user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, wintypes.HWND, wintypes.HANDLE, wintypes.HINSTANCE, wintypes.LPVOID]
    _user32.RegisterClassW(ctypes.byref(cls))
    ws_ex = 0x80 | 0x08000000  # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    hwnd = _user32.CreateWindowExW(ws_ex, wndclass, title, 0x80000000, -32000, -32000, 40, 40, None, None, cls.hInstance, None)  # WS_POPUP
    if show:
        _user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
    return hwnd, lambda: _user32.DestroyWindow(hwnd), proc


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    watch = sub.add_parser("watch")
    watch.add_argument("--seconds", type=float, default=30)
    watch.add_argument("--root-pid", type=int, default=None)
    watch.add_argument("--out", default="")
    watch.add_argument("--stop-file", default="")
    args = parser.parse_args(argv)
    watcher = WindowWatcher(args.root_pid if args.root_pid is not None else -1).start()
    end = time.monotonic() + args.seconds
    while time.monotonic() < end and not (args.stop_file and os.path.exists(args.stop_file)):
        time.sleep(0.2)
    events = watcher.stop()
    text = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


@unittest.skipUnless(IS_WINDOWS, "window sampling is Windows only")
class WindowWatcherTests(unittest.TestCase):
    """The watcher sees a window a user could not notice, and does not report one that is hidden."""

    def test_a_visible_off_screen_tool_window_is_recorded_and_attributed_to_this_process(self):
        watcher = WindowWatcher().start()
        hwnd, destroy, _keep = create_offscreen_tool_window()
        try:
            time.sleep(0.4)
        finally:
            destroy()
        events = watcher.stop()
        mine = [event for event in events if event["hwnd"] == int(hwnd)]
        self.assertEqual(len(mine), 1, events)
        self.assertTrue(mine[0]["attributable"])
        self.assertEqual(mine[0]["rect"][:2], [-32000, -32000])

    def test_a_hidden_window_is_not_recorded(self):
        watcher = WindowWatcher().start()
        hwnd, destroy, _keep = create_offscreen_tool_window(show=False)
        time.sleep(0.4)
        destroy()
        events = watcher.stop()
        self.assertEqual([event for event in events if event["hwnd"] == int(hwnd)], [])

    def test_a_window_of_another_process_tree_is_not_attributed_to_this_one(self):
        watcher = WindowWatcher(root_pid=4).start()  # PID 4 (System) owns none of the test's windows
        hwnd, destroy, _keep = create_offscreen_tool_window()
        try:
            time.sleep(0.4)
        finally:
            destroy()
        events = watcher.stop()
        self.assertTrue([event for event in events if event["hwnd"] == int(hwnd)])
        self.assertEqual(watcher.ours(), [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        sys.exit(main())
    unittest.main()
