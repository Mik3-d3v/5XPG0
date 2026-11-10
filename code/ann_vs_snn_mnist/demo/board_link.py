"""board_link.py — PC side of the live demos: talk to demo_server_pynq.py.

Shared by the MNIST demo (demo_server.py) and the ECG demo
(../../ann_vs_snn_ecg/demo/ecg_demo_server.py):

  load_export(dir)   model.bin + meta.json of an export -> (meta, int layers)
  BoardLink          one TCP connection, JSON lines, reconnect on errors
  Board              connection + watchdog: pings the board every few
                     seconds and, when its server does not answer (e.g. after
                     a power cycle: no server, no bitstream), restarts it over
                     ssh, so a demo recovers by itself.
"""

import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time

import numpy as np

REMOTE = "/home/xilinx/mnist_snn"
# the board-side server, started over ssh after a board (re)boot; -f -n so
# the ssh session does not stay open. Started only if none is running, so
# two demos (MNIST, ECG) recovering at the same time do not kill each
# other's fresh server.
# (the pgrep pattern is anchored to the python process: the remote shell's
# own command line contains the script name and must not match)
BOARD_START = ("pgrep -f '^/usr/local/share/pynq-venv/bin/python3 -u demo_server_pynq' "
               ">/dev/null || "
               "{{ cd {r} && nohup sudo bash -c 'source /etc/profile.d/xrt_setup.sh && "
               "/usr/local/share/pynq-venv/bin/python3 -u demo_server_pynq.py' "
               "> demo.log 2>&1 < /dev/null & }}")
WATCH_S = 4               # how often the watchdog checks the board
RESTART_EVERY_S = 8       # minimum time between two restart attempts
CLK_HZ = 100_000_000


def load_export(d):
    """model.bin + meta.json of an export folder -> (meta, integer layers)."""
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(d, "model.bin"), "rb") as f:
        blob = f.read()
    _magic, n_layers, _n_in, _T = struct.unpack_from("<4I", blob, 0)
    layers = []
    for k in range(n_layers):
        ni, no, p0, p1, w_off, b_off, stride = struct.unpack_from(
            "<7i", blob, 16 + 28 * k)
        w = np.frombuffer(blob, np.int8, ni * stride, w_off).reshape(ni, stride)
        b = np.frombuffer(blob, "<i4", no, b_off)
        layers.append({"w": w[:, :no].copy(), "b": b.astype(np.int32),
                       "p0": p0, "p1": p1})
    return meta, layers


class BoardLink:
    """One TCP connection to demo_server_pynq.py, reconnecting on errors."""

    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = self.f = None
        self.lock = threading.Lock()

    def _connect(self):
        # short connect timeout (a rebooting board should fail fast), longer
        # timeout for the replies (a slow model / firmware can take seconds)
        self.sock = socket.create_connection(self.addr, timeout=3)
        self.sock.settimeout(30)
        self.f = self.sock.makefile("rwb")

    def call(self, req):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.f is None:
                        self._connect()
                    self.f.write((json.dumps(req) + "\n").encode())
                    self.f.flush()
                    rep = json.loads(self.f.readline())
                    if "error" in rep:
                        raise RuntimeError(rep["error"])
                    return rep
                except (OSError, ValueError):
                    self.sock = self.f = None
                    if attempt:
                        raise


class Board:
    """The board as the demos see it: status, inference, auto-reconnect.
    host=None: offline (software) mode, every call returns None."""

    def __init__(self, host, port=9999):
        self.host = host
        self.link = BoardLink(host, port) if host else None
        self.reconnecting = False
        self.last_restart = 0.0
        self.status = self.ping()
        if self.link is not None:
            threading.Thread(target=self._watchdog, daemon=True).start()

    @property
    def online(self):
        return bool(self.status.get("board"))

    def ping(self):
        if self.link is None:
            return {"board": False, "reason": "offline mode"}
        try:
            rep = self.link.call({"op": "ping"})
            return {"board": True, "bit": rep["bit"], "firmware": rep["firmware"],
                    "models": rep["models"]}
        except Exception as e:
            return {"board": False, "reason": str(e)}

    def infer(self, model, fw, T, data):
        """-> board reply dict (pred, cycles, ms, outputs, ...) or None."""
        if self.link is None or not self.online:
            return None
        try:
            import base64
            return self.link.call({"op": "infer", "model": model, "fw": fw,
                                   "T": T, "input": base64.b64encode(data).decode()})
        except Exception as e:
            self.status = {"board": False, "reason": str(e)}
            return None

    # -------------------------------------------------------- reconnection
    def _start_board_server(self, kill=False):
        """Start demo_server_pynq.py on the board over ssh (kill=True: stop a
        running one first, for the manual Reconnect)."""
        ssh = shutil.which("ssh") or "C:/Program Files/Git/usr/bin/ssh.exe"
        opts = ["-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
        self.last_restart = time.time()
        # no captured output: the backgrounded ssh -f would keep the pipe
        # open and block this thread
        quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                 "stderr": subprocess.DEVNULL, "timeout": 20}
        try:
            if kill:
                subprocess.run([ssh, *opts, f"xilinx@{self.host}",
                                "sudo pkill -f '[d]emo_server_pynq.py' || true"], **quiet)
            r = subprocess.run([ssh, "-f", "-n", *opts, f"xilinx@{self.host}",
                                BOARD_START.format(r=REMOTE)], **quiet)
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False                     # board still booting: retry later

    def reconnect(self, force=False):
        """Ping; if the board server does not answer, restart it and wait
        until it serves (loading the bitstream takes a few seconds)."""
        if self.link is None:
            return self.status
        if force:
            self.last_restart = 0.0
        self.status = self.ping()
        if self.online:
            self.reconnecting = False
            return self.status
        self.reconnecting = True
        # wait for the server only if ssh could start it (the board answers
        # ping a few seconds before its ssh service is up)
        if (time.time() - self.last_restart > RESTART_EVERY_S
                and self._start_board_server(kill=force)):
            for _ in range(15):              # up to ~30 s (bitstream load)
                time.sleep(2)
                self.status = self.ping()
                if self.online:
                    break
        self.reconnecting = not self.online
        return self.status

    def _watchdog(self):
        while True:
            time.sleep(WATCH_S)
            try:
                self.reconnect()
            except Exception:
                pass

    def status_json(self):
        return {**self.status, "reconnecting": self.reconnecting}
