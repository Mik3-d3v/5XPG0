#!/usr/bin/env python3
"""demo_server_pynq.py — live-inference server for the presentation demo.

Runs ON THE PYNQ-Z2 as root (start_demo.sh starts it). Loads the bitstream
once (spike_top_snn.bit: it runs the baseline, fast and hw firmware) and
then serves inference requests over TCP, one JSON object per line:

  {"op": "ping"}
      -> {"ok": true, "bit": ..., "models": [...], "firmware": [...]}
  {"op": "infer", "model": "ann_mnist_seed0", "fw": "fast", "T": 25,
   "input": "<base64 input bytes, as in inputs.bin>"}
      -> {"pred", "cycles", "ms", "aux", "outputs", "err", "switch_ms"}

The BRAM holds one model at a time (an MNIST model is 103 KB of the 176 KB
model area), so switching model or firmware rewrites the firmware image and
model.bin with the CPU in reset and waits until the firmware answers the
mailbox again. Repeated requests for the same model/firmware skip that.
Stdlib only; needs spike_pynq.py of the bring-up flow next to it.

Usage: demo_server_pynq.py [--port 9999] [--bit spike_top_snn.bit]
"""

import argparse
import base64
import json
import os
import socketserver
import threading
import time

from spike_pynq import MAILBOX_BASE, Bram, Regs, load_bit

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_BASE = 0x10800
INPUT_BASE = 0x11000
INPUT_MAX_BYTES = 0x2000
MODEL_BASE = 0x13000
MODEL_MAX_BYTES = 0x3F000 - MODEL_BASE
MB_CMD_ECHO = 1
MB_CMD_RUN = 16
CLK_HZ = 100_000_000
FIRMWARE = {"fast": "snn_mnist_fast.bin", "baseline": "snn_mnist.bin",
            "hw": "snn_mnist_hw.bin"}
SIGNED_OUTPUT = ("ANN", "GRU", "RSNN")     # int32 logits, not spike counts


class Board:
    def __init__(self, bit):
        load_bit(bit)
        self.bit = bit
        self.regs, self.bram = Regs(), Bram()
        self.lock = threading.Lock()
        self.loaded = None                  # (fw, model)
        self.fw_cache, self.model_cache, self.meta = {}, {}, {}

    def _read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def _mb(self, opcode, args=(), timeout=30.0):
        a = (list(args) + [0, 0, 0])[:3]
        b = self.bram
        seq = b.rd32(MAILBOX_BASE)
        b.wr32(MAILBOX_BASE + 8, opcode)
        for i, v in enumerate(a):
            b.wr32(MAILBOX_BASE + 12 + 4 * i, v)
        b.wr32(MAILBOX_BASE, seq + 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if b.rd32(MAILBOX_BASE + 4) == seq + 1:
                return [b.rd32(MAILBOX_BASE + 24 + 4 * i) for i in range(4)]
            time.sleep(0.0002)
        raise TimeoutError("mailbox timeout")

    def _switch(self, fw, model):
        if self.loaded == (fw, model):
            return 0.0
        t0 = time.perf_counter()
        if fw not in self.fw_cache:
            self.fw_cache[fw] = self._read(os.path.join(HERE, FIRMWARE[fw]))
        if model not in self.model_cache:
            d = os.path.join(HERE, model)
            self.model_cache[model] = self._read(os.path.join(d, "model.bin"))
            with open(os.path.join(d, "meta.json")) as f:
                self.meta[model] = json.load(f)
        blob = self.model_cache[model]
        if len(blob) > MODEL_MAX_BYTES:
            raise ValueError("model too large")
        self.regs.ctrl_reset(True)
        time.sleep(0.002)
        self.bram.write_image(self.fw_cache[fw])
        self.bram.m[MODEL_BASE:MODEL_BASE + len(blob)] = blob
        self.regs.ctrl_reset(False)
        # the firmware answers the mailbox once model_check / prepare are done
        deadline = time.time() + 5.0
        while True:
            try:
                if self._mb(MB_CMD_ECHO, [0xC0FFEE, 1, 2], timeout=0.2)[0] == 0xC0FFEE:
                    break
            except TimeoutError:
                pass
            if time.time() > deadline:
                raise TimeoutError("firmware did not start")
        self.bram.console_read()            # drain the boot banner
        self.loaded = (fw, model)
        return (time.perf_counter() - t0) * 1e3

    def infer(self, model, fw, T, data):
        with self.lock:
            switch_ms = self._switch(fw, model)
            meta = self.meta[model]
            if len(data) > INPUT_MAX_BYTES:
                raise ValueError("input too large")
            self.bram.m[INPUT_BASE:INPUT_BASE + len(data)] = data
            pred, cycles, aux, err = self._mb(MB_CMD_RUN, [T])
            outs = [self.bram.rd32(RESULT_BASE + 4 * o) for o in range(meta["n_out"])]
            if meta["model"] in SIGNED_OUTPUT:
                outs = [v - (1 << 32) if v >= 1 << 31 else v for v in outs]
            return {"pred": pred, "cycles": cycles, "ms": cycles / CLK_HZ * 1e3,
                    "aux": aux, "err": err, "outputs": outs,
                    "switch_ms": round(switch_ms, 1), "fw": fw, "model": model}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                req = json.loads(line)
                if req.get("op") == "ping":
                    models = sorted(d for d in os.listdir(HERE)
                                    if os.path.exists(os.path.join(HERE, d, "model.bin")))
                    rep = {"ok": True, "bit": BOARD.bit, "models": models,
                           "firmware": [k for k, v in FIRMWARE.items()
                                        if os.path.exists(os.path.join(HERE, v))]}
                elif req.get("op") == "infer":
                    rep = BOARD.infer(req["model"], req.get("fw", "fast"),
                                      int(req.get("T", 0)),
                                      base64.b64decode(req["input"]))
                else:
                    rep = {"error": "unknown op"}
            except Exception as e:                       # report, keep serving
                rep = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(rep) + "\n").encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global BOARD
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--bit", default="spike_top_snn.bit")
    a = p.parse_args()
    os.chdir(HERE)
    BOARD = Board(a.bit)
    print(f"bitstream {a.bit} loaded, serving on port {a.port}", flush=True)
    with Server(("0.0.0.0", a.port), Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
