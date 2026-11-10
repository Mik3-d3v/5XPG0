#!/usr/bin/env python3
"""demo_server.py — live ANN vs SNN demo: draw a digit, classify it on the FPGA.

Serves the web UI (index.html) on http://localhost:8000 and, for every
drawing the UI sends:
  1. preprocesses it like MNIST (crop, scale the digit into a 20x20 box,
     centre it by its centre of mass in 28x28, uint8 pixels),
  2. ANN: sends the 784 pixels to the PicoRV32 on the PYNQ-Z2,
     SNN: rate-encodes them (T = 25 Bernoulli spike bitmaps, as in
     train_mnist.py) and sends the spikes,
  3. runs both on the board (demo_server_pynq.py, real cycle counts) and the
     same int8 models on the PC (quant.py), and checks that the board output
     equals the PC int8 reference bit for bit,
  4. returns predictions, scores, latency, events and the SNN spike raster
     (from the bit-exact PC model) as JSON.

The SNN can run on the fast firmware (software) or on the hw firmware (SNN
coprocessor, custom instructions); the UI switches between them.

Board power cycle: a watchdog pings the board every few seconds; when the
board server does not answer (e.g. after a reboot) it restarts it over ssh
and reconnects, so the page recovers by itself ("reconnecting…" in the
header, plus a Reconnect button). No need to restart the demo.

Run (from code/ann_vs_snn_mnist/, venv active, after demo/start_demo.sh has
started the board side):
  python demo/demo_server.py                 # board from ../pynqz2_riscv_flow/.board-ip
  python demo/demo_server.py --offline       # no board: PC int8 only, no latency
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pynq"))
sys.path.insert(0, HERE)
from quant import int_forward_ann, int_forward_snn, pack_spikes  # noqa: E402
from board_link import Board, load_export as _load  # noqa: E402

POWER_W = 1.695           # Vivado on-chip power estimate, spike_top_snn.bit
EXPORT = os.path.join(ROOT, "pynq", "export")


def load_export(name):
    return _load(os.path.join(EXPORT, name))


# ------------------------------------------------------------------ input
def preprocess(png_bytes):
    """Canvas PNG (white digit on black) -> 28x28 uint8, MNIST style."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    a = np.asarray(img, np.float32)
    ys, xs = np.nonzero(a > 30)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    s = 20.0 / max(w, h)
    crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))),
                       Image.LANCZOS)
    c = np.asarray(crop, np.float32)
    tot = c.sum()
    cy = (c.sum(1) * np.arange(c.shape[0])).sum() / tot
    cx = (c.sum(0) * np.arange(c.shape[1])).sum() / tot
    out = np.zeros((28, 28), np.float32)
    oy, ox = int(round(14 - cy)), int(round(14 - cx))
    for y in range(c.shape[0]):              # paste with clipping
        for x in range(c.shape[1]):
            yy, xx = y + oy, x + ox
            if 0 <= yy < 28 and 0 <= xx < 28:
                out[yy, xx] = c[y, x]
    return np.clip(out, 0, 255).astype(np.uint8)


def rate_encode(pixels, T, seed):
    """As train_mnist.rate_encode: spike if U(0,1) < pixel / 255."""
    g = torch.Generator().manual_seed(seed)
    p = torch.from_numpy(pixels.reshape(1, -1).astype(np.float32) / 255.0)
    return (torch.rand(T, 1, 784, generator=g) < p).to(torch.uint8).numpy()


# ------------------------------------------------------------------ demo
class Demo:
    def __init__(self, args):
        self.ann_name, self.snn_name = args.ann, args.snn
        self.ann_meta, self.ann = load_export(args.ann)
        self.snn_meta, self.snn = load_export(args.snn)
        self.T = self.snn_meta["timesteps"]
        # connection + watchdog (auto-reconnect after a board power cycle)
        self.board = Board(None if args.offline else args.board, args.board_port)
        self.counter = 0

    @property
    def status(self):
        return self.board.status

    def _board(self, model, fw, T, data):
        return self.board.infer(model, fw, T, data)

    def infer(self, png, snn_fw):
        self.counter += 1
        px = preprocess(png)
        flat = px.reshape(1, -1)

        # ---------------- ANN
        logits_pc, nz_hidden = int_forward_ann(self.ann, flat, return_hidden=True)
        logits_pc = logits_pc[0]
        b_ann = self._board(self.ann_name, "fast", 0, flat.tobytes())
        z = logits_pc * self.ann_meta["out_scale"]
        prob = np.exp(z - z.max()); prob /= prob.sum()
        nz_px = int((flat != 0).sum())
        ann = {
            "pred": int(np.argmax(logits_pc)), "scores": prob.round(4).tolist(),
            "logits": logits_pc.tolist(),
            "nonzero_pixels": nz_px, "nonzero_hidden": int(nz_hidden[0]),
            "macs_dense": 784 * 128 + 128 * 10,
            "macs_executed": nz_px * 128 + int(nz_hidden[0]) * 10,
        }
        if b_ann:
            ann.update(board=True, ms=b_ann["ms"], cycles=b_ann["cycles"],
                       board_pred=b_ann["pred"], switch_ms=b_ann["switch_ms"],
                       bit_exact=b_ann["outputs"] == ann["logits"])

        # ---------------- SNN
        spikes = rate_encode(px, self.T, seed=self.counter)        # [T, 1, 784]
        rec = []
        counts_pc = int_forward_snn(self.snn, spikes, record=rec)[0]
        data = pack_spikes(spikes)[0].tobytes()
        b_snn = self._board(self.snn_name, snn_fw, self.T, data)
        in_steps = spikes[:, 0].sum(1)
        changes = int(np.abs(np.diff(np.concatenate(
            [np.zeros((1, 784), np.uint8), spikes[:, 0]]).astype(np.int8), axis=0)).sum())
        hid_steps = [int(r[0][0].sum()) for r in rec]
        out_raster = [r[-1][0].astype(int).tolist() for r in rec]  # [T][10]
        sops = int(in_steps.sum()) * 128 + sum(hid_steps) * 10
        snn = {
            "pred": int(np.argmax(counts_pc)), "counts": counts_pc.tolist(),
            "scores": (counts_pc / max(1, counts_pc.sum())).round(4).tolist(),
            "input_spikes": int(in_steps.sum()), "input_changes": changes,
            "hidden_spikes": sum(hid_steps), "output_spikes": int(counts_pc.sum()),
            "sops": sops, "T": self.T, "fw": snn_fw,
            "raster": out_raster, "hidden_per_step": hid_steps,
            "input_per_step": in_steps.astype(int).tolist(),
        }
        if b_snn:
            snn.update(board=True, ms=b_snn["ms"], cycles=b_snn["cycles"],
                       board_pred=b_snn["pred"], switch_ms=b_snn["switch_ms"],
                       bit_exact=b_snn["outputs"] == snn["counts"])
        for m in (ann, snn):
            if m.get("board"):
                m["energy_uj"] = round(m["ms"] * 1e-3 * POWER_W * 1e6, 1)

        frames = np.packbits(spikes[:, 0], axis=1, bitorder="little")  # [T, 98]
        return {"id": self.counter, "pixels": px.reshape(-1).tolist(),
                "frames": base64.b64encode(frames.tobytes()).decode(),
                "ann": ann, "snn": snn, "status": self.status}


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                   # quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json({**DEMO.board.status_json(),
                        "ann": DEMO.ann_name, "snn": DEMO.snn_name,
                        "T": DEMO.T,
                        "pc_accuracy": {"ann": DEMO.ann_meta["pc_test_accuracy_int"],
                                        "snn": DEMO.snn_meta["pc_test_accuracy_int"]}})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/reconnect":            # the page's button
            return self._json(DEMO.board.reconnect(force=True))
        if self.path != "/api/infer":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            png = base64.b64decode(req["png"].split(",", 1)[1])
            t0 = time.perf_counter()
            rep = DEMO.infer(png, req.get("snn_fw", "fast"))
            rep["roundtrip_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            self._json(rep)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global DEMO
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ip_file = os.path.join(ROOT, "..", "pynqz2_riscv_flow", ".board-ip")
    default_ip = open(ip_file).read().strip() if os.path.exists(ip_file) else "192.168.2.99"
    p.add_argument("--board", default=default_ip)
    p.add_argument("--board-port", type=int, default=9999)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ann", default="ann_mnist_seed0")
    p.add_argument("--snn", default="snn_mnist_seed0")
    p.add_argument("--offline", action="store_true",
                   help="no board: PC int8 models only (no latency)")
    a = p.parse_args()
    DEMO = Demo(a)
    st = DEMO.status
    print("board:", "connected (" + st["bit"] + ")" if st.get("board")
          else "NOT connected - " + st.get("reason", ""))
    print(f"demo on http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
