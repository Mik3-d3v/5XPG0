#!/usr/bin/env python3
"""ecg_demo_server.py — live ECG demo: a patient's ECG plays back, two networks
classify every beat (N / S / V) on the PYNQ-Z2 RISC-V core, side by side.

Serves the web UI (index.html) on http://localhost:8001.

Signal: a held-out MIT-BIH patient (DS2), lead MLII, band-pass filtered and
at 180 Hz exactly as in prepare_mitbih.py. A player advances a simulated
clock at the chosen speed (0.5x .. 25x). A beat is classified once the next
R peak has arrived (the RR features need it), i.e. about one beat late, as a
monitor would. The beat windows and features are those of the DS2 test set
(checked on load), so the demo shows the models exactly as they were
evaluated.

Model pairs (seed 0 of the final models):
  beat     ANN (129-64-3)  vs  SNN (direct encoding, T = 10)
  stream   GRU (H = 64)    vs  recurrent SNN (level-crossing events)
The SNN of a pair can run on the fast firmware (software) or on the
coprocessor (custom instructions).

Modes:
  FPGA      every beat runs on the board (real cycle counts) and on the PC int8
            model, which must agree bit for bit; the activations shown come
            from the bit-exact PC model. The board connection reconnects by
            itself after a power cycle (board_link.Board).
  software  PC int8 models only (--offline or the switch in the UI); the
            latency shown is the PC's, not the board's.

The board is shared: the two models take turns (at most ~250 ms each), so a
slow model (the GRU needs ~1.1 s per beat) delays only its own results.

Run (from code/ann_vs_snn_ecg/, venv active; demo/start_demo.sh does all):
  python demo/ecg_demo_server.py              # FPGA
  python demo/ecg_demo_server.py --offline    # software only
"""

import argparse
import base64
import collections
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from scipy import signal as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # code/ann_vs_snn_ecg
MNIST = os.path.join(ROOT, "..", "ann_vs_snn_mnist")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MNIST, "pynq"))
sys.path.insert(0, os.path.join(MNIST, "demo"))
import prepare_mitbih as prep  # noqa: E402
from train_ecg import delta_encode, load_components, make_features  # noqa: E402
from quant import (int_forward_ann, int_forward_gru, int_forward_rsnn,  # noqa: E402
                   int_forward_snn, pad4, quantize_inputs)
from board_link import Board, load_export  # noqa: E402

FS = prep.FS                                      # 180 Hz
CLASSES = ["N", "S", "V"]
POWER_W = 1.695                                   # Vivado estimate, spike_top_snn.bit
SPEEDS = [0.5, 1, 2, 5, 10, 25]
PAIRS = {
    "beat": [("ann_ecg_seed0", "export"), ("snn_ecg_seed0", "export")],
    "stream": [("gru_ecg_seed0", "export_stream"), ("rsnn_ecg_seed0", "export_stream")],
}
TURN_S = 0.25                                     # board time per model turn
NOTES = {  # short descriptions of the DS2 records (MIT-BIH directory)
    100: "mostly normal", 105: "noisy, some V", 200: "frequent V (bigeminy)",
    202: "atrial flutter / fibrillation", 210: "atrial fibrillation, V",
    212: "right bundle branch block", 213: "V and fusion beats",
    214: "left bundle branch block, V", 219: "atrial fibrillation, V",
    221: "atrial fibrillation, V", 222: "S-rich, atrial flutter",
    228: "frequent V", 232: "most S beats of DS2", 233: "frequent V",
    234: "normal, some S",
}


# ------------------------------------------------------------------ data
def record_beats(rec, data_dir):
    """Filtered 180 Hz signal [mV] and the kept beats (as prepare_mitbih:
    R index, label) of one record."""
    sig, r_idx, sym = prep.load_record(rec, data_dir)
    sig = sps.decimate(prep.bandpass(sig), prep.DECIM, zero_phase=True)
    r_idx = r_idx // prep.DECIM
    keep = [i for i, s in enumerate(sym) if s in set("NLRejAaJSVEFf/Q")]
    r_idx, sym = r_idx[keep], [sym[i] for i in keep]
    pre, post = int(prep.PRE_S * FS), int(prep.POST_S * FS)
    beats = []
    for i in range(len(r_idx)):
        c = r_idx[i]
        if c - pre < 0 or c + post > len(sig):
            continue
        if 1 <= i < len(r_idx) - 1 and sym[i] in prep.AAMI:
            # ready when the window is complete and the next R peak is seen
            beats.append((int(c), prep.AAMI[sym[i]], int(max(c + post, r_idx[i + 1]))))
    return sig.astype(np.float32), beats, pre, post


class Dataset:
    """DS2: per record the signal + beats, and the model features per beat."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        comp, y, rec, _ = load_components("ds2", data_dir)
        self.x = make_features(comp, True, "diff").numpy()         # [N, 129]
        self.beat_raw = comp[:, :126].numpy()                      # z-scored beat
        self.y, self.rec = y.numpy(), rec
        self.rows = {int(r): np.where(rec == r)[0] for r in np.unique(rec)}
        self.cache = {}

    def records(self):
        out = []
        for r in prep.DS2:
            cnt = np.bincount(self.y[self.rows[r]], minlength=3).tolist()
            out.append({"rec": r, "n": cnt, "note": NOTES.get(r, "")})
        return out

    def load(self, r):
        if r not in self.cache:
            sig, beats, pre, post = record_beats(r, os.path.join(self.data_dir, "mitdb"))
            rows = self.rows[r]
            assert len(rows) == len(beats), f"record {r}: beat count mismatch"
            # same windows as the test set: z-score of the window == npz beat
            for k in (0, len(beats) // 2, len(beats) - 1):
                c = beats[k][0]
                w = sig[c - pre:c + post].astype(np.float64)
                w = (w - w.mean()) / (w.std() + 1e-6)
                assert np.allclose(w, self.beat_raw[rows[k]], atol=1e-3), \
                    f"record {r}: beat windows differ from the test set"
                assert beats[k][1] == self.y[rows[k]]
            self.cache[r] = (sig, beats, rows)
        return self.cache[r]


# ------------------------------------------------------------------ models
class Model:
    """One exported int8 model: board input bytes, PC reference, activations."""

    def __init__(self, name, folder, s_x):
        self.name = name
        self.meta, self.layers = load_export(os.path.join(ROOT, "pynq", folder, name))
        self.kind = self.meta["model"]
        self.s_x = s_x
        self.T = self.meta["timesteps"] if self.kind != "ANN" else 0
        cfg = self.meta["config"]
        self.theta, self.levels = cfg.get("delta_theta"), cfg.get("delta_levels")
        self.label = {"ANN": "ANN", "SNN": "SNN", "GRU": "GRU",
                      "RSNN": "Recurrent SNN"}[self.kind]
        self.desc = {
            "ANN": "129-64-3 · int8 · ReLU · one pass",
            "SNN": f"129-64-3 · LIF · direct encoding · T = {self.T}",
            "GRU": "GRU H = 64 · reads 126 samples · 3 gates per step",
            "RSNN": f"LIF H = 64 · level-crossing events (θ = {self.theta})",
        }[self.kind]
        self.params = self.meta.get("params") or sum(
            l["n_in"] * l["n_out"] + l["n_out"] for l in self.meta["layers"])

    def run(self, x):
        """float features [129] -> (board bytes, pc outputs, probs, activations)."""
        xq = quantize_inputs(x[None], self.s_x)                    # [1, 129]
        act = {}
        if self.kind == "ANN":
            l0 = self.layers[0]
            acc = xq.astype(np.int64) @ l0["w"].astype(np.int64) + l0["b"]
            h = np.minimum((np.maximum(acc, 0) * l0["p0"] + (1 << 23)) >> 24, 255)
            out = int_forward_ann(self.layers, xq)[0]
            data = pad4(xq).tobytes()
            act = {"hidden": h[0].astype(int).tolist(), "events": int((xq != 0).sum()),
                   "work": f"{int((xq != 0).sum())} nonzero inputs × 64 MACs + "
                           f"{int((h != 0).sum())} active hidden × 3"}
            probs = softmax(out * self.meta["out_scale"])
        elif self.kind == "SNN":
            rec = []
            out = int_forward_snn(self.layers, None, direct_x=(self.T, xq),
                                  record=rec)[0]
            data = pad4(xq).tobytes()
            hid = np.stack([r[0][0] for r in rec])                  # [T, 64]
            act = {"raster": [[int(t), int(n)] for t, n in zip(*np.nonzero(hid))],
                   "out_raster": [r[-1][0].astype(int).tolist() for r in rec],
                   "steps": self.T, "events": int(hid.sum()),
                   "work": f"first layer: 1 MAC pass ({int((xq != 0).sum())} inputs), "
                           f"then {self.T} steps · {int(hid.sum())} hidden spikes"}
            probs = out / out.sum() if out.sum() else np.full(3, 1 / 3)
        elif self.kind == "GRU":
            L = x.shape[0] - 3
            rec = []
            out = int_forward_gru(self.layers, xq[:, :L], xq[:, L:], record=rec)[0][0]
            data = np.concatenate([pad4(xq[:, :L])[0].view(np.uint8),
                                   np.pad(xq[0, L:].view(np.uint8), (0, 1))]).tobytes()
            hs = np.stack([r[0] for r in rec]).astype(np.int8)      # [T, 64]
            act = {"heat": base64.b64encode(hs.tobytes()).decode(), "steps": L,
                   "events": L,
                   "work": f"{L} steps × 3 gates × 65 × 64 = "
                           f"{L * 3 * 65 * 64:,} MACs, every sample"}
            probs = softmax(out * self.meta["out_scale"])
        else:                                                       # RSNN
            L = x.shape[0] - 3
            ev = delta_encode(torch.from_numpy(x[None, :L]), self.theta,
                              self.levels).numpy()                  # [T, 1, 2K]
            rr = xq[:, L:]
            rec = []
            out = int_forward_rsnn(self.layers, ev, rr, record=rec)[0][0]
            bits = (ev[:, 0].astype(np.uint32)
                    << np.arange(ev.shape[2], dtype=np.uint32)).sum(1).astype("<u4")
            data = bits.tobytes() + np.pad(rr[0].view(np.uint8), (0, 1)).tobytes()
            hid = np.stack([r[0] for r in rec])                     # [T, 64]
            evs = ev[:, 0]
            act = {"raster": [[int(t), int(n)] for t, n in zip(*np.nonzero(hid))],
                   "in_events": [[int(t), int(c)] for t, c in zip(*np.nonzero(evs))],
                   "levels": self.levels, "steps": L,
                   "events": int(evs.sum()),
                   "work": f"{int(evs.sum())} input events + {int(hid.sum())} hidden "
                           f"spikes × 64 synapses; silent steps cost only the leak"}
            probs = softmax(out * self.meta["out_scale"])
        return data, np.asarray(out).astype(int).tolist(), probs, act


def softmax(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


# ------------------------------------------------------------------ metrics
class Metrics:
    def __init__(self):
        self.cm = np.zeros((3, 3), int)
        self.ms, self.pc_ms, self.cycles, self.rr_ms = [], [], 0, 0.0
        self.bit_exact = self.board_n = 0

    def add(self, true, pred, ms, cycles, pc_ms, rr_ms, exact):
        self.cm[true, pred] += 1
        self.pc_ms.append(pc_ms)
        self.rr_ms += rr_ms
        if ms is not None:
            self.ms.append(ms)
            self.cycles += cycles
            self.board_n += 1
            self.bit_exact += int(exact)

    def json(self):
        cm = self.cm
        tp = np.diag(cm).astype(float)
        se = tp / np.maximum(cm.sum(1), 1)
        pp = tp / np.maximum(cm.sum(0), 1)
        f1 = 2 * se * pp / np.maximum(se + pp, 1e-9)
        present = cm.sum(1) > 0                  # macro over classes seen so far
        ms = np.array(self.ms) if self.ms else None
        return {
            "n": int(cm.sum()), "cm": cm.tolist(),
            "acc": float(tp.sum() / max(cm.sum(), 1)),
            "macro_f1": float(f1[present].mean()) if present.any() else None,
            "se": se.round(4).tolist(), "ppv": pp.round(4).tolist(),
            "f1": f1.round(4).tolist(), "present": present.tolist(),
            "board_n": self.board_n, "bit_exact": self.bit_exact,
            "ms_mean": float(ms.mean()) if ms is not None else None,
            "ms_p95": float(np.percentile(ms, 95)) if ms is not None else None,
            "ms_last": float(ms[-1]) if ms is not None else None,
            "pc_ms_mean": float(np.mean(self.pc_ms)) if self.pc_ms else None,
            "load": float(ms.sum() / self.rr_ms) if ms is not None and self.rr_ms else None,
            "energy_mj": float(ms.sum() * 1e-3 * POWER_W * 1e3) if ms is not None else 0.0,
        }


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, args):
        self.data = Dataset(args.data_dir)
        meta0, _ = load_export(os.path.join(ROOT, "pynq", "export", "ann_ecg_seed0"))
        self.s_x = meta0["input_scale"]
        self.models = {name: Model(name, folder, self.s_x)
                       for pair in PAIRS.values() for name, folder in pair}
        self.board = Board(None if args.offline else args.board, args.board_port)
        self.offline = args.offline
        self.lock = threading.RLock()
        self.pair, self.fw, self.use_board = "beat", "fast", not args.offline
        self.speed, self.playing = 1.0, True
        self.events = collections.deque(maxlen=600)     # (seq, result)
        self.seq = 0
        self.latest = {}                                # slot -> (seq, activations)
        self.set_record(232, reset=True)
        threading.Thread(target=self._player, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # -------------------------------------------------------------- control
    def slots(self):
        return [name for name, _ in PAIRS[self.pair]]

    def set_record(self, r, reset=False):
        with self.lock:
            self.sig, self.beats, self.rows = self.data.load(r)
            self.rec, self.pos, self.next_beat = r, 0.0, 0
            self.gen = getattr(self, "gen", 0) + 1
            self.queue = {s: collections.deque() for s in self.slots()}
            if reset or not hasattr(self, "metrics"):
                self.reset_metrics()

    def reset_metrics(self):
        with self.lock:
            self.metrics = {s: Metrics() for s in self.slots()}
            self.agree = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0, "same": 0}

    def seek(self, cls):
        """Jump to 2 s before the next beat of class cls (S / V)."""
        with self.lock:
            target = CLASSES.index(cls)
            for k in range(self.next_beat, len(self.beats)):
                if self.beats[k][1] == target:
                    self.pos = max(0.0, self.beats[k][0] - 2 * FS)
                    self.next_beat = k
                    while self.next_beat > 0 and self.beats[self.next_beat - 1][2] > self.pos:
                        self.next_beat -= 1
                    self.gen += 1
                    for q in self.queue.values():
                        q.clear()
                    return True
        return False

    def control(self, req):
        a = req.get("action")
        with self.lock:
            if a == "play":
                self.playing = True
            elif a == "pause":
                self.playing = False
            elif a == "speed":
                self.speed = float(req["value"])
            elif a == "record":
                self.set_record(int(req["value"]))
            elif a == "restart":
                self.set_record(self.rec)
            elif a == "seek":
                self.seek(req["value"])
            elif a == "pair" and req["value"] in PAIRS and req["value"] != self.pair:
                self.pair = req["value"]
                self.latest = {}
                self.set_record(self.rec, reset=True)
            elif a == "fw":
                self.fw = req["value"]
            elif a == "source":
                self.use_board = req["value"] == "board" and not self.offline
            elif a == "reset":
                self.reset_metrics()
            elif a == "reconnect":
                threading.Thread(target=self.board.reconnect, args=(True,),
                                 daemon=True).start()

    # -------------------------------------------------------------- player
    def _player(self):
        last = time.perf_counter()
        while True:
            time.sleep(0.03)
            now = time.perf_counter()
            dt, last = now - last, now
            with self.lock:
                if not self.playing:
                    continue
                self.pos = min(self.pos + self.speed * FS * dt, len(self.sig) - 1)
                while (self.next_beat < len(self.beats)
                       and self.beats[self.next_beat][2] <= self.pos):
                    for q in self.queue.values():
                        q.append((self.gen, self.next_beat))
                    self.next_beat += 1
                if self.pos >= len(self.sig) - 1:
                    self.playing = False                 # end of the record

    # -------------------------------------------------------------- worker
    def _worker(self):
        turn = 0
        while True:
            with self.lock:
                slots = self.slots()
            slot = slots[turn % len(slots)]
            turn += 1
            t_end = time.perf_counter() + TURN_S
            did = False
            while time.perf_counter() < t_end:
                with self.lock:
                    q = self.queue.get(slot)
                    if not q:
                        break
                    gen, k = q.popleft()
                    if gen != self.gen:
                        continue
                    beats, rows, sig_rec = self.beats, self.rows, self.rec
                    fw = self.fw if self.models[slot].kind in ("SNN", "RSNN") else "fast"
                    use_board = self.use_board
                self._classify(slot, k, beats, rows, sig_rec, fw, use_board, gen)
                did = True
            if not did:
                time.sleep(0.01)

    def _classify(self, slot, k, beats, rows, rec, fw, use_board, gen):
        m = self.models[slot]
        r_idx, true, _ = beats[k]
        x = self.data.x[rows[k]]
        t0 = time.perf_counter()
        data, out_pc, probs, act = m.run(x)
        pc_ms = (time.perf_counter() - t0) * 1e3
        pred = int(np.argmax(out_pc))
        res = {"slot": slot, "k": k, "rec": rec, "r": r_idx, "true": int(true),
               "pred": pred, "probs": np.round(probs, 4).tolist(), "out": out_pc,
               "pc_ms": round(pc_ms, 2), "fw": fw}
        b = self.board.infer(m.name, fw, m.T, data) if use_board else None
        if b is not None:
            res.update(board=True, ms=b["ms"], cycles=b["cycles"],
                       exact=b["outputs"] == out_pc, board_pred=b["pred"])
            pred = res["pred"] = int(b["pred"])
        rr_ms = ((beats[k + 1][0] - beats[k][0]) if k + 1 < len(beats) else FS) / FS * 1e3
        with self.lock:
            if gen != self.gen and rec != self.rec:
                return
            self.metrics[slot].add(int(true), pred, res.get("ms"), res.get("cycles", 0),
                                   pc_ms, rr_ms, res.get("exact", False))
            self.seq += 1
            res["seq"] = self.seq
            self.events.append(res)
            self.latest[slot] = (self.seq, {"k": k, "true": int(true), "pred": pred,
                                            "beat": np.round(x[:126], 3).tolist(),
                                            "act": act})
            self._agree(k, rec)

    def _agree(self, k, rec):
        """Pair-level agreement once both models classified beat k."""
        a, b = self.slots()
        ra = rb = None
        for e in reversed(self.events):
            if e["k"] == k and e["rec"] == rec:
                if e["slot"] == a and ra is None:
                    ra = e
                if e["slot"] == b and rb is None:
                    rb = e
            if ra and rb:
                break
        if not (ra and rb) or ra.get("counted") or rb.get("counted"):
            return
        ra["counted"] = rb["counted"] = True
        oa, ob = ra["pred"] == ra["true"], rb["pred"] == rb["true"]
        self.agree["both" if oa and ob else "only_a" if oa else
                   "only_b" if ob else "neither"] += 1
        self.agree["same"] += int(ra["pred"] == rb["pred"])

    # -------------------------------------------------------------- poll
    def poll(self, since, sig_from, known):
        with self.lock:
            pos = int(self.pos)
            lo = max(sig_from, pos - 12 * FS, 0)
            if sig_from < 0 or sig_from > pos:
                lo = max(0, pos - 12 * FS)
            sig = np.round(self.sig[lo:pos + 1], 3).tolist()
            ev = [{k: v for k, v in e.items() if k != "counted"}
                  for e in self.events if e["seq"] > since]
            latest = {s: v[1] for s, v in self.latest.items() if v[0] > known.get(s, 0)}
            latest_seq = {s: v[0] for s, v in self.latest.items()}
            # every beat of the visible window (not just of the new samples)
            beats_vis = [(b[0], b[1]) for b in self.beats
                         if pos - 12 * FS <= b[0] <= pos]
            pending = {s: len(q) for s, q in self.queue.items()}
            lag = {}
            for s, q in self.queue.items():
                lag[s] = round((pos - self.beats[q[0][1]][0]) / FS, 2) if q else 0.0
            return {
                "seq": self.seq, "pos": pos, "rec": self.rec, "len": len(self.sig),
                "playing": self.playing, "speed": self.speed, "pair": self.pair,
                "fw": self.fw, "source": "board" if self.use_board else "pc",
                "sig_from": lo, "sig": sig, "beats": beats_vis, "events": ev,
                "latest": latest, "latest_seq": latest_seq,
                "metrics": {s: m.json() for s, m in self.metrics.items()},
                "agree": self.agree, "pending": pending, "lag": lag,
                "status": self.board.status_json(), "offline": self.offline,
            }

    def info(self):
        return {"records": self.data.records(), "speeds": SPEEDS, "fs": FS,
                "pairs": {p: [{"name": n, "kind": self.models[n].kind,
                               "label": self.models[n].label,
                               "desc": self.models[n].desc,
                               "params": self.models[n].params,
                               "ds2_f1": self.models[n].meta["pc_metrics_int"]["macro_f1"]}
                              for n, _ in v] for p, v in PAIRS.items()},
                "classes": CLASSES, "offline": self.offline}


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
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
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/api/info":
            self._json(ENGINE.info())
        elif u.path == "/api/poll":
            known = json.loads(q.get("known", "{}"))
            self._json(ENGINE.poll(int(q.get("since", 0)), int(q.get("sig", -1)), known))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/control":
            return self._send(404, b"not found", "text/plain")
        try:
            ENGINE.control(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._json({"ok": True})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global ENGINE
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ip_file = os.path.join(ROOT, "..", "pynqz2_riscv_flow", ".board-ip")
    default_ip = open(ip_file).read().strip() if os.path.exists(ip_file) else "192.168.2.99"
    p.add_argument("--board", default=default_ip)
    p.add_argument("--board-port", type=int, default=9999)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--offline", action="store_true",
                   help="software mode: PC int8 models only")
    a = p.parse_args()
    ENGINE = Engine(a)
    st = ENGINE.board.status
    print("board:", "connected (" + st["bit"] + ")" if st.get("board")
          else "NOT connected - " + st.get("reason", ""), flush=True)
    print(f"demo on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
