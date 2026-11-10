#!/usr/bin/env python3
"""benchmark_pynq.py — benchmark.py, but measured on the PYNQ-Z2 RISC-V core.

For every checkpoint of the last train_mnist.py run (runs/manifest.json),
ANN and SNN:
  1. export it (export_model.py) unless export/<name>/ already holds enough
     test samples,
  2. run it on the board (deploy_pynq.sh, fast firmware by default),
  3. compute the same per-seed metrics as benchmark.py from the board
     results: test accuracy and loss, parameters, ops/sample (ANN: dense
     MACs, SNN: SOPs), spikes, plus what the board adds: latency (timer
     cycles of the RISC-V core), the synaptic operations the fast kernel
     really executes, and bit-exactness against the PC int8 reference.
The ANN-vs-SNN overview (mean +- std over seeds) is printed and written to
runs/summary_pynq_<fw>.md and runs/summary_pynq_<fw>.json.

Run from code/ann_vs_snn_mnist/ in Git Bash, venv active:
  python pynq/benchmark_pynq.py                    # full test set, ~12 min per seed pair
  python pynq/benchmark_pynq.py --n 1000           # first 1000 test images
  python pynq/benchmark_pynq.py --models ANN       # only the ANNs
  python pynq/benchmark_pynq.py --fw baseline --n 1000
  python pynq/benchmark_pynq.py --no-run           # only re-summarize board results
  python pynq/benchmark_pynq.py --fw muriscv       # ANN on muRISCV-NN (external
                                                   # reference, ANN only)
--fw mac (snn_mnist_mac.bin, spike_top_both.bit): the ANN's layers on the
ANN MAC coprocessor (custom-1), the SNN on the SNN coprocessor as in --fw hw.
--fw muriscv runs the ANNs from export/<name>_muriscv/ (muriscv_ref.py, run
if missing) on snn_mnist_muriscv.bin; it is a dense kernel, so its executed
ops are the dense MACs.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_mnist import MANIFEST, OUT_DIR  # noqa: E402

CLK_HZ = 100_000_000

METRICS = (  # (key, label, format) — same style as benchmark.py
    ("test_accuracy", "board test accuracy [%]", lambda v: f"{v*100:.2f}"),
    ("test_loss", "board test loss", lambda v: f"{v:.4f}"),
    ("pc_float_accuracy", "PC float accuracy, same images [%]", lambda v: f"{v*100:.2f}"),
    ("bit_exact", "board == PC int8 reference [%]", lambda v: f"{v*100:.2f}"),
    ("params", "parameters", lambda v: f"{v:.0f}"),
    ("ops_per_sample", "ops/sample, benchmark.py (MAC or SOP)", lambda v: f"{v:,.0f}"),
    ("ops_executed", "ops/sample executed by this firmware", lambda v: f"{v:,.0f}"),
    ("in_events", "nonzero inputs / input spikes per sample", lambda v: f"{v:.1f}"),
    ("hidden_events", "nonzero hidden / hidden spikes per sample", lambda v: f"{v:.1f}"),
    ("spikes_out_per_sample", "output spikes/sample", lambda v: f"{v:.1f}"),
    ("latency_ms_per_sample", "board latency [ms/sample]", lambda v: f"{v:.2f}"),
    ("latency_ms_min", "board latency min [ms]", lambda v: f"{v:.2f}"),
    ("latency_ms_max", "board latency max [ms]", lambda v: f"{v:.2f}"),
    ("cycles_per_sample", "board cycles/sample", lambda v: f"{v:,.0f}"),
    ("cycles_per_op", "cycles per executed op", lambda v: f"{v:.1f}"),
    ("model_bytes", "model size in BRAM [B]", lambda v: f"{v:.0f}"),
)


def board_metrics(name, fw):
    d = os.path.join(HERE, "export", name)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(d, f"board_results_{fw}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    is_snn = meta["model"] == "SNN"
    lay = meta["layers"]
    out = np.array([[int(c) for c in r["outputs"].split()] for r in rows], float)
    if not is_snn:
        out *= meta["out_scale"]            # int logits -> float logits
    y = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["board_pred"]) for r in rows])
    # cross-entropy as nn.CrossEntropyLoss in benchmark.py (SNN: on spike counts)
    z = out - out.max(1, keepdims=True)
    loss = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    col = lambda k: np.array([float(r[k]) for r in rows])
    if is_snn:
        ops = col("in_events") * lay[0]["n_out"] + col("hidden_events") * lay[1]["n_out"]
    else:
        ops = np.full(len(rows), sum(l["n_in"] * l["n_out"] for l in lay), float)
    if fw in ("fast", "hw", "mac"):  # nonzero inputs (ANN) / input changes (SNN) + hidden events
        executed = col("in_changes") * lay[0]["n_out"] + col("hidden_events") * lay[1]["n_out"]
    else:               # baseline: dense MACs (ANN) / every SOP (SNN)
        executed = ops
    cyc = col("cycles")
    m = {
        "model": meta["model"], "seed": meta["config"]["seed"], "firmware": fw,
        "n_samples": len(rows),
        "test_accuracy": float((pred == y).mean()),
        "test_loss": float(loss),
        "pc_float_accuracy": float((col("pc_float_pred") == y).mean()),
        "bit_exact": float(col("outputs_match").mean()),
        "params": sum(l["n_in"] * l["n_out"] + l["n_out"] for l in lay),
        "ops_per_sample": float(ops.mean()),
        "ops_executed": float(executed.mean()),
        "in_events": float(col("in_events").mean()),
        "hidden_events": float(col("hidden_events").mean()),
        "latency_ms_per_sample": float(cyc.mean() / CLK_HZ * 1e3),
        "latency_ms_min": float(cyc.min() / CLK_HZ * 1e3),
        "latency_ms_max": float(cyc.max() / CLK_HZ * 1e3),
        "cycles_per_sample": float(cyc.mean()),
        "cycles_per_op": float(cyc.sum() / executed.sum()),
        "model_bytes": meta["model_bytes"],
    }
    if is_snn:
        m["spikes_out_per_sample"] = float(col("out_events").mean())
    return m


def summarize(results, fw, n):
    names = sorted({r["model"] for r in results})
    seeds = sorted({r["seed"] for r in results})
    lines = [f"PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware {fw}, "
             f"{n} MNIST test images per seed, seeds {seeds}",
             "",
             "| metric (mean +- std over seeds) | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    stats = {}
    for name in names:
        rs = [r for r in results if r["model"] == name]
        stats[name] = {}
        for key, _, _ in METRICS:
            vals = [r[key] for r in rs if key in r]
            if vals:
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                stats[name][key] = (statistics.mean(vals), std)
    for key, label, fmt in METRICS:
        cells = [f"{fmt(stats[nm][key][0])} +- {fmt(stats[nm][key][1])}"
                 if key in stats[nm] else "n/a" for nm in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["",
              "ops/sample, benchmark.py: ANN dense MACs, SNN synaptic operations "
              "(one add per spike per fan-out synapse, over all timesteps).",
              "Executed ops: baseline firmware = the benchmark.py ops; fast firmware = "
              "ANN MACs of nonzero inputs/activations, SNN row updates of input "
              "changes (delta) + hidden spikes.",
              "Loss: cross-entropy on the int8 logits (ANN, rescaled) / "
              "output spike counts (SNN)."]
    return "\n".join(lines), stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ip", default=None, help="board IP (default: .board-ip of the flow)")
    p.add_argument("--n", type=int, default=10000, help="test images per seed")
    p.add_argument("--fw", choices=["fast", "baseline", "hw", "muriscv", "mac"], default="fast")
    p.add_argument("--models", nargs="+", default=["ANN", "SNN"], choices=["ANN", "SNN"])
    p.add_argument("--no-run", action="store_true",
                   help="do not touch the board, summarize existing results")
    p.add_argument("--resume", action="store_true",
                   help="continue interrupted board runs")
    a = p.parse_args()
    if a.fw == "muriscv":
        a.models = ["ANN"]                 # the library level is ANN only

    with open(MANIFEST) as f:
        ckpts = [r["checkpoint"] for r in json.load(f)["runs"]
                 if os.path.basename(r["checkpoint"])[:3].upper() in a.models]
    bash = shutil.which("bash")
    ip = a.ip
    if not ip and not a.no_run:
        with open(os.path.join(HERE, "..", "..", "pynqz2_riscv_flow", ".board-ip")) as f:
            ip = f.read().strip()
    results = []
    for ck in ckpts:
        name = os.path.splitext(os.path.basename(ck))[0]
        if a.fw == "muriscv":
            name += "_muriscv"
        meta_path = os.path.join(HERE, "export", name, "meta.json")
        if not a.no_run:
            n_have = 0
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                n_have = meta["n_samples"] if "input_bytes" in meta else 0
            if a.fw == "muriscv" and n_have < a.n:   # always the full test set
                subprocess.run([sys.executable, os.path.join(HERE, "muriscv_ref.py"), ck],
                               check=True)
            elif n_have < a.n:
                subprocess.run([sys.executable, os.path.join(HERE, "export_model.py"),
                                ck, "--n-samples", str(a.n)], check=True)
            cmd = [bash, os.path.join(HERE, "deploy_pynq.sh"), ip, name,
                   "--", "--fw", a.fw, "--n", str(a.n)]
            cmd += ["--resume"] if a.resume else []
            print(f"\n=== {name} on the board ({a.fw}) ===", flush=True)
            subprocess.run(cmd, check=True)
        r = board_metrics(name, a.fw)
        results.append(r)
        print(f"{name}: board acc {r['test_accuracy']*100:.2f}%  "
              f"{r['latency_ms_per_sample']:.2f} ms/sample  "
              f"bit-exact {r['bit_exact']*100:.1f}%", flush=True)

    n = min(r["n_samples"] for r in results)
    table, stats = summarize(results, a.fw, n)
    base = os.path.join(OUT_DIR, f"summary_pynq_{a.fw}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(base + ".json", "w") as f:
        json.dump({"firmware": a.fw, "n_samples": n, "runs": results,
                   "stats": {nm: {k: {"mean": m, "std": s} for k, (m, s) in st.items()}
                             for nm, st in stats.items()}},
                  f, indent=2)
    print("\n" + table)
    print(f"\nwritten to {base}.md and .json")


if __name__ == "__main__":
    main()
