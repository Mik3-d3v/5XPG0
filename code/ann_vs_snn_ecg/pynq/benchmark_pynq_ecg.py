#!/usr/bin/env python3
"""benchmark_pynq_ecg.py — benchmark_ecg.py, but measured on the PYNQ-Z2 RISC-V core.

The ECG counterpart of ../../ann_vs_snn_mnist/pynq/benchmark_pynq.py, using
the same firmware image, deploy script and board-side runner. For every
checkpoint of the last train_ecg.py --final run (runs/manifest.json):
  1. export it (pynq/export_ecg.py) unless pynq/export/<name>/ exists,
  2. run it on the board (../../ann_vs_snn_mnist/pynq/deploy_pynq.sh),
  3. compute benchmark_ecg.py's metrics from the BOARD predictions: macro-F1,
     accuracy, per-class Se / +P, weighted CE loss, plus latency (timer
     cycles of the RISC-V core), executed operations and bit-exactness
     against the PC int8 reference.
The ANN-vs-SNN overview (mean +- std over seeds) with confusion matrices,
per-record table and paired per-beat comparison is printed and written to
runs/summary_pynq_<fw>.md / .json.

Run from code/ann_vs_snn_ecg/ in Git Bash, venv active:
  python pynq/benchmark_pynq_ecg.py                  # all DS2 beats, fast firmware
  python pynq/benchmark_pynq_ecg.py --fw baseline
  python pynq/benchmark_pynq_ecg.py --n 2000 --models ANN
  python pynq/benchmark_pynq_ecg.py --no-run         # only re-summarize
  python pynq/benchmark_pynq_ecg.py --fw muriscv     # ANN on muRISCV-NN (external
      reference, ANN only: pynq/export/<name>_muriscv/, export_ecg_muriscv.py)
  python pynq/benchmark_pynq_ecg.py --fw mac         # ANN / GRU on the ANN MAC
      coprocessor, SNN / RSNN on the SNN coprocessor (the direct-encoded
      SNN's first layer on the MAC unit), spike_top_both.bit
  python pynq/benchmark_pynq_ecg.py --stream         # streaming GRU vs RSNN
      (runs/manifest_stream.json, pynq/export_stream/, export_stream.py,
       runs/summary_pynq_stream_<fw>.md)
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
from train_ecg import BBB_RECORDS, MANIFEST, OUT_DIR  # noqa: E402

MNIST_PYNQ = os.path.join(HERE, "..", "..", "ann_vs_snn_mnist", "pynq")
EXPORT = os.path.join(HERE, "export")
CLK_HZ = 100_000_000

METRICS = (  # (key, label, format)
    ("test_macro_f1", "board macro-F1 (N/S/V)", lambda v: f"{v:.3f}"),
    ("test_accuracy", "board accuracy [%]", lambda v: f"{v*100:.2f}"),
    ("test_se_N", "Se N [%]", lambda v: f"{v*100:.2f}"),
    ("test_se_S", "Se S [%]", lambda v: f"{v*100:.2f}"),
    ("test_se_V", "Se V [%]", lambda v: f"{v*100:.2f}"),
    ("test_ppv_N", "+P N [%]", lambda v: f"{v*100:.2f}"),
    ("test_ppv_S", "+P S [%]", lambda v: f"{v*100:.2f}"),
    ("test_ppv_V", "+P V [%]", lambda v: f"{v*100:.2f}"),
    ("test_loss", "board loss (weighted CE)", lambda v: f"{v:.4f}"),
    ("pc_float_macro_f1", "PC float macro-F1, same beats", lambda v: f"{v:.3f}"),
    ("pc_float_accuracy", "PC float accuracy, same beats [%]", lambda v: f"{v*100:.2f}"),
    ("bit_exact", "board == PC int8 reference [%]", lambda v: f"{v*100:.2f}"),
    ("params", "parameters", lambda v: f"{v:.0f}"),
    ("ops_per_sample", "ops/beat, benchmark_ecg.py (MAC + SOP)", lambda v: f"{v:,.0f}"),
    ("ops_executed", "ops/beat executed by this firmware", lambda v: f"{v:,.0f}"),
    ("in_events", "nonzero int8 inputs per beat (of 129)", lambda v: f"{v:.1f}"),
    ("hidden_events", "nonzero hidden / hidden spikes per beat", lambda v: f"{v:.1f}"),
    ("spikes_out_per_sample", "output spikes/beat", lambda v: f"{v:.1f}"),
    ("latency_ms_per_sample", "board latency [ms/beat]", lambda v: f"{v:.3f}"),
    ("latency_ms_min", "board latency min [ms]", lambda v: f"{v:.3f}"),
    ("latency_ms_max", "board latency max [ms]", lambda v: f"{v:.3f}"),
    ("cycles_per_sample", "board cycles/beat", lambda v: f"{v:,.0f}"),
    ("cycles_per_op", "cycles per executed op", lambda v: f"{v:.1f}"),
    ("model_bytes", "model size in BRAM [B]", lambda v: f"{v:.0f}"),
)


def cm_metrics(cm, classes):
    """accuracy, macro-F1, per-class Se / +P (as train_ecg.cm_metrics)."""
    tp = np.diag(cm).astype(float)
    se = tp / np.maximum(cm.sum(1), 1)
    ppv = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * se * ppv / np.maximum(se + ppv, 1e-9)
    m = {"accuracy": tp.sum() / cm.sum(), "macro_f1": f1.mean()}
    for i, c in enumerate(classes):
        m[f"se_{c}"], m[f"ppv_{c}"] = se[i], ppv[i]
    return {k: float(v) for k, v in m.items()}


def confusion(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def load_rows(name, fw):
    d = os.path.join(EXPORT, name)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(d, f"board_results_{fw}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    rec = np.fromfile(os.path.join(d, "records.bin"), "<i4")[:len(rows)]
    return meta, rows, rec


def board_metrics(name, fw):
    meta, rows, rec = load_rows(name, fw)
    cfg, classes = meta["config"], meta["classes"]
    k = len(classes)
    is_snn = meta["model"] == "SNN"
    lay = meta["layers"]
    col = lambda c: np.array([float(r[c]) for r in rows])
    out = np.array([[int(c) for c in r["outputs"].split()] for r in rows], float)
    if not is_snn:
        out *= meta["out_scale"]            # int logits -> float logits
    y = col("label").astype(int)
    pred = col("board_pred").astype(int)
    pf = col("pc_float_pred").astype(int)
    # weighted cross-entropy exactly as in benchmark_ecg.py (SNN: on spike
    # counts): the class-weighted mean per batch of 1024 beats, averaged
    # over the batches weighted by their size
    wcls = np.array(cfg["class_weights"])
    z = out - out.max(1, keepdims=True)
    nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))
    loss = sum((wcls[y[i:i + 1024]] * nll[i:i + 1024]).sum()
               / wcls[y[i:i + 1024]].sum() * len(y[i:i + 1024])
               for i in range(0, len(y), 1024)) / len(y)
    cm = confusion(y, pred, k)
    n_in, h, c_out = lay[0]["n_in"], lay[0]["n_out"], lay[1]["n_out"]
    ins, hid = col("in_events"), col("hidden_events")
    if meta["model"] == "GRU":      # dense MACs, every sample
        ops = np.full(len(rows), float(meta["macs_per_sample"]))
        executed = ops
    elif meta["model"] == "RSNN":   # SOPs: input events and hidden spikes,
        H = lay[0]["n_out"]         # each to all H hidden neurons
        ops = (ins + hid) * H
        executed = ops
    elif is_snn:      # direct encoding: dense first layer once + SOPs
        ops = n_in * h + hid * c_out
    else:
        ops = np.full(len(rows), n_in * h + h * c_out, float)
    if meta["model"] in ("GRU", "RSNN"):
        pass
    elif fw in ("fast", "hw", "mac"):  # nonzero inputs x fan-out + hidden events x fan-out
        executed = ins * h + hid * c_out
    else:               # baseline: dense first layer, dense / every SOP after
        executed = ops
    cyc = col("cycles")
    m = {"model": meta["model"], "seed": cfg["seed"], "firmware": fw,
         "n_samples": len(rows),
         **{f"test_{a}": v for a, v in cm_metrics(cm, classes).items()},
         "test_loss": float(loss),
         "pc_float_macro_f1": cm_metrics(confusion(y, pf, k), classes)["macro_f1"],
         "pc_float_accuracy": float((pf == y).mean()),
         "bit_exact": float(col("outputs_match").mean()),
         "params": meta.get("params",
                            sum(l["n_in"] * l["n_out"] + l["n_out"] for l in lay)),
         "ops_per_sample": float(ops.mean()),
         "ops_executed": float(executed.mean()),
         "in_events": float(ins.mean()),
         "hidden_events": float(hid.mean()),
         "latency_ms_per_sample": float(cyc.mean() / CLK_HZ * 1e3),
         "latency_ms_min": float(cyc.min() / CLK_HZ * 1e3),
         "latency_ms_max": float(cyc.max() / CLK_HZ * 1e3),
         "cycles_per_sample": float(cyc.mean()),
         "cycles_per_op": float(cyc.sum() / executed.sum()),
         "model_bytes": meta["model_bytes"],
         "timesteps": meta["timesteps"],
         "confusion": cm.tolist()}
    if is_snn:
        m["spikes_out_per_sample"] = float(col("out_events").mean())
    return m, {"y": y, "pred": pred, "rec": rec, "cycles": cyc}


def summarize(results, beats, fw, n, classes):
    names = sorted({r["model"] for r in results})
    seeds = sorted({r["seed"] for r in results})
    T = {r["model"]: r["timesteps"] for r in results}
    lines = [f"PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware {fw}, "
             f"{n} MIT-BIH DS2 beats per seed, seeds {seeds}, SNN T={T.get('SNN')}",
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
              "ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) "
              "first-layer MACs once + SOPs of the hidden spikes over T steps.",
              "Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of "
              "the nonzero int8 inputs + hidden events x fan-out.",
              "Loss: class-weighted cross-entropy on the int8 logits (ANN, "
              "rescaled) / output spike counts (SNN), averaged per batch of "
              "1024 beats as in benchmark_ecg.py."]

    for name in names:
        cm = sum(np.array(r["confusion"]) for r in results if r["model"] == name)
        lines += ["", f"{name} board confusion matrix (rows = true, cols = "
                  f"predicted, summed over seeds), classes {classes}:"]
        lines += ["    " + " ".join(f"{v:7d}" for v in row) for row in cm.tolist()]

    # per record (mean over seeds)
    k = len(classes)
    any_b = next(iter(beats.values()))
    head = ["record"] + [f"n_{c}" for c in classes]
    for nm in names:
        head += [f"{nm} acc"] + [f"{nm} Se_{c}" for c in classes[1:]] + [f"{nm} ms"]
    lines += ["", "Per-record board results (mean over seeds):", "",
              "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in np.unique(any_b["rec"]):
        sel = any_b["rec"] == r
        cnt = np.bincount(any_b["y"][sel], minlength=k)
        row = [f"{r}{' (BBB)' if r in BBB_RECORDS else ''}"] + [str(int(v)) for v in cnt]
        for nm in names:
            bs = [b for (m, _), b in beats.items() if m == nm]
            row.append(f"{np.mean([(b['pred'][sel] == b['y'][sel]).mean() for b in bs]):.3f}")
            for c in range(1, k):
                yc = sel & (any_b["y"] == c)
                row.append("-" if not yc.any() else
                           f"{np.mean([(b['pred'][yc] == c).mean() for b in bs]):.2f}")
            row.append(f"{np.mean([b['cycles'][sel].mean() for b in bs]) / CLK_HZ * 1e3:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # paired per-beat view (same beat, same seed)
    an, sn = (names + names)[:2] if len(names) != 2 else names
    an, sn = ("ANN", "SNN") if set(names) == {"ANN", "SNN"} else (an, sn)
    pairs = [s for s in seeds if (an, s) in beats and (sn, s) in beats]
    if pairs:
        lines += ["", "Paired per-beat comparison (same DS2 beat, same seed):", "",
                  "| | " + " | ".join(f"seed {s}" for s in pairs) + " |",
                  "|---|" + "---|" * len(pairs)]
        rows = {}
        for s in pairs:
            A, S = beats[(an, s)], beats[(sn, s)]
            ratio = S["cycles"] / A["cycles"]
            a_ok, s_ok = A["pred"] == A["y"], S["pred"] == S["y"]
            for key, v in [
                    (f"{sn} / {an} latency, mean of totals", f"{S['cycles'].mean() / A['cycles'].mean():.2f}"),
                    (f"{sn} / {an} latency, median per beat", f"{np.median(ratio):.2f}"),
                    ("min / max per beat", f"{ratio.min():.2f} / {ratio.max():.2f}"),
                    (f"beats where the {sn} is faster", f"{int((ratio < 1).sum()):,}"),
                    (f"corr({an}, {sn} latency)", f"{np.corrcoef(A['cycles'], S['cycles'])[0, 1]:.3f}"),
                    ("both correct", f"{int((a_ok & s_ok).sum()):,}"),
                    (f"only {an} correct", f"{int((a_ok & ~s_ok).sum()):,}"),
                    (f"only {sn} correct", f"{int((~a_ok & s_ok).sum()):,}"),
                    ("both wrong", f"{int((~a_ok & ~s_ok).sum()):,}"),
                    ("same prediction", f"{int((A['pred'] == S['pred']).sum()):,}")]:
                rows.setdefault(key, []).append(v)
        lines += [f"| {k_} | " + " | ".join(v) + " |" for k_, v in rows.items()]
    return "\n".join(lines), stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ip", default=None, help="board IP (default: .board-ip of the flow)")
    p.add_argument("--n", type=int, default=0, help="DS2 beats per model (0 = all)")
    p.add_argument("--fw", choices=["fast", "baseline", "hw", "muriscv", "mac"], default="fast")
    p.add_argument("--models", nargs="+", default=None,
                   choices=["ANN", "SNN", "GRU", "RSNN"])
    p.add_argument("--stream", action="store_true",
                   help="streaming models (GRU, RSNN): runs/manifest_stream.json")
    p.add_argument("--no-run", action="store_true",
                   help="do not touch the board, summarize existing results")
    p.add_argument("--resume", action="store_true",
                   help="continue interrupted board runs")
    a = p.parse_args()
    global EXPORT
    if a.stream:
        manifest = os.path.join(OUT_DIR, "manifest_stream.json")
        EXPORT = os.path.join(HERE, "export_stream")
        exporter, prefix = "export_stream.py", "summary_pynq_stream"
        a.models = a.models or ["GRU", "RSNN"]
    else:
        manifest, exporter, prefix = MANIFEST, "export_ecg.py", "summary_pynq"
        a.models = a.models or ["ANN", "SNN"]
    if a.fw == "muriscv":                  # the library level is ANN only
        assert not a.stream, "--fw muriscv: feed-forward ANN only"
        a.models, exporter = ["ANN"], "export_ecg_muriscv.py"

    with open(manifest) as f:
        ckpts = [r["checkpoint"] for r in json.load(f)["runs"]
                 if os.path.basename(r["checkpoint"]).split("_")[0].upper() in a.models]
    names = [os.path.splitext(os.path.basename(c))[0] + ("_muriscv" if a.fw == "muriscv" else "")
             for c in ckpts]
    if not a.no_run and not all(os.path.exists(os.path.join(EXPORT, nm, "meta.json"))
                                for nm in names):
        # one call: the input scale is calibrated over all models together
        subprocess.run([sys.executable, os.path.join(HERE, exporter)], check=True)
    ip = a.ip
    if not ip and not a.no_run:
        with open(os.path.join(HERE, "..", "..", "pynqz2_riscv_flow", ".board-ip")) as f:
            ip = f.read().strip()
    bash = shutil.which("bash")
    results, beats = [], {}
    for nm in names:
        if not a.no_run:
            cmd = [bash, os.path.join(MNIST_PYNQ, "deploy_pynq.sh"), ip,
                   os.path.join(EXPORT, nm).replace("\\", "/"), "--", "--fw", a.fw]
            cmd += ["--n", str(a.n)] if a.n else []
            cmd += ["--resume"] if a.resume else []
            print(f"\n=== {nm} on the board ({a.fw}) ===", flush=True)
            subprocess.run(cmd, check=True)
        r, b = board_metrics(nm, a.fw)
        results.append(r)
        beats[(r["model"], r["seed"])] = b
        print(f"{nm}: board macro-F1 {r['test_macro_f1']:.3f}  acc "
              f"{r['test_accuracy']*100:.2f}%  {r['latency_ms_per_sample']:.3f} ms/beat  "
              f"bit-exact {r['bit_exact']*100:.2f}%", flush=True)

    n = min(r["n_samples"] for r in results)
    with open(os.path.join(EXPORT, names[0], "meta.json")) as f:
        classes = json.load(f)["classes"]
    table, stats = summarize(results, beats, a.fw, n, classes)
    base = os.path.join(OUT_DIR, f"{prefix}_{a.fw}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(base + ".json", "w") as f:
        json.dump({"firmware": a.fw, "n_samples": n, "runs": results,
                   "stats": {nm: {k: {"mean": m, "std": s} for k, (m, s) in st.items()}
                             for nm, st in stats.items()}}, f, indent=2)
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + table)
    print(f"\nwritten to {base}.md and .json")


if __name__ == "__main__":
    main()
