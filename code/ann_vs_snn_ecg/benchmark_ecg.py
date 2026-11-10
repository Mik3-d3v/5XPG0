#!/usr/bin/env python3
"""benchmark_ecg.py — compare the trained ANN vs SNN on MIT-BIH DS2.

Mirrors benchmark.py. Loads every checkpoint of the last train_ecg.py run
(runs/manifest.json) and reports, per model and seed, on the full DS2 test set
(22 patients never seen in training or validation):
  * test macro-F1, accuracy, loss, and per-class Se (recall) / +P (precision),
  * parameter count,
  * cost proxies per beat:
      MACs  — dense multiply-accumulates (ANN: both layers; SNN with direct
              encoding: the first layer once, its input current is constant)
      SOPs  — synaptic operations (one add per spike per fan-out synapse,
              over all timesteps); event-driven hardware only "pays" for these
      ops   — MACs + SOPs, the single number comparable to benchmark.py
              (a MAC costs more than an add - discuss this, don't hide it)
  * for the SNN, input, hidden and output spikes per beat,
  * mean inference latency on the PC CPU (a proxy only — the real target is
    the RISC-V core on the PYNQ-Z2!).

The overview is mean +- std over seeds. It is printed, written to
runs/summary.md and runs/summary.json (incl. the confusion matrix summed over
seeds), and logged to the TensorBoard run of the training (HPARAMS tab: one
row per model/seed, TEXT tab: "summary").

Run:  python train_ecg.py && python benchmark_ecg.py
"""

import json
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from train_ecg import (MANIFEST, OUT_DIR, build_model, cm_metrics, confusion,
                       encode, load_split)

BATCH = 1024
N_LATENCY = 200       # beats per latency batch


@torch.no_grad()
def evaluate(model, loader, cfg):
    """Test loss, metrics, confusion matrix and cost proxies over DS2."""
    is_snn = cfg["model"] == "SNN"
    classes = cfg["classes"]
    n_cls = len(classes)
    # same loss definition as in train_ecg.py (weighted, batch mean)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg["class_weights"]))
    model.eval()
    total_loss = 0.0
    n = 0
    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
    preds = []
    in_spk = hid_spk = out_spk = 0.0
    for x, y in loader:
        if is_snn:
            s_in = encode(x, cfg["timesteps"], cfg["encoding"])
            s_out, s_hid = model(s_in, return_hidden=True)
            out = s_out.sum(dim=0)
            if cfg["encoding"] == "rate":
                in_spk += s_in.sum().item()
            hid_spk += s_hid.sum().item()
            out_spk += s_out.sum().item()
        else:
            out = model(x)
        total_loss += loss_fn(out, y).item() * y.shape[0]
        n += y.shape[0]
        cm += confusion(y, out.argmax(1), n_cls)
        preds.append(out.argmax(1))

    m = {f"test_{k}": v for k, v in cm_metrics(cm, classes).items()}
    m["test_loss"] = total_loss / n
    if is_snn:
        h = model.fc1.out_features
        n_in = model.fc1.in_features
        macs = n_in * h if cfg["encoding"] == "direct" else 0
        sops = (in_spk * h + hid_spk * n_cls) / n
        m["spikes_in_per_sample"] = in_spk / n
        m["spikes_hidden_per_sample"] = hid_spk / n
        m["spikes_out_per_sample"] = out_spk / n
    else:
        macs = model.macs_per_sample() if hasattr(model, "macs_per_sample") else 0
        sops = 0.0
    m["macs_per_sample"] = macs
    m["sops_per_sample"] = sops
    m["ops_per_sample"] = macs + sops
    return m, cm, torch.cat(preds)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency(model, x, cfg, repeat=10):
    """Mean PC-CPU latency in ms per beat (batched)."""
    model.eval()
    if cfg["model"] == "SNN":
        run = lambda: model(encode(x, cfg["timesteps"], cfg["encoding"])).sum(0)
    else:
        run = lambda: model(x)
    for _ in range(3):      # warm-up
        run()
    t0 = time.perf_counter()
    for _ in range(repeat):
        run()
    return (time.perf_counter() - t0) / repeat * 1000.0 / x.shape[0]


# ----------------------------------------------------------------------
def metric_table(classes):
    """(key, label, format) rows; per-class rows depend on the class list."""
    pct = lambda v: f"{v*100:.2f}"
    rows = [
        ("test_macro_f1", "test macro-F1", lambda v: f"{v:.3f}"),
        ("test_accuracy", "test accuracy [%]", pct),
    ]
    rows += [(f"test_se_{c}", f"Se {c} [%]", pct) for c in classes]
    rows += [(f"test_ppv_{c}", f"+P {c} [%]", pct) for c in classes]
    rows += [
        ("test_loss", "test loss (weighted CE)", lambda v: f"{v:.4f}"),
        ("val_macro_f1", "val macro-F1", lambda v: f"{v:.3f}"),
        ("val_accuracy", "val accuracy [%]", pct),
        ("params", "parameters", lambda v: f"{v:.0f}"),
        ("macs_per_sample", "MACs/beat", lambda v: f"{v:,.0f}"),
        ("sops_per_sample", "SOPs/beat", lambda v: f"{v:,.0f}"),
        ("ops_per_sample", "ops/beat (MAC + SOP)", lambda v: f"{v:,.0f}"),
        ("spikes_in_per_sample", "input spikes/beat", lambda v: f"{v:.1f}"),
        ("spikes_hidden_per_sample", "hidden spikes/beat", lambda v: f"{v:.1f}"),
        ("spikes_out_per_sample", "output spikes/beat", lambda v: f"{v:.1f}"),
        ("latency_ms_per_sample", "PC latency [ms/beat]", lambda v: f"{v:.4f}"),
        ("train_time_s", "train time [s]", lambda v: f"{v:.1f}"),
    ]
    return rows


def summarize(results, train_args, classes):
    """mean +- std over seeds for every metric, as a markdown table."""
    names = sorted({r["model"] for r in results})
    metrics = metric_table(classes)
    stats = {}
    for name in names:
        rs = [r for r in results if r["model"] == name]
        stats[name] = {}
        for key, _, _ in metrics:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                stats[name][key] = (statistics.mean(vals), std)

    a = train_args
    lines = [
        f"epochs={a['epochs']}, seeds={a['seeds']}, batch={a['batch']}, "
        f"lr={a['lr']}, hidden={a['hidden']}, timesteps={a['timesteps']}, "
        f"beta={a['beta']}, encoding={a['encoding']}, use_rr={a['use_rr']}, "
        f"morph={a.get('morph', 'raw')}, augment={a.get('augment', False)}, "
        f"wd={a.get('weight_decay')}",
        f"test set: MIT-BIH DS2 (inter-patient), classes={classes}",
        "",
        "| metric (mean +- std over seeds) | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key, label, fmt in metrics:
        cells = []
        for name in names:
            if key in stats[name]:
                mean, std = stats[name][key]
                cells.append(f"{fmt(mean)} +- {fmt(std)}")
            else:
                cells.append("n/a")
        if any(c != "n/a" for c in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "MACs are dense multiply-accumulates; SOPs are synaptic "
              "operations (one add per spike per fan-out synapse, over all "
              "timesteps). A MAC is more expensive than an add."]
    return "\n".join(lines), stats


def per_record_table(preds, y, rec, classes):
    """Per DS2 record: beats per class and accuracy / Se of each model
    (mean over seeds). Se is '-' where the record has no beat of a class."""
    names = sorted({n for n, _ in preds})
    n_cls = len(classes)
    head = ["record"] + [f"n_{c}" for c in classes]
    for n in names:
        head += [f"{n} acc"] + [f"{n} Se_{c}" for c in classes[1:]]
    lines = ["Per-record DS2 results (mean over seeds):", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in np.unique(rec):
        k = torch.from_numpy(rec == r)
        cnt = torch.bincount(y[k], minlength=n_cls)
        row = [str(int(r))] + [str(int(v)) for v in cnt]
        for n in names:
            ps = [p[k] for (m, _), p in preds.items() if m == n]
            row.append(f"{np.mean([(p == y[k]).float().mean().item() for p in ps]):.3f}")
            for c in range(1, n_cls):
                yc = y[k] == c
                row.append("-" if not yc.any() else
                           f"{np.mean([(p[yc] == c).float().mean().item() for p in ps]):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def paired_table(preds, y):
    """Same beat, same seed: where do ANN and SNN agree / differ?"""
    names = sorted({n for n, _ in preds})
    an, sn = ("ANN", "SNN") if set(names) == {"ANN", "SNN"} else (names + names)[:2]
    seeds = sorted({s for n, s in preds if n == an} &
                   {s for n, s in preds if n == sn})
    if not seeds:
        return ""
    lines = ["Paired per-beat comparison on DS2 (same seed):", "",
             "| | " + " | ".join(f"seed {s}" for s in seeds) + " |",
             "|---|" + "---|" * len(seeds)]
    rows = {"both correct": [], f"only {an} correct": [], f"only {sn} correct": [],
            "both wrong": [], "same prediction": []}
    for s in seeds:
        a, b = preds[(an, s)] == y, preds[(sn, s)] == y
        rows["both correct"].append(int((a & b).sum()))
        rows[f"only {an} correct"].append(int((a & ~b).sum()))
        rows[f"only {sn} correct"].append(int((~a & b).sum()))
        rows["both wrong"].append(int((~a & ~b).sum()))
        rows["same prediction"].append(
            int((preds[(an, s)] == preds[(sn, s)]).sum()))
    for k, v in rows.items():
        lines.append(f"| {k} | " + " | ".join(f"{x:,}" for x in v) + " |")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--prefix", default="",
                    help="output names: runs/<prefix>summary.md, ... ")
    args = ap.parse_args()
    try:
        with open(args.manifest) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"no {args.manifest} - run train_ecg.py first")
        return

    a = manifest["args"]
    x, y, rec, classes = load_split("ds2", a["use_rr"], a["data_dir"],
                                  a.get("morph", "raw"))
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)
    x_lat = x[:N_LATENCY]

    results, cms, preds = [], {}, {}
    for run in manifest["runs"]:
        ckpt = torch.load(run["checkpoint"])
        cfg = ckpt["config"]
        model = build_model(cfg)
        model.load_state_dict(ckpt["state_dict"])

        torch.manual_seed(cfg["seed"])     # reproducible rate encoding
        m, cm, pred = evaluate(model, loader, cfg)
        preds[(cfg["model"], cfg["seed"])] = pred
        r = {"model": cfg["model"], "seed": cfg["seed"], **m,
             "val_accuracy": ckpt["val_accuracy"],
             "val_macro_f1": ckpt["val_macro_f1"],
             "params": n_params(model),
             "latency_ms_per_sample": latency(model, x_lat, cfg),
             "train_time_s": ckpt["train_time_s"]}
        results.append(r)
        cms[cfg["model"]] = cms.get(cfg["model"], 0) + cm
        print(f"{r['model']} seed={r['seed']}: "
              f"test_f1={r['test_macro_f1']:.3f}  "
              f"test_acc={r['test_accuracy']*100:.2f}%  "
              f"ops/beat={r['ops_per_sample']:,.0f}  "
              f"latency={r['latency_ms_per_sample']:.4f} ms/beat")

        hparams = {k: cfg[k] for k in ("model", "seed", "epochs", "batch",
                                       "lr", "hidden", "timesteps", "beta",
                                       "encoding", "use_rr")}
        with SummaryWriter(run["log_dir"]) as w:
            w.add_hparams(hparams,
                          {f"hparam/{k}": v for k, v in r.items()
                           if k not in ("model", "seed") and v is not None},
                          run_name=".")

    table, stats = summarize(results, a, classes)
    cm_lines = []
    for name, cm in cms.items():
        cm_lines += ["", f"{name} confusion matrix (rows = true, cols = "
                     f"predicted, summed over seeds), classes {classes}:"]
        cm_lines += ["    " + " ".join(f"{v:7d}" for v in row)
                     for row in cm.tolist()]
    table_full = (table + "\n" + "\n".join(cm_lines) + "\n\n" +
                  per_record_table(preds, y, rec, classes) + "\n\n" +
                  paired_table(preds, y))
    np.savez_compressed(os.path.join(OUT_DIR, args.prefix + "ds2_predictions.npz"),
                        y=y.numpy(), rec=rec,
                        **{f"{n.lower()}_seed{s}": p.numpy()
                           for (n, s), p in preds.items()})

    with open(os.path.join(OUT_DIR, args.prefix + "summary.md"), "w", encoding="utf-8") as f:
        f.write(table_full + "\n")
    with open(os.path.join(OUT_DIR, args.prefix + "summary.json"), "w") as f:
        json.dump({"train_args": a, "classes": classes, "runs": results,
                   "confusion_matrices": {k: v.tolist()
                                          for k, v in cms.items()},
                   "stats": {n: {k: {"mean": m, "std": s}
                                 for k, (m, s) in st.items()}
                             for n, st in stats.items()}}, f, indent=2)
    with SummaryWriter(os.path.join(manifest["log_dir"], "summary")) as w:
        w.add_text("summary", table_full)

    print("\n" + table_full)
    print("\nRemember: measure the real thing on the RISC-V core too.")


if __name__ == "__main__":
    main()