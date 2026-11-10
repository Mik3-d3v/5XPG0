#!/usr/bin/env python3
"""benchmark.py — GROUP 4: compare the trained ANN vs SNN.

Loads every checkpoint of the last train_mnist.py run (runs/manifest.json) and
reports, per model and seed, on the full MNIST test set:
  * test accuracy and loss,
  * parameter count,
  * a cost proxy per sample: dense MACs for the ANN, synaptic operations
    (SOPs: one add per spike per fan-out synapse, over all timesteps) for the
    SNN — event-driven hardware only "pays" when spikes move,
  * for the SNN, hidden and output spikes per sample,
  * mean inference latency on the PC CPU (a proxy only — the real target is
    the RISC-V core on the PYNQ-Z2!).

The overview is mean +- std over seeds. It is printed, written to
runs/summary.md and runs/summary.json, and logged to the TensorBoard run of
the training (HPARAMS tab: one row per model/seed, TEXT tab: "summary").

Still to do for a scientific benchmark: on-board timing, and a discussion of
what is being held constant. Those are your choices.

Run:  python train_mnist.py && python benchmark.py
"""

import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from train_mnist import MANIFEST, OUT_DIR, build_model, rate_encode

BATCH = 256
N_LATENCY = 200       # samples per latency batch


@torch.no_grad()
def evaluate(model, loader, is_snn, n_steps):
    """Test loss, accuracy and cost proxies over the whole loader."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    correct = n = 0
    in_spk = hid_spk = out_spk = 0.0
    for x, y in loader:
        if is_snn:
            s_in = rate_encode(x, n_steps)
            s_out, s_hid = model(s_in, return_hidden=True)
            out = s_out.sum(dim=0)
            in_spk += s_in.sum().item()
            hid_spk += s_hid.sum().item()
            out_spk += s_out.sum().item()
        else:
            out = model(x)
        total_loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.shape[0]
    m = {"test_accuracy": correct / n, "test_loss": total_loss / n}
    if is_snn:
        n_hidden = model.fc1.out_features
        m["ops_per_sample"] = (in_spk * n_hidden + hid_spk * 10) / n
        m["spikes_hidden_per_sample"] = hid_spk / n
        m["spikes_out_per_sample"] = out_spk / n
    else:
        m["ops_per_sample"] = model.macs_per_sample()
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency(model, x, is_snn, n_steps, repeat=10):
    """Mean PC-CPU latency in ms per sample (batched)."""
    model.eval()
    run = lambda: model(rate_encode(x, n_steps)).sum(0) if is_snn else model(x)
    for _ in range(3):      # warm-up
        run()
    t0 = time.perf_counter()
    for _ in range(repeat):
        run()
    return (time.perf_counter() - t0) / repeat * 1000.0 / x.shape[0]


# ----------------------------------------------------------------------
METRICS = (  # (key, label, format)
    ("test_accuracy", "test accuracy [%]", lambda v: f"{v*100:.2f}"),
    ("test_loss", "test loss", lambda v: f"{v:.4f}"),
    ("val_accuracy", "val accuracy [%]", lambda v: f"{v*100:.2f}"),
    ("params", "parameters", lambda v: f"{v:.0f}"),
    ("ops_per_sample", "ops/sample (MAC or SOP)", lambda v: f"{v:,.0f}"),
    ("spikes_hidden_per_sample", "hidden spikes/sample", lambda v: f"{v:.1f}"),
    ("spikes_out_per_sample", "output spikes/sample", lambda v: f"{v:.1f}"),
    ("latency_ms_per_sample", "PC latency [ms/sample]", lambda v: f"{v:.4f}"),
    ("train_time_s", "train time [s]", lambda v: f"{v:.1f}"),
)


def summarize(results, train_args):
    """mean +- std over seeds for every metric, as a markdown table."""
    names = sorted({r["model"] for r in results})
    stats = {}
    for name in names:
        rs = [r for r in results if r["model"] == name]
        stats[name] = {}
        for key, _, _ in METRICS:
            vals = [r[key] for r in rs if key in r]
            if vals:
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                stats[name][key] = (statistics.mean(vals), std)

    a = train_args
    lines = [
        f"epochs={a['epochs']}, seeds={a['seeds']}, batch={a['batch']}, "
        f"lr={a['lr']}, hidden={a['hidden']}, timesteps={a['timesteps']}, "
        f"beta={a['beta']}",
        "",
        "| metric (mean +- std over seeds) | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key, label, fmt in METRICS:
        cells = []
        for name in names:
            if key in stats[name]:
                mean, std = stats[name][key]
                cells.append(f"{fmt(mean)} +- {fmt(std)}")
            else:
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "ANN ops are dense MACs; SNN ops are synaptic operations "
              "(one add per spike per fan-out synapse, over all timesteps)."]
    return "\n".join(lines), stats


def main():
    try:
        with open(MANIFEST) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"no {MANIFEST} - run train_mnist.py first")
        return

    tf = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.MNIST("./data", train=False, download=True,
                                    transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False)
    x_lat, _ = next(iter(torch.utils.data.DataLoader(ds, batch_size=N_LATENCY)))

    results = []
    for run in manifest["runs"]:
        ckpt = torch.load(run["checkpoint"])
        cfg = ckpt["config"]
        is_snn = cfg["model"] == "SNN"
        model = build_model(cfg)
        model.load_state_dict(ckpt["state_dict"])

        torch.manual_seed(cfg["seed"])     # reproducible rate encoding
        r = {"model": cfg["model"], "seed": cfg["seed"],
             **evaluate(model, loader, is_snn, cfg["timesteps"]),
             "val_accuracy": ckpt["val_accuracy"],
             "params": n_params(model),
             "latency_ms_per_sample": latency(model, x_lat, is_snn,
                                              cfg["timesteps"]),
             "train_time_s": ckpt["train_time_s"]}
        results.append(r)
        print(f"{r['model']} seed={r['seed']}: "
              f"test_acc={r['test_accuracy']*100:.2f}%  "
              f"ops/sample={r['ops_per_sample']:,.0f}  "
              f"latency={r['latency_ms_per_sample']:.4f} ms/sample")

        hparams = {k: cfg[k] for k in ("model", "seed", "epochs", "batch",
                                       "lr", "hidden", "timesteps", "beta")}
        with SummaryWriter(run["log_dir"]) as w:
            w.add_hparams(hparams,
                          {f"hparam/{k}": v for k, v in r.items()
                           if k not in ("model", "seed")},
                          run_name=".")

    table, stats = summarize(results, manifest["args"])
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"train_args": manifest["args"], "runs": results,
                   "stats": {n: {k: {"mean": m, "std": s}
                                 for k, (m, s) in st.items()}
                             for n, st in stats.items()}}, f, indent=2)
    with SummaryWriter(os.path.join(manifest["log_dir"], "summary")) as w:
        w.add_text("summary", table)

    print("\n" + table)
    print("\nRemember: measure the real thing on the RISC-V core too.")


if __name__ == "__main__":
    main()
