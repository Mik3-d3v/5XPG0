#!/usr/bin/env python3
"""export_ecg.py — quantize the trained ECG ANN/SNN for the PYNQ-Z2 RISC-V core.

The ECG counterpart of ../../ann_vs_snn_mnist/pynq/export_model.py; the
quantisation, the bit-exact integer models and the model.bin layout are the
shared ones of ../../ann_vs_snn_mnist/pynq/quant.py, and the firmware is the
same image (magic "ANN2": int8 inputs, "SNN3": int8 inputs with direct
encoding). For every checkpoint of runs/manifest.json this writes
pynq/export/<name>/:
  model.bin       int8 weights + int32 biases + layer params
  inputs.bin      [N][4*ceil(F/4)] int8 DS2 features, zero-padded to words
  labels.bin      [N] uint8 labels (0 N, 1 S, 2 V)
  ref_logits.bin  ANN: [N][3] int32 logits      } of the INTEGER model on the
  ref_counts.bin  SNN: [N][3] int32 spike counts } PC; the board must match
  float_pred.bin  [N] uint8 predictions of the float model (float inputs)
  meta.json       shapes, scales, calibration, PC metrics (float and int8)

Input quantisation (the features are real-valued, unlike MNIST pixels):
  x_q = clip(round(x / s_x), -127, 127), one symmetric scale s_x for all
  features, shared by ANN and SNN (same input bytes for both models).
  s_x = P(|x|) / 127 with P a high percentile of |x| over the DS1 (training)
  features: the maximum is useless (one RR ratio of 294 against a typical
  |x| of 0.1). P is chosen among CLIP_PCTS by the float/int8 prediction
  agreement on DS1, summed over all exported models - DS2 is not used.
  ANN activation scales are calibrated on the quantized DS1 inputs.

Run (from ann_vs_snn_ecg/, venv active):
  python pynq/export_ecg.py                 # all models, all DS2 beats
  python pynq/export_ecg.py --n-samples 500
  python pynq/export_ecg.py --timesteps 5 --n-samples 2000 --out pynq/export_T       runs/snn_ecg_seed0.pt
      # latency-vs-T study: the trained SNN simulated for another T
      # (-> export_T/snn_ecg_seed0_T5; accuracy is not tuned for that T)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ann_vs_snn_mnist", "pynq"))
from train_ecg import (MANIFEST, build_model, cm_metrics, confusion,  # noqa: E402
                       encode, load_split)
from quant import (ANN_S8_MAGIC, SNN_DIRECT_MAGIC, int_forward_ann,  # noqa: E402
                   int_forward_snn, layer_meta, pack_model, pad4,
                   quantize_ann, quantize_inputs, quantize_snn)

CLIP_PCTS = (99.9, 99.95, 99.99, 99.999)   # candidates for the input scale


def load_model(path):
    ckpt = torch.load(path)
    cfg = ckpt["config"]
    if cfg["model"] == "SNN" and cfg["encoding"] != "direct":
        raise ValueError("only direct encoding is exported (rate: see README)")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt, cfg, model


@torch.no_grad()
def float_pred(model, cfg, x):
    if cfg["model"] == "SNN":
        return model(encode(x, cfg["timesteps"], "direct")).sum(0).argmax(1).numpy()
    return model(x).argmax(1).numpy()


def quantize(ckpt, cfg, s_x, calib_q):
    """-> (integer layers, magic, int forward: int8 [B, F] -> (out, hidden))."""
    sd = ckpt["state_dict"]
    if cfg["model"] == "SNN":
        layers = quantize_snn(sd, in_scale=s_x)
        T = cfg["timesteps"]
        fwd = lambda xq: int_forward_snn(layers, None, direct_x=(T, xq),
                                         return_hidden=True)
        return layers, SNN_DIRECT_MAGIC, fwd
    layers = quantize_ann(sd, calib_q.astype(np.float64) * s_x, s_in=s_x)
    return layers, ANN_S8_MAGIC, lambda xq: int_forward_ann(layers, xq, True)


def int_pred(fwd, xq, bs=4096):
    outs, hids = zip(*(fwd(xq[i:i + bs]) for i in range(0, len(xq), bs)))
    return np.concatenate(outs), np.concatenate(hids)


def choose_scale(models, x1):
    """Input scale from DS1 only: the clip percentile with the highest summed
    float/int8 agreement over the models (ties: the larger percentile)."""
    a = np.abs(x1.numpy())
    rows, best = [], None
    for pct in CLIP_PCTS:
        s_x = float(np.percentile(a, pct)) / 127.0
        xq = quantize_inputs(x1.numpy(), s_x)
        agree = []
        for ckpt, cfg, model in models:
            _, _, fwd = quantize(ckpt, cfg, s_x, xq)
            out, _ = int_pred(fwd, xq)
            agree.append(float((out.argmax(1) == float_pred(model, cfg, x1)).mean()))
        rows.append({"percentile": pct, "s_x": s_x, "agreement_ds1": agree,
                     "zero_inputs": float((xq == 0).mean()),
                     "clipped": float((a / s_x > 127.5).mean())})
        if best is None or sum(agree) >= sum(best["agreement_ds1"]):
            best = rows[-1]
        print(f"  clip at P{pct:<7} s_x={s_x:.4f}  zero inputs "
              f"{rows[-1]['zero_inputs']*100:4.1f}%  DS1 float/int agreement "
              + " ".join(f"{v*100:.2f}" for v in agree), flush=True)
    return best, rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="*",
                   help="checkpoint .pt files (default: all runs in runs/manifest.json)")
    p.add_argument("--n-samples", type=int, default=0,
                   help="DS2 beats to export for the board (0 = all)")
    p.add_argument("--out", default=os.path.join(HERE, "export"))
    p.add_argument("--timesteps", type=int, default=0,
                   help="simulate the SNNs for this T instead of the trained one "
                        "(latency study; folder suffix _T<T>); the input scale "
                        "is the one of the full export (export/*/meta.json)")
    p.add_argument("--input-scale", type=float, default=0.0,
                   help="use this input scale s_x instead of calibrating it "
                        "(sweeps: keep the published scale fixed)")
    p.add_argument("--sample", type=int, default=0,
                   help="board files for a seeded random subset of this many DS2 "
                        "beats (PC metrics stay on all DS2); indices in sample_idx.bin")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--manifest", default=MANIFEST,
                   help="checkpoints to export when none are given")
    a = p.parse_args()

    ckpts = a.checkpoints
    if not ckpts:
        with open(a.manifest) as f:
            ckpts = [r["checkpoint"] for r in json.load(f)["runs"]]
    models = [load_model(c) for c in ckpts]
    cfg0 = models[0][1]
    for _, cfg, _ in models:      # one input format for all models
        assert (cfg["use_rr"], cfg["morph"]) == (cfg0["use_rr"], cfg0["morph"])
    x1, _, _, _ = load_split("ds1", cfg0["use_rr"], cfg0["data_dir"], cfg0["morph"])
    x2, y2, rec2, classes = load_split("ds2", cfg0["use_rr"], cfg0["data_dir"],
                                       cfg0["morph"])

    if a.input_scale:
        best, rows = {"s_x": a.input_scale, "percentile": None}, []
        print(f"input scale given: s_x = {a.input_scale:.5f}")
    elif a.timesteps:
        for _, cfg, _ in models:
            cfg["timesteps"] = a.timesteps
        with open(os.path.join(HERE, "export", os.path.splitext(os.path.basename(
                ckpts[0]))[0], "meta.json")) as f:
            m0 = json.load(f)
        rows = m0["input_calibration"]
        best = next(r for r in rows if r["percentile"] == m0["input_clip_percentile"])
        print(f"T = {a.timesteps}, input scale of the full export")
    else:
        print("input scale (DS1 only):")
        best, rows = choose_scale(models, x1)
    s_x = best["s_x"]
    print(f"-> clip at P{best['percentile']}, s_x = {s_x:.5f}")
    xq1 = quantize_inputs(x1.numpy(), s_x)
    xq2 = quantize_inputs(x2.numpy(), s_x)
    n = len(xq2) if a.n_samples <= 0 else min(a.n_samples, len(xq2))
    n_in = xq2.shape[1]
    yn = y2.numpy()
    sel = np.arange(n)                      # DS2 beats written for the board
    if a.sample:
        sel = np.sort(np.random.default_rng(a.sample_seed).choice(
            len(xq2), min(a.sample, len(xq2)), replace=False))
        n = len(sel)

    for path, (ckpt, cfg, model) in zip(ckpts, models):
        is_snn = cfg["model"] == "SNN"
        layers, magic, fwd = quantize(ckpt, cfg, s_x, xq1)
        T = cfg["timesteps"] if is_snn else 0
        blob = pack_model(layers, magic, T)
        out, hid = int_pred(fwd, xq2)
        pq = out.argmax(1)                  # first max, like the firmware
        pf = float_pred(model, cfg, x2)
        m_f = cm_metrics(confusion(y2, torch.from_numpy(pf), 3), classes)
        m_q = cm_metrics(confusion(y2, torch.from_numpy(pq), 3), classes)

        name = os.path.splitext(os.path.basename(path))[0]
        if a.timesteps:
            name += f"_T{a.timesteps}"
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        ref_file = "ref_counts.bin" if is_snn else "ref_logits.bin"
        with open(os.path.join(d, "model.bin"), "wb") as f:
            f.write(blob)
        pad4(xq2[sel]).tofile(os.path.join(d, "inputs.bin"))
        yn[sel].astype(np.uint8).tofile(os.path.join(d, "labels.bin"))
        np.asarray(rec2)[sel].astype("<i4").tofile(os.path.join(d, "records.bin"))
        out[sel].astype("<i4").tofile(os.path.join(d, ref_file))
        pf[sel].astype(np.uint8).tofile(os.path.join(d, "float_pred.bin"))
        if a.sample:
            sel.astype("<i4").tofile(os.path.join(d, "sample_idx.bin"))
        meta = {
            "model": cfg["model"], "checkpoint": path.replace("\\", "/"),
            "config": cfg, "task": "ecg", "classes": classes,
            "n_samples": n, "timesteps": T, "n_in": n_in,
            "n_out": int(layers[-1]["w"].shape[1]),
            "input_kind": "s8", "input_bytes": int(pad4(xq2[:1]).shape[1]),
            "input_scale": s_x, "input_calibration": rows,
            "input_clip_percentile": best["percentile"],
            "ref_file": ref_file, "model_bytes": len(blob),
            "magic": hex(magic), "layers": layer_meta(layers),
            "pc_test_accuracy_float": m_f["accuracy"],
            "pc_test_accuracy_int": m_q["accuracy"],
            "pc_float_int_agreement": float((pf == pq).mean()),
            "pc_metrics_float": m_f, "pc_metrics_int": m_q,
            "pc_hidden_events_int": float(hid.mean()),
            "zero_inputs_ds2": float((xq2 == 0).mean()),
        }
        if a.sample:
            meta["sample"] = {"n": int(n), "seed": a.sample_seed, "file": "sample_idx.bin",
                              "note": "board files hold this DS2 subset; pc_* metrics "
                                      "are on all DS2 beats"}
        if not is_snn:
            meta["out_scale"] = layers[-1]["acc_scale"]
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        dims = "-".join([str(n_in)] + [str(l["n_out"]) for l in meta["layers"]])
        print(f"{name}: {cfg['model']} {dims}{f' T={T}' if T else ''}, "
              f"{len(blob)} B model, {n} beats -> {d}\n"
              f"  DS2 float: macro-F1 {m_f['macro_f1']:.3f} acc {m_f['accuracy']*100:.2f}%"
              f"   int8: macro-F1 {m_q['macro_f1']:.3f} acc {m_q['accuracy']*100:.2f}%"
              f"   agree {meta['pc_float_int_agreement']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
