#!/usr/bin/env python3
"""export_stream.py — quantize the streaming ECG models (RSNN, GRU) for the board.

Counterpart of export_ecg.py for the models that read the beat sample by
sample (train_ecg.py --models RSNN GRU). Same output files per model in
pynq/export_stream/<name>/ (model.bin, inputs.bin, labels.bin, records.bin,
ref_logits.bin, float_pred.bin, meta.json), same firmware image.

GRU ("GRU1", quant.int_forward_gru): the beat - template samples and the RR
features as int8 (one shared scale, as the ANN), weights int8, hidden state
int8, gates through fixed-point pre-activations and Q15 look-up tables.

RSNN ("RSN1", quant.int_forward_rsnn): the level-crossing encoding of
beat - template runs on the host, exactly as in training, so the board gets
the events the float model saw (one u32 bitmap per sample, T = 126) plus the
3 RR features as int8. Only the weights and the RR features are quantized.

Input scale of the int8 features: as export_ecg.py, P99.99 of |x| over the
DS1 (training) features / 127, shared by all models.

Run (from ann_vs_snn_ecg/, venv active):
  python pynq/export_stream.py                          # runs/manifest_stream.json
  python pynq/export_stream.py runs/rsnn_ecg_seed0.pt --n-samples 500
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
from train_ecg import (N_RR, OUT_DIR, build_model, cm_metrics,  # noqa: E402
                       confusion, load_split)
from quant import (RSNN_MAGIC, int_forward_gru, int_forward_rsnn,  # noqa: E402
                   layer_meta, pack_gru, pack_model, pad4, quantize_gru,
                   quantize_inputs, quantize_rsnn)

MANIFEST_STREAM = os.path.join(OUT_DIR, "manifest_stream.json")
CLIP_PCT = 99.99              # input scale rule chosen on DS1 (export_ecg.py)
BATCH = 2048


def input_scale(data_dir):
    x1, _, _, _ = load_split("ds1", True, data_dir, "diff")
    return float(np.percentile(np.abs(x1.numpy()), CLIP_PCT)) / 127.0


@torch.no_grad()
def export_rsnn(path, ckpt, cfg, model, x2, y2, rec2, classes, s_x, sel, out_root):
    layers = quantize_rsnn(ckpt["state_dict"], s_x, cfg.get("readout_bins", 1))
    blob = pack_model(layers, RSNN_MAGIC, x2.shape[1] - N_RR)
    T = x2.shape[1] - N_RR
    logits, hidden, words, pf, events = [], [], [], [], []
    for i in range(0, len(x2), BATCH):
        xb = x2[i:i + BATCH]
        ev = model.encode(xb)                              # [T, B, n_ev]
        rr_q = quantize_inputs(xb[:, T:].numpy(), s_x)     # [B, 3]
        lo, hi = int_forward_rsnn(layers, ev.numpy(), rr_q)
        logits.append(lo)
        hidden.append(hi)
        pf.append(model(xb).argmax(1).numpy())
        bits = (ev.permute(1, 0, 2).numpy().astype(np.uint32)
                << np.arange(ev.shape[2], dtype=np.uint32)).sum(2)   # [B, T]
        rrw = np.pad(rr_q.view(np.uint8), ((0, 0), (0, 4 - N_RR))).view("<u4")
        words.append(np.concatenate([bits.astype("<u4"), rrw], axis=1))
        events.append(ev.sum((0, 2)).numpy())
    logits, hidden = np.concatenate(logits), np.concatenate(hidden)
    words, pf = np.concatenate(words), np.concatenate(pf)
    events = np.concatenate(events)
    pq = logits.argmax(1)
    yn = y2.numpy()
    m_f = cm_metrics(confusion(y2, torch.from_numpy(pf), 3), classes)
    m_q = cm_metrics(confusion(y2, torch.from_numpy(pq), 3), classes)

    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.bin"), "wb") as f:
        f.write(blob)
    words[sel].tofile(os.path.join(d, "inputs.bin"))
    yn[sel].astype(np.uint8).tofile(os.path.join(d, "labels.bin"))
    rec2[sel].astype("<i4").tofile(os.path.join(d, "records.bin"))
    logits[sel].astype("<i4").tofile(os.path.join(d, "ref_logits.bin"))
    pf[sel].astype(np.uint8).tofile(os.path.join(d, "float_pred.bin"))
    meta = {
        "model": "RSNN", "checkpoint": path.replace("\\", "/"), "config": cfg,
        "task": "ecg", "classes": classes, "n_samples": len(sel), "sample_indices": sel.tolist(),
        "params": sum(q.numel() for q in model.parameters()),
        "timesteps": T, "n_in": int(layers[0]["w"].shape[0]),
        "n_out": int(layers[1]["w"].shape[1]),
        "input_kind": "spikes", "words_per_step": 1,
        "input_bytes": 4 * (T + 1), "input_scale": s_x,
        "input_clip_percentile": CLIP_PCT,
        "ref_file": "ref_logits.bin", "model_bytes": len(blob),
        "magic": hex(RSNN_MAGIC), "layers": layer_meta(layers),
        "out_scale": layers[1]["logit_scale"],
        "pc_test_accuracy_float": m_f["accuracy"],
        "pc_test_accuracy_int": m_q["accuracy"],
        "pc_float_int_agreement": float((pf == pq).mean()),
        "pc_metrics_float": m_f, "pc_metrics_int": m_q,
        "pc_hidden_events_int": float(hidden.mean()),
        "pc_input_events": float(events.mean()),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: RSNN {meta['n_in']}-{layers[0]['w'].shape[1]}-3, T={T}, "
          f"{len(blob)} B model, {len(sel)} beats -> {d}\n"
          f"  DS2 float: macro-F1 {m_f['macro_f1']:.3f} acc {m_f['accuracy']*100:.2f}%"
          f"   int8: macro-F1 {m_q['macro_f1']:.3f} acc {m_q['accuracy']*100:.2f}%"
          f"   agree {meta['pc_float_int_agreement']*100:.2f}%   events/beat "
          f"{events.mean():.1f}, hidden spikes/beat {hidden.mean():.1f}", flush=True)


@torch.no_grad()
def export_gru(path, ckpt, cfg, model, x2, y2, rec2, classes, s_x, sel, out_root):
    T = x2.shape[1] - N_RR
    bins = cfg.get("readout_bins", 1)
    layers = quantize_gru(ckpt["state_dict"], s_x, bins)
    blob = pack_gru(layers)
    xq = quantize_inputs(x2.numpy(), s_x)                  # [N, T + 3]
    logits, hidden = [], []
    for i in range(0, len(xq), BATCH):
        lo, nz = int_forward_gru(layers, xq[i:i + BATCH, :T], xq[i:i + BATCH, T:])
        logits.append(lo)
        hidden.append(nz)
    logits, hidden = np.concatenate(logits), np.concatenate(hidden)
    pf = torch.cat([model(x2[i:i + BATCH]).argmax(1)
                    for i in range(0, len(x2), BATCH)]).numpy()
    pq = logits.argmax(1)
    yn = y2.numpy()
    m_f = cm_metrics(confusion(y2, torch.from_numpy(pf), 3), classes)
    m_q = cm_metrics(confusion(y2, torch.from_numpy(pq), 3), classes)
    rrw = np.pad(xq[:, T:].view(np.uint8), ((0, 0), (0, 4 - N_RR)))
    inputs = np.concatenate([pad4(xq[:, :T]).view(np.uint8), rrw], axis=1)

    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.bin"), "wb") as f:
        f.write(blob)
    inputs[sel].tofile(os.path.join(d, "inputs.bin"))
    yn[sel].astype(np.uint8).tofile(os.path.join(d, "labels.bin"))
    rec2[sel].astype("<i4").tofile(os.path.join(d, "records.bin"))
    logits[sel].astype("<i4").tofile(os.path.join(d, "ref_logits.bin"))
    pf[sel].astype(np.uint8).tofile(os.path.join(d, "float_pred.bin"))
    H = int(layers[0]["w"].shape[0])
    meta = {
        "model": "GRU", "checkpoint": path.replace("\\", "/"), "config": cfg,
        "task": "ecg", "classes": classes, "n_samples": len(sel), "sample_indices": sel.tolist(),
        "params": sum(q.numel() for q in model.parameters()),
        "timesteps": T, "n_in": T, "n_out": int(layers[2]["w"].shape[1]),
        "hidden": H, "readout_bins": bins,
        "input_kind": "s8", "input_bytes": int(inputs.shape[1]),
        "input_scale": s_x, "input_clip_percentile": CLIP_PCT,
        "ref_file": "ref_logits.bin", "model_bytes": len(blob),
        "magic": hex(0x31555247), "layers": layer_meta(layers),
        "out_scale": layers[2]["logit_scale"],
        "macs_per_sample": model.macs_per_sample(),
        "pc_test_accuracy_float": m_f["accuracy"],
        "pc_test_accuracy_int": m_q["accuracy"],
        "pc_float_int_agreement": float((pf == pq).mean()),
        "pc_metrics_float": m_f, "pc_metrics_int": m_q,
        "pc_hidden_nonzero_int": float(hidden.mean()),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: GRU H={H} bins={bins}, T={T}, {len(blob)} B model, {len(sel)} beats -> {d}\n"
          f"  DS2 float: macro-F1 {m_f['macro_f1']:.3f} acc {m_f['accuracy']*100:.2f}%"
          f"   int8: macro-F1 {m_q['macro_f1']:.3f} acc {m_q['accuracy']*100:.2f}%"
          f"   agree {meta['pc_float_int_agreement']*100:.2f}%", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="*",
                   help="checkpoint .pt files (default: runs/manifest_stream.json)")
    p.add_argument("--n-samples", type=int, default=0,
                   help="DS2 beats to export for the board (0 = all)")
    p.add_argument("--sample", type=int, default=0,
                   help="export a random subset of this many DS2 beats (same "
                        "beats for every model; the PC metrics stay on all beats)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "export_stream"))
    a = p.parse_args()

    ckpts = a.checkpoints
    if not ckpts:
        with open(MANIFEST_STREAM) as f:
            ckpts = [r["checkpoint"] for r in json.load(f)["runs"]]
    s_x = None
    for path in ckpts:
        ckpt = torch.load(path)
        cfg = ckpt["config"]
        model = build_model(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        if s_x is None:
            s_x = input_scale(cfg["data_dir"])
            x2, y2, rec2, classes = load_split("ds2", True, cfg["data_dir"], "diff")
            n = len(x2) if a.n_samples <= 0 else min(a.n_samples, len(x2))
            sel = np.arange(n)
            if a.sample:      # random subset, same for every model
                sel = np.sort(np.random.default_rng(a.sample_seed).choice(
                    len(x2), a.sample, replace=False))
        if cfg["model"] == "RSNN":
            export_rsnn(path, ckpt, cfg, model, x2, y2, rec2, classes, s_x, sel, a.out)
        elif cfg["model"] == "GRU":
            export_gru(path, ckpt, cfg, model, x2, y2, rec2, classes, s_x, sel, a.out)
        else:
            raise ValueError(f"no exporter for {cfg['model']} yet")


if __name__ == "__main__":
    main()
