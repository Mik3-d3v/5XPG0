#!/usr/bin/env python3
"""export_model.py — quantize trained ANN/SNN checkpoints for the PYNQ-Z2 RISC-V core.

For every checkpoint (default: all runs in runs/manifest.json) this writes
pynq/export/<name>/:
  model.bin       int8 weights + int32 biases + layer params (layout below)
  inputs.bin      the MNIST test inputs as the board gets them
                    SNN: rate-encoded spikes, [N][T][words] uint32 bitmaps
                    ANN: raw pixels, [N][784] uint8
  labels.bin      [N] uint8 test labels
  ref_counts.bin  SNN: [N][n_out] int32 output spike counts  } of the INTEGER
  ref_logits.bin  ANN: [N][n_out] int32 output logits        } model on the PC;
                  the board must reproduce these exactly
  float_pred.bin  [N] uint8 predictions of the float model (same inputs)
  meta.json       shapes, scales, PC accuracies

Why quantize: the float32 weights (~400 KB) do not fit the 256 KB BRAM and
PicoRV32 has no FPU. Both models use the same weight scheme: per layer one
scale s_w = max|W| / 127, W_q = round(W / s_w) as int8, int32 biases.

SNN (snnTorch Leaky, reset_delay=True, reset by subtraction):
  b_q = round(b / s_w), thr_q = round(thr / s_w), beta as Q16;
  mem_t = ((beta_q16 * mem_{t-1}) >> 16) + I_t - spk_{t-1} * thr_q
  spk_t = mem_t > thr_q
  Spikes are binary, so the scales of different layers never mix. Inputs are
  made exactly as in benchmark.py (rate_encode, test set, batches of 256,
  torch.manual_seed(seed)): the board sees the same spike trains.

ANN (Linear-ReLU-...-Linear):
  inputs are the uint8 pixels (ToTensor = pixel / 255, so this is exact);
  acc = W_q x_q + b_q            int32, scale s_acc = s_w * s_in
  hidden: h_q = min(255, (relu(acc) * m_q + 2^23) >> 24)   uint8,
          m_q = round(s_acc / s_h * 2^24), s_h = max hidden activation / 255
          calibrated on the first 10 000 TRAINING images (never the test set)
  output: logits = acc (int32), prediction = first argmax.

model.bin layout (little endian, mirrored in firmware/snn_layout.h):
  u32 magic ("SNN2" or "ANN1"), n_layers, n_in, timesteps (ANN: 0)
  per layer: u32 n_in, n_out, p0, p1, w_off, b_off, w_stride
             SNN: p0 = beta_q16, p1 = thr_q;  ANN: p0 = m_q (0 = output), p1 = 0
             (offsets in bytes from the start of model.bin)
  weights int8 [n_in][w_stride] (input-major, each row zero-padded to a
             multiple of 16 bytes so the fast firmware can load words),
  biases int32 [n_out]

Run (from ann_vs_snn_mnist/, venv active):
  python pynq/export_model.py                     # all models, 1000 test images
  python pynq/export_model.py --n-samples 10000   # the full test set
  python pynq/export_model.py runs/ann_mnist_seed0.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from train_mnist import MANIFEST, build_model, rate_encode  # noqa: E402
from quant import (ANN_MAGIC, SNN_MAGIC, int_forward_ann,  # noqa: E402
                   int_forward_snn, layer_meta, pack_model, pack_spikes,
                   quantize_ann, quantize_snn)

BATCH = 256                   # same as benchmark.py (keeps the RNG stream equal)
N_CALIB = 10000               # training images for the ANN activation scales


# quantisation, integer reference models and model.bin packing: quant.py


def calib_images():
    ds = torchvision.datasets.MNIST("./data", train=True, download=True)
    return ds.data[:N_CALIB].reshape(N_CALIB, -1).double().numpy() / 255.0


@torch.no_grad()
def export_one(ckpt_path, ds, n_samples, out_root):
    ckpt = torch.load(ckpt_path)
    cfg = ckpt["config"]
    is_snn = cfg["model"] == "SNN"
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if is_snn:
        T = cfg["timesteps"]
        layers = quantize_snn(ckpt["state_dict"])
        blob = pack_model(layers, SNN_MAGIC, T)
    else:
        T = 0
        layers = quantize_ann(ckpt["state_dict"], calib_images())
        blob = pack_model(layers, ANN_MAGIC, 0)

    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False)
    torch.manual_seed(cfg["seed"])          # exactly as benchmark.py
    n = ok_f = ok_q = agree = 0
    packed, labels, refs, fpreds = [], [], [], []
    for x, y in loader:
        if is_snn:
            s_in = rate_encode(x, T)
            pf = model(s_in).sum(0).argmax(1).numpy()
            q_in = s_in
            oq = int_forward_snn(layers, s_in.numpy())
        else:
            pf = model(x).argmax(1).numpy()
            q_in = np.rint(x.reshape(len(x), -1).numpy() * 255).astype(np.uint8)
            oq = int_forward_ann(layers, q_in)
        pq = oq.argmax(1)                    # first max, like the firmware
        yn = y.numpy()
        ok_f += (pf == yn).sum()
        ok_q += (pq == yn).sum()
        agree += (pf == pq).sum()
        n += len(yn)
        keep = max(0, min(len(yn), n_samples - (n - len(yn))))
        if keep:
            packed.append(pack_spikes(q_in[:, :keep].numpy()) if is_snn
                          else q_in[:keep])
            labels.append(yn[:keep])
            refs.append(oq[:keep])
            fpreds.append(pf[:keep])

    ref_file = "ref_counts.bin" if is_snn else "ref_logits.bin"
    with open(os.path.join(out, "model.bin"), "wb") as f:
        f.write(blob)
    np.concatenate(packed).tofile(os.path.join(out, "inputs.bin"))
    np.concatenate(labels).astype(np.uint8).tofile(os.path.join(out, "labels.bin"))
    np.concatenate(refs).astype("<i4").tofile(os.path.join(out, ref_file))
    np.concatenate(fpreds).astype(np.uint8).tofile(os.path.join(out, "float_pred.bin"))

    n_in = int(layers[0]["w"].shape[0])
    n_kept = sum(len(l) for l in labels)
    meta = {
        "model": cfg["model"], "checkpoint": ckpt_path, "config": cfg,
        "n_samples": n_kept, "timesteps": T, "n_in": n_in,
        "n_out": int(layers[-1]["w"].shape[1]),
        "words_per_step": (n_in + 31) // 32,
        "input_bytes": T * 4 * ((n_in + 31) // 32) if is_snn else n_in,
        "ref_file": ref_file,
        "model_bytes": len(blob),
        "layers": layer_meta(layers),
        "pc_test_accuracy_float": ok_f / n,
        "pc_test_accuracy_int": ok_q / n,
        "pc_float_int_agreement": agree / n,
    }
    if not is_snn:   # float logits = int logits * out_scale (for the test loss)
        meta["out_scale"] = layers[-1]["acc_scale"]
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    dims = "-".join([str(n_in)] + [str(l["n_out"]) for l in meta["layers"]])
    print(f"{name}: {cfg['model']} {dims}, {len(blob)} B model, {n_kept} samples -> {out}\n"
          f"  full test set: float {ok_f / n * 100:.2f}%  int8 {ok_q / n * 100:.2f}%  "
          f"float/int agree {agree / n * 100:.2f}%")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="*",
                   help="checkpoint .pt files (default: all runs in runs/manifest.json)")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="test images to export for the board (default 1000)")
    p.add_argument("--out", default=os.path.join(HERE, "export"))
    a = p.parse_args()

    ckpts = a.checkpoints
    if not ckpts:
        with open(MANIFEST) as f:
            ckpts = [r["checkpoint"] for r in json.load(f)["runs"]]
    ds = torchvision.datasets.MNIST("./data", train=False, download=True,
                                    transform=torchvision.transforms.ToTensor())
    for c in ckpts:
        export_one(c, ds, a.n_samples, a.out)


if __name__ == "__main__":
    main()
