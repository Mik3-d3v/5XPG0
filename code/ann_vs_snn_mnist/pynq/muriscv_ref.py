#!/usr/bin/env python3
"""muriscv_ref.py — map our int8 MNIST ANN onto muRISCV-NN (CMSIS-NN s8) and
export it for the "muriscv" firmware level. No retraining: the same float
checkpoints, the same int8 weights and scales as export_model.py.

Mapping (per layer, CMSIS-NN s8 fully connected):
  weights   our symmetric int8 weights, transposed to row-major [n_out][n_in]
            (one row per output, as CMSIS-NN expects), filter_offset 0
  input     MNIST pixel p (uint8) as int8 (p - 128), input_offset +128,
            so the kernel sees exactly p, as our kernels do
  hidden    our uint8 ReLU activation h as int8 (h - 128): output_offset
            -128 with activation clamp [-128, 127] = ReLU + saturation at
            255; the next layer uses input_offset +128
  biases    int32 at scale s_in * s_w: identical to ours
  requant   the real scale our kernel uses, m_q / 2^24, as a CMSIS Q31
            multiplier + shift (TFLite QuantizeMultiplier convention:
            M = q * 2^shift, q in [0.5, 1), multiplier = round(q * 2^31))
  output    the library's fully-connected kernel only produces int8, so the
            logits are requantised to int8 too: s_out = max |logit| over
            the same 10 000 training images used for the hidden scale,
            / 127; output_offset 0, clamp [-128, 127]. Prediction = first
            argmax of the int8 logits (ties are possible, unlike int32).
Rounding: our kernel computes (acc * m_q + 2^23) >> 24 (one rounding);
CMSIS-NN rounds twice (doubling high multiply, then a rounding shift with
the midpoint away from zero), so a few activations differ by one LSB.

For each checkpoint this writes pynq/export/<name>_muriscv/:
  model.bin       ANN3 layout (see muriscv/ann_muriscv.c)
  inputs.bin      [N][784] int8 (p - 128)
  labels.bin, float_pred.bin   as export_model.py
  ref_logits.bin  [N][10] int32: the int8 logits of the Python emulation of
                  the library (emulate()); host_ref must reproduce them
  meta.json
Run (from code/ann_vs_snn_mnist/, venv active):  python pynq/muriscv_ref.py
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from quant import int_forward_ann, quantize_ann  # noqa: E402

ANN3_MAGIC = 0x334E4E41       # "ANN3"
N_CALIB = 10000


# ------------------------------------------------------------------ CMSIS math
def quantize_multiplier(m):
    """real multiplier -> (Q31 multiplier, shift), TFLite / CMSIS-NN convention."""
    if m == 0:
        return 0, 0
    q, shift = np.frexp(m)                       # m = q * 2^shift, q in [0.5, 1)
    q_fixed = int(round(q * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q_fixed, int(shift)


def requantize(acc, mult, shift):
    """muriscv_nn_requantize, portable C branch (default build), vectorised:
    divide_by_power_of_two(doubling_high_mult_no_sat(acc << left, mult), right)."""
    acc = np.asarray(acc, np.int64)
    left, right = max(shift, 0), max(-shift, 0)
    x = acc * (1 << left)
    assert np.abs(x).max() < 2 ** 31, "int32 overflow in requantize"
    dhm = (x * mult + (1 << 30)) >> 31           # (int64) m1 * m2 + 2^30, >> 31
    mask = (1 << right) - 1
    rem = dhm & mask
    res = dhm >> right
    thr = (mask >> 1) + (res < 0)
    return res + (rem > thr)


def fc_s8(x, layer):
    """muriscv_nn_fully_connected_s8 on int8 inputs x [B, n_in] -> int8 [B, n_out]."""
    w = layer["w"].astype(np.int64)              # [n_out, n_in]
    acc = (x.astype(np.int64) + layer["in_off"]) @ w.T + layer["b"]
    out = requantize(acc, layer["mult"], layer["shift"]) + layer["out_off"]
    return np.clip(out, layer["act_min"], layer["act_max"]).astype(np.int8), acc


def emulate(layers, x):
    """Python model of the library: int8 input [B, n_in] -> int8 logits."""
    for layer in layers:
        x, _ = fc_s8(x, layer)
    return x


# ------------------------------------------------------------------ mapping
def map_ann(sd, calib):
    """Our int8 MNIST ANN (quant.quantize_ann) -> ANN3 layers + the output scale."""
    ours = quantize_ann(sd, calib)               # same weights, biases, m_q
    calib_u8 = np.rint(calib * 255).astype(np.uint8)
    acc_last = int_forward_ann(ours, calib_u8)   # our int32 logits on calibration
    layers, out_scale = map_layers(ours, acc_last, in_off=128)
    return layers, ours, out_scale


def map_layers(ours, acc_last, in_off):
    """Our integer layers (quant.quantize_ann) -> ANN3 layers + the float value
    of one int8 output LSB. acc_last: our int32 logits on the calibration
    inputs (sets the int8 output scale); in_off: input_offset of the first
    layer (+128 for uint8 inputs stored as p - 128, 0 for int8 inputs)."""
    layers = []
    for i, l in enumerate(ours):
        last = i == len(ours) - 1
        if last:   # int8 logits: max |acc| over the calibration images -> 127
            m_real = 127.0 / float(np.abs(acc_last).max())
        else:      # exactly the scale of our hidden requantisation
            m_real = l["p0"] / 2.0 ** 24
        mult, shift = quantize_multiplier(m_real)
        layers.append({
            "w": np.ascontiguousarray(l["w"].T),          # [n_out][n_in]
            "b": l["b"].astype(np.int64),
            "mult": mult, "shift": shift, "m_real": m_real,
            "in_off": in_off if i == 0 else 128,          # hidden: h - 128 -> +128
            "out_off": 0 if last else -128,
            "act_min": -128, "act_max": 127,
            "acc_scale": l["acc_scale"],
        })
    out_scale = layers[-1]["acc_scale"] / layers[-1]["m_real"]   # float per int8 LSB
    return layers, out_scale


def pack_ann3(layers):
    hdr_words = 4 + 7 * len(layers)
    body = b""
    descs = []
    off = 4 * hdr_words
    for l in layers:
        n_out, n_in = l["w"].shape
        w_off = off + len(body)
        body += l["w"].astype(np.int8).tobytes()
        body += b"\0" * (-len(body) % 4)
        b_off = off + len(body)
        body += l["b"].astype("<i4").tobytes()
        body += struct.pack("<4i", l["in_off"], l["out_off"], l["act_min"], l["act_max"])
        descs += [n_in, n_out, l["mult"] & 0xFFFFFFFF, l["shift"] & 0xFFFFFFFF,
                  w_off, b_off, n_in]
    hdr = [ANN3_MAGIC, len(layers), layers[0]["w"].shape[1], 0] + descs
    return struct.pack(f"<{len(hdr)}I", *hdr) + body


# ------------------------------------------------------------------ export
def calib_images():
    import torchvision                  # MNIST only (the ECG exporter has none)
    ds = torchvision.datasets.MNIST("./data", train=True, download=True)
    return ds.data[:N_CALIB].reshape(N_CALIB, -1).double().numpy() / 255.0


@torch.no_grad()
def export_one(ckpt_path, test_x, test_y, calib, out_root):
    ckpt = torch.load(ckpt_path)
    cfg = ckpt["config"]
    assert cfg["model"] == "ANN"
    layers, ours, out_scale = map_ann(ckpt["state_dict"], calib)
    from train_mnist import build_model
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pf = model(torch.from_numpy(test_x / 255.0).float()).argmax(1).numpy()
    xq = (test_x.astype(np.int16) - 128).astype(np.int8)
    logits = emulate(layers, xq).astype(np.int64)
    pm = logits.argmax(1)
    po = int_forward_ann(ours, test_x).argmax(1)
    name = os.path.splitext(os.path.basename(ckpt_path))[0] + "_muriscv"
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    blob = pack_ann3(layers)
    open(os.path.join(d, "model.bin"), "wb").write(blob)
    xq.tofile(os.path.join(d, "inputs.bin"))
    test_y.astype(np.uint8).tofile(os.path.join(d, "labels.bin"))
    pf.astype(np.uint8).tofile(os.path.join(d, "float_pred.bin"))
    logits.astype("<i4").tofile(os.path.join(d, "ref_logits.bin"))
    top = np.sort(logits, 1)
    meta = {
        "model": "ANN", "variant": "muriscv", "checkpoint": ckpt_path.replace("\\", "/"),
        "config": cfg, "n_samples": int(len(test_y)), "timesteps": 0,
        "n_in": int(xq.shape[1]), "n_out": int(logits.shape[1]),
        "input_kind": "s8", "input_zero": -128, "input_bytes": int(xq.shape[1]),
        "ref_file": "ref_logits.bin", "model_bytes": len(blob),
        "out_scale": out_scale, "magic": hex(ANN3_MAGIC),
        "layers": [{k: (v if not isinstance(v, np.ndarray) else None)
                    for k, v in l.items() if k not in ("w", "b")} |
                   {"n_in": int(l["w"].shape[1]), "n_out": int(l["w"].shape[0])}
                   for l in layers],
        "pc_test_accuracy_float": float((pf == test_y).mean()),
        "pc_test_accuracy_int": float((pm == test_y).mean()),
        "pc_test_accuracy_ours_int8": float((po == test_y).mean()),
        "agreement_with_ours_int8": float((pm == po).mean()),
        "agreement_with_float": float((pm == pf).mean()),
        "argmax_ties": int((top[:, -1] == top[:, -2]).sum()),
        "saturated_logits": int((np.abs(logits) >= 127).sum()),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d, meta


def main():
    import torchvision
    from train_mnist import MANIFEST
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="*")
    p.add_argument("--out", default=os.path.join(HERE, "export"))
    a = p.parse_args()
    ckpts = a.checkpoints
    if not ckpts:
        with open(MANIFEST) as f:
            ckpts = [r["checkpoint"] for r in json.load(f)["runs"]
                     if os.path.basename(r["checkpoint"]).startswith("ann")]
    ds = torchvision.datasets.MNIST("./data", train=False, download=True)
    test_x = ds.data.reshape(len(ds), -1).numpy().astype(np.uint8)
    test_y = ds.targets.numpy()
    calib = calib_images()
    for c in ckpts:
        d, m = export_one(c, test_x, test_y, calib, a.out)
        print(f"{os.path.basename(d)}: float {m['pc_test_accuracy_float']*100:.2f}%  "
              f"ours int8 {m['pc_test_accuracy_ours_int8']*100:.2f}%  "
              f"muriscv {m['pc_test_accuracy_int']*100:.2f}%  agree w/ ours "
              f"{m['agreement_with_ours_int8']*100:.2f}%  ties {m['argmax_ties']}  "
              f"saturated {m['saturated_logits']}  -> {d}")


if __name__ == "__main__":
    main()
