#!/usr/bin/env python3
"""export_ecg_muriscv.py — the ECG ANN on muRISCV-NN (external ANN reference).

The ECG counterpart of ../../ann_vs_snn_mnist/pynq/muriscv_ref.py: maps our
int8 ECG ANN onto the CMSIS-NN s8 fully-connected layer of the unmodified
muRISCV-NN library, for the "muriscv" firmware level. No retraining and no
new calibration: the same checkpoints, the same input scale s_x (read from
pynq/export/<name>/meta.json, written by export_ecg.py) and the same int8
weights, biases and hidden scales as our ANN, which is checked against the
existing ref_logits.bin first.

Mapping (see muriscv_ref.map_layers):
  input     the int8 beat features x_q as they are, input_offset 0
  hidden    our uint8 ReLU activation h as int8 (h - 128), output_offset -128
  output    int8 logits (the library's only output type): 127 / max |our
            int32 logit| over the DS1 (training) beats, output_offset 0

For each ANN checkpoint this writes pynq/export/<name>_muriscv/:
  model.bin (ANN3), inputs.bin (the same padded int8 beats as export_ecg.py),
  labels.bin, records.bin, float_pred.bin (copied), ref_logits.bin ([N][3]
  int32: the int8 logits of the Python model of the library), meta.json
Run from code/ann_vs_snn_ecg/ (venv active), after export_ecg.py:
  python pynq/export_ecg_muriscv.py
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_PYNQ = os.path.join(HERE, "..", "..", "ann_vs_snn_mnist", "pynq")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, MNIST_PYNQ)
from train_ecg import MANIFEST, cm_metrics, confusion, load_split  # noqa: E402
from quant import int_forward_ann, quantize_ann, quantize_inputs  # noqa: E402
from muriscv_ref import ANN3_MAGIC, emulate, map_layers, pack_ann3  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="*",
                   help="ANN checkpoints (default: the ANNs of runs/manifest.json)")
    p.add_argument("--export", default=os.path.join(HERE, "export"))
    a = p.parse_args()
    ckpts = a.checkpoints
    if not ckpts:
        with open(MANIFEST) as f:
            ckpts = [r["checkpoint"] for r in json.load(f)["runs"]
                     if os.path.basename(r["checkpoint"]).startswith("ann")]

    data = {}
    for path in ckpts:
        name = os.path.splitext(os.path.basename(path))[0]
        src = os.path.join(a.export, name)
        with open(os.path.join(src, "meta.json")) as f:
            m0 = json.load(f)
        cfg, s_x, classes = m0["config"], m0["input_scale"], m0["classes"]
        key = (cfg["use_rr"], cfg["morph"], cfg["data_dir"])
        if key not in data:
            x1, _, _, _ = load_split("ds1", cfg["use_rr"], cfg["data_dir"], cfg["morph"])
            x2, y2, _, _ = load_split("ds2", cfg["use_rr"], cfg["data_dir"], cfg["morph"])
            data[key] = (x1.numpy(), x2.numpy(), y2.numpy())
        x1, x2, y2 = data[key]
        xq1, xq2 = quantize_inputs(x1, s_x), quantize_inputs(x2, s_x)

        # our int8 ANN exactly as export_ecg.py quantizes it
        sd = torch.load(path)["state_dict"]
        ours = quantize_ann(sd, xq1.astype(np.float64) * s_x, s_in=s_x)
        acc2 = int_forward_ann(ours, xq2)
        n = m0["n_samples"]
        ref = np.fromfile(os.path.join(src, "ref_logits.bin"), "<i4").reshape(-1, 3)
        assert (acc2[:n] == ref).all(), f"{name}: does not reproduce our int8 ANN"

        layers, out_scale = map_layers(ours, int_forward_ann(ours, xq1), in_off=0)
        logits = emulate(layers, xq2[:n]).astype(np.int64)
        pm, po = logits.argmax(1), acc2[:n].argmax(1)
        pf = np.fromfile(os.path.join(src, "float_pred.bin"), np.uint8)[:n]
        y = y2[:n]
        m_q = cm_metrics(confusion(torch.from_numpy(y), torch.from_numpy(pm), 3), classes)
        m_o = cm_metrics(confusion(torch.from_numpy(y), torch.from_numpy(po), 3), classes)
        m_f = cm_metrics(confusion(torch.from_numpy(y), torch.from_numpy(pf), 3), classes)

        d = os.path.join(a.export, name + "_muriscv")
        os.makedirs(d, exist_ok=True)
        blob = pack_ann3(layers)
        with open(os.path.join(d, "model.bin"), "wb") as f:
            f.write(blob)
        for fn in ("inputs.bin", "labels.bin", "records.bin", "float_pred.bin"):
            shutil.copyfile(os.path.join(src, fn), os.path.join(d, fn))
        logits.astype("<i4").tofile(os.path.join(d, "ref_logits.bin"))
        top = np.sort(logits, 1)
        meta = {
            "model": "ANN", "variant": "muriscv", "checkpoint": path.replace("\\", "/"),
            "config": cfg, "task": "ecg", "classes": classes,
            "n_samples": n, "timesteps": 0, "n_in": m0["n_in"], "n_out": 3,
            "input_kind": "s8", "input_zero": 0, "input_bytes": m0["input_bytes"],
            "input_scale": s_x, "ref_file": "ref_logits.bin", "model_bytes": len(blob),
            "magic": hex(ANN3_MAGIC), "out_scale": out_scale,
            "layers": [{k: v for k, v in l.items() if k not in ("w", "b")} |
                       {"n_in": int(l["w"].shape[1]), "n_out": int(l["w"].shape[0])}
                       for l in layers],
            "pc_test_accuracy_float": m_f["accuracy"],
            "pc_test_accuracy_int": m_q["accuracy"],
            "pc_metrics_float": m_f, "pc_metrics_int": m_q, "pc_metrics_ours_int8": m_o,
            "agreement_with_ours_int8": float((pm == po).mean()),
            "agreement_with_float": float((pm == pf).mean()),
            "argmax_ties": int((top[:, -1] == top[:, -2]).sum()),
            "saturated_logits": int((np.abs(logits) >= 127).sum()),
        }
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{name}_muriscv: macro-F1 float {m_f['macro_f1']:.3f}  ours int8 "
              f"{m_o['macro_f1']:.3f}  muriscv {m_q['macro_f1']:.3f}  agree w/ ours "
              f"{meta['agreement_with_ours_int8'] * 100:.2f}%  ties {meta['argmax_ties']}  "
              f"saturated {meta['saturated_logits']}  {len(blob)} B -> {d}", flush=True)


if __name__ == "__main__":
    main()
