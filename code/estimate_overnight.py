#!/usr/bin/env python3
"""estimate_overnight.py — extrapolate the full overnight run from the smoke run.

Smoke = 1 seed, 1 epoch, 50 board samples (overnight.sh smoke). Per step:
  training   smoke time x epochs x seeds (data loading counted every epoch:
             an upper bound); parallel steps of one job: sum / PAR, but at
             least the longest single step
  export     smoke time x seeds (PC evaluation on all DS2 / all test images)
  board      fixed part (smoke time - 50 x per-sample time) + N x per-sample
             time; per-sample = board latency + host overhead, measured in
             the smoke run. Configurations without a smoke board run are
             scaled from the most expensive one of their kind (SNN: ~ hidden x T,
             ANN: ~ hidden), which only feeds the estimate.
Usage: python code/estimate_overnight.py [smoke results dir]
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "docs", "results", "overnight_2026-10-12", "smoke")
PAR = 6
T_SMOKE = {r["step"]: float(r["seconds"]) for r in csv.DictReader(open(os.path.join(SM, "timing.csv")))
           if r["status"] == "ok"}
ECG_X = os.path.join(HERE, "ann_vs_snn_ecg", "pynq", "export_overnight_smoke")
MN_X = os.path.join(HERE, "ann_vs_snn_mnist", "pynq", "export_overnight_smoke")


def per_sample(exp, fw):
    """seconds per board sample (latency + host overhead) from a smoke run"""
    p = os.path.join(exp, f"board_summary_{fw}.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    return (s["latency_ms_mean"] + s["host_ms_mean"]) / 1e3


def board(step, exp, fw, n, scale=1.0, ref=None):
    """full board time of one step; ref = (smoke step, export) to scale from"""
    ps = per_sample(exp, fw) if os.path.isdir(exp) else None
    t = T_SMOKE.get(step)
    if ps is None or t is None:
        rstep, rexp = ref
        ps, t = per_sample(rexp, fw) * scale, T_SMOKE[rstep]
        fixed = t - 50 * per_sample(rexp, fw)
    else:
        fixed = t - 50 * ps
    return max(fixed, 0) + n * ps


def train(steps, epochs, seeds):
    ts = [T_SMOKE[s] * epochs * seeds for s in steps if s in T_SMOKE]
    return max(sum(ts) / PAR, max(ts)) if ts else 0.0


def main():
    est = {}
    ecg = ["ann_h8", "ann_h16", "ann_h32", "ann_h64", "ann_h128"] + \
          [f"snn_h{h}_T{T}" for h in (16, 32, 64) for T in (2, 5, 10)]
    # A
    a_tr = train([f"A_train_{c}" for c in ecg], 10, 3)
    a_ex = sum(T_SMOKE.get(f"A_export_{c}", 0) * 3 for c in ecg)
    a_bd = 0.0
    for c in ecg:
        h = int(c.split("_h")[1].split("_")[0])
        for fw in (["fast", "hw"] if c.startswith("snn") else ["fast"]):
            if c.startswith("snn"):
                T = int(c.split("_T")[1])
                ref = ("A_board_snn_h64_T10_s0_" + fw, os.path.join(ECG_X, "ecgA_snn_h64_T10_seed0"))
                sc = (h / 64) * (T / 10)
            else:
                ref = ("A_board_ann_h128_s0_fast", os.path.join(ECG_X, "ecgA_ann_h128_seed0"))
                sc = h / 128
            a_bd += 3 * board(f"A_board_{c}_s0_{fw}", os.path.join(ECG_X, f"ecgA_{c}_seed0"),
                              fw, 5000, sc, ref)
    est["A"] = (a_tr, a_ex, a_bd)
    # B
    Ts = (2, 4, 6, 8, 12)
    b_tr = train([f"B_train_T{T}" for T in Ts], 5, 2)
    b_ex = sum(T_SMOKE.get(f"B_export_T{T}", 0) * 2 for T in Ts)
    b_bd = sum(2 * board(f"B_board_T{T}_s0_{fw}", os.path.join(MN_X, f"mnistB_snn_T{T}_seed0"), fw,
                         2000, T / 12, (f"B_board_T12_s0_{fw}",
                                        os.path.join(MN_X, "mnistB_snn_T12_seed0")))
               for T in Ts for fw in ("fast", "hw"))
    est["B"] = (b_tr, b_ex, b_bd)
    # C
    c_tr = train([f"C_train_h{h}" for h in (16, 32)], 10, 3)
    c_ex = sum(T_SMOKE.get(f"C_export_h{h}", 0) for h in (16, 32))   # all seeds in one call
    c_bd = sum(3 * board(f"C_board_h{h}_s0", os.path.join(ECG_X, f"ecgC_gru_h{h}_seed0"), "fast",
                         200, h / 32, ("C_board_h32_s0", os.path.join(ECG_X, "ecgC_gru_h32_seed0")))
               for h in (16, 32))
    est["C"] = (c_tr, c_ex, c_bd)
    # D
    d_tr = T_SMOKE.get("D_train", 0) * 5
    d_ex = sum(T_SMOKE.get(f"D_export_{m}", 0) for m in ("ann", "snn"))
    d_bd = sum(board(f"D_board_{m}", os.path.join(MN_X, f"mnistD_{m}_seed2"), "fast", 10000)
               for m in ("ann", "snn"))
    est["D"] = (d_tr, d_ex, d_bd)
    # E: smoke ran ann_h64 and snn_h64_T10 (1 epoch, 5 folds)
    e_one = {k: T_SMOKE.get(f"E_cv_{k}", 0) * 10 for k in ("ann_h64", "snn_h64_T10")}
    e_ts = [e_one["ann_h64"] * h / 64 if c.startswith("ann") else
            e_one["snn_h64_T10"] * (int(c.split("_T")[1]) / 10) for c in ecg
            for h in [int(c.split("_h")[1].split("_")[0])]]
    est["E"] = (max(sum(e_ts) / PAR, max(e_ts)), 0.0, 0.0)

    tot = 0.0
    print(f"{'job':4s} {'train':>8s} {'export':>8s} {'board':>8s} {'total':>8s} {'cumulative':>11s}")
    for j, (t, e, b) in est.items():
        s = t + e + b
        tot += s
        print(f"{j:4s} {t / 60:7.0f}m {e / 60:7.0f}m {b / 60:7.0f}m {s / 60:7.0f}m "
              f"{tot / 3600:9.2f} h")
    print(f"total {tot / 3600:.2f} h  (smoke steps used: {len(T_SMOKE)})")


if __name__ == "__main__":
    main()
