#!/usr/bin/env python3
"""mac_coproc_summary.py - tables of the ANN MAC coprocessor study
(docs/mac_coprocessor.md) from the cycle-exact RTL simulations in
docs/results/mac_coproc/sim_rtl/ (pynqz2_riscv_flow/sim/soc_sim.py) and the
published board runs. Read-only on the inputs.

Levels: fast = software (stock SoC), hw = SNN kernel on the SNN unit,
mac = image with both units (ANN / GRU / SNN first layer on the MAC unit,
SNN / RSNN neurons on the SNN unit).

Writes docs/results/mac_coproc/summary_sim_rtl.md and .json.
"""
import csv, gzip, json, os, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "docs", "results")
SIM = os.path.join(RES, "mac_coproc", "sim_rtl")
CLK = 100e6


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def sim(name):
    p = os.path.join(SIM, name + ".csv")
    return read(p) if os.path.exists(p) else None


def ms(c): return c / CLK * 1e3
def mstd(v): return (float(st.mean(v)), float(st.stdev(v)) if len(v) > 1 else 0.0)


out = {"bit_exact": {}, "feedforward": {}, "streaming": {}, "board_cycle_check": {}}

# ---------------------------------------------------------------- bit-exactness, board cycle check
for f in sorted(os.listdir(SIM)):
    if not f.endswith(".csv"):
        continue
    rows = read(os.path.join(SIM, f)).values()
    n = len(rows)
    out["bit_exact"][f[:-4]] = [sum(int(r["outputs_match"]) for r in rows), n]
    rc = [r for r in rows if r["ref_cycles"] != ""]
    if rc and not f.startswith("stream_gru") or f == "stream_gru_seed0_fast.csv":
        # (the GRU mac runs carry the board's *software* GRU cycles, for pairing only)
        out["board_cycle_check"][f[:-4]] = dict(
            n=len(rc), same_cycles=sum(int(r["ref_cycles"]) == int(r["cycles"]) for r in rc),
            same_outputs=sum(int(r["ref_outputs_match"]) for r in rc))

# ---------------------------------------------------------------- feedforward (paired subsets)
for task, seeds in (("ecg", [0, 1, 2]), ("mnist", [0, 1])):
    per = {k: [] for k in ("ann_fast", "ann_mac", "snn_hw", "snn_mac")}
    pair = {"snn_mac_faster_than_ann_mac": [], "snn_hw_faster_than_ann_fast": []}
    n_s = None
    for s in seeds:
        d = {k: sim(f"{task}_{k.split('_')[0]}_seed{s}_{k.split('_')[1]}") for k in per}
        if any(v is None for v in d.values()):
            continue
        idx = sorted(set.intersection(*(set(v) for v in d.values())))
        n_s = len(idx)
        cyc = {k: np.array([int(d[k][i]["cycles"]) for i in idx], float) for k in d}
        for k in per:
            per[k].append(float(ms(cyc[k].mean())))
        pair["snn_mac_faster_than_ann_mac"].append([int((cyc["snn_mac"] < cyc["ann_mac"]).sum()), len(idx)])
        pair["snn_hw_faster_than_ann_fast"].append([int((cyc["snn_hw"] < cyc["ann_fast"]).sum()), len(idx)])
    if n_s is None:
        continue
    r = {k: mstd(v) for k, v in per.items()}
    r.update(n_per_seed=n_s, seeds=seeds, pairs=pair,
             ann_speedup=float(st.mean(a / b for a, b in zip(per["ann_fast"], per["ann_mac"]))),
             snn_mac_vs_hw=float(st.mean(a / b for a, b in zip(per["snn_hw"], per["snn_mac"]))),
             snn_over_ann_sw_hw=float(st.mean(a / b for a, b in zip(per["snn_hw"], per["ann_fast"]))),
             snn_over_ann_both_units=float(st.mean(a / b for a, b in zip(per["snn_mac"], per["ann_mac"]))))
    out["feedforward"][task] = r

# ---------------------------------------------------------------- streaming
gm, gf, rr, pairs = [], [], [], []
for s in (0, 1, 2):
    g = sim(f"stream_gru_seed{s}_mac")
    r_ = sim(f"stream_rsnn_seed{s}_mac")
    if g is None:
        continue
    bf = read(os.path.join(RES, f"ecg_streaming/board_subset/gru_seed{s}_fast_board_results.csv.gz"))
    bh = read(os.path.join(RES, f"ecg_streaming/board_subset/rsnn_seed{s}_hw_board_results.csv.gz"))
    idx = sorted(g)
    cg = np.array([int(g[i]["cycles"]) for i in idx], float)
    cf = np.array([int(bf[i]["cycles"]) for i in idx], float)
    ch = np.array([int(bh[i]["cycles"]) for i in idx], float)   # RSNN hw board = mac image (checked)
    gm.append(ms(cg.mean())); gf.append(ms(cf.mean()))
    pairs.append([int((ch < cg).sum()), len(idx)])
    if r_ is not None:
        rr.append(ms(np.mean([int(v["cycles"]) for v in r_.values()])))
if gm:
    out["streaming"] = dict(
        gru_mac=mstd(gm), gru_fast_board_same_beats=mstd(gf), rsnn_mac_200=mstd(rr) if rr else None,
        gru_speedup=float(st.mean(a / b for a, b in zip(gf, gm))),
        rsnn_faster_than_gru_mac=pairs, n_gru_per_seed=len(idx))

# ---------------------------------------------------------------- markdown
B = json.load(open(os.path.join(HERE, "results.json")))
FF, SS = B["feedforward"], B["streaming"]
f2 = lambda v, d=2: f"{v[0]:.{d}f} ± {v[1]:.{d}f}"
L = ["# ANN MAC coprocessor: cycle-exact RTL simulation results",
     "", "Generated by docs/report_analysis/mac_coproc_summary.py from docs/results/mac_coproc/sim_rtl/.",
     "Whole SoC in xsim (PicoRV32, BRAM wait states, coprocessors), real firmware images, real exported",
     "samples; the simulation reproduces the board's cycle counts exactly (see the cycle check below).",
     "Random subsets (fixed seed), the same samples for every level; mean ± std over model seeds.", ""]
L += ["## Feedforward (same samples for all four levels)", "",
      "| | samples / seed | ANN sw | ANN + MAC | ANN speed-up | SNN + SNN unit | SNN + both units | SNN / ANN, sw-ANN vs hw-SNN | SNN / ANN, both units |",
      "|---|---|---|---|---|---|---|---|---|"]
for t, r in out["feedforward"].items():
    L.append(f"| {t.upper()} | {r['n_per_seed']} | {f2(r['ann_fast'], 3)} ms | {f2(r['ann_mac'], 3)} ms | "
             f"{r['ann_speedup']:.2f}× | {f2(r['snn_hw'], 3)} ms | {f2(r['snn_mac'], 3)} ms | "
             f"{r['snn_over_ann_sw_hw']:.2f} | {r['snn_over_ann_both_units']:.2f} |")
L += ["", "Paired, same sample and seed: SNN faster on n / N", "",
      "| | SNN (SNN unit) vs ANN (sw) | SNN (both units) vs ANN (MAC unit) |", "|---|---|---|"]
for t, r in out["feedforward"].items():
    a = " · ".join(f"{x}/{n}" for x, n in r["pairs"]["snn_hw_faster_than_ann_fast"])
    b = " · ".join(f"{x}/{n}" for x, n in r["pairs"]["snn_mac_faster_than_ann_mac"])
    L.append(f"| {t.upper()} | {a} | {b} |")
if out["streaming"]:
    S = out["streaming"]
    L += ["", "## Streaming ECG", "",
          f"GRU on the MAC unit: {S['n_gru_per_seed']} random beats per seed of the 1 000-beat subset "
          "(a GRU beat is ~0.5e8 cycles in simulation); GRU software latency from the board on the same beats.",
          "RSNN: its kernel does not use the MAC unit; on the both-units image it takes exactly the cycles of the",
          "SNN-unit board runs (cycle check below), so the board's RSNN hw cycles of the same beats are used for the pairing.", "",
          "| | GRU sw (board, same beats) | GRU + MAC unit | GRU speed-up | RSNN + SNN unit (200 beats/seed) | RSNN faster than GRU + MAC |",
          "|---|---|---|---|---|---|",
          f"| ECG stream | {f2(S['gru_fast_board_same_beats'], 1)} ms | {f2(S['gru_mac'], 1)} ms | {S['gru_speedup']:.2f}× | "
          f"{f2(S['rsnn_mac_200'], 2) if S['rsnn_mac_200'] else '-'} ms | "
          + " · ".join(f"{x}/{n}" for x, n in S["rsnn_faster_than_gru_mac"]) + " |"]
L += ["", "## Bit-exactness (outputs == PC int8 reference)", "", "| run | n |", "|---|---|"]
L += [f"| {k} | {a}/{n} |" for k, (a, n) in out["bit_exact"].items()]
L += ["", "## Cycle check against board runs of the same level and firmware", "",
      "Where the sim job had a board CSV of the same level: identical cycles / outputs per sample.",
      "Full-DS2 ECG ANN fast and MNIST fast CSVs are from firmware v3.0 / v3.1 (constant offset of a few",
      "cycles against v3.3, docs/snn_coprocessor_and_streaming.md section 5), so they match in outputs only.", "",
      "| run | n | same cycles | same outputs |", "|---|---|---|---|"]
L += [f"| {k} | {v['n']} | {v['same_cycles']} | {v['same_outputs']} |" for k, v in out["board_cycle_check"].items()]
L += ["", "## Full-set board references (results.json)", "",
      f"MNIST ANN fast {FF['MNIST/ann/fast']['lat_ms']['mean']:.2f} ms, SNN fast {FF['MNIST/snn/fast']['lat_ms']['mean']:.2f} ms, "
      f"SNN hw {FF['MNIST/snn/hw']['lat_ms']['mean']:.2f} ms; ECG ANN fast {FF['ECG/ann/fast']['lat_ms']['mean']:.2f} ms, "
      f"SNN fast {FF['ECG/snn/fast']['lat_ms']['mean']:.2f} ms, SNN hw {FF['ECG/snn/hw']['lat_ms']['mean']:.2f} ms; "
      f"stream GRU fast {SS['gru/fast']['lat_ms']['mean']:.0f} ms, RSNN fast {SS['rsnn/fast']['lat_ms']['mean']:.1f} ms, "
      f"RSNN hw {SS['rsnn/hw']['lat_ms']['mean']:.2f} ms."]
os.makedirs(os.path.join(RES, "mac_coproc"), exist_ok=True)
json.dump(out, open(os.path.join(RES, "mac_coproc", "summary_sim_rtl.json"), "w"), indent=1)
open(os.path.join(RES, "mac_coproc", "summary_sim_rtl.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
