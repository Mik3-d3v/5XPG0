#!/usr/bin/env python3
"""compute_results.py - recompute every number used in the final report
(Results / Hardware support / Benchmarking sections) from the curated board
runs in docs/results/.  Read-only on the inputs.

Run from anywhere:  python3 docs/report_analysis/compute_results.py
Writes (next to this script): results.json, results_summary.csv, results_summary.md

Variant meaning (verified in code/ann_vs_snn_mnist/pynq/README.md, firmware/main.c,
docs/*.md):
  baseline = straightforward RV32IM C kernels (dense ANN, byte-wise SNN row adds)
  fast     = same networks, optimised software (zero/delta skipping, 4xint8 SWAR,
             register blocking), stock bitstream
  hw       = fast firmware with the SNN kernel replaced by custom PCPI instructions
             (snn_pcpi.v coprocessor, spike_top_snn.bit); ANN code unchanged
  mac      = hw image + the ANN MAC coprocessor (mac_pcpi.v, spike_top_both.bit):
             ANN / GRU / direct-encoded SNN first layer on the MAC unit, SNN / RSNN
             neurons on the SNN unit (docs/mac_coprocessor.md, docs/results/mac_coproc/board/)
  muriscv  = the same int8 ANN through the unmodified muRISCV-NN (CMSIS-NN port)
             scalar C fully-connected kernels: external optimised ANN reference
Latency = PicoRV32 cycles between mailbox command and reply (timer_now() around
the inference call in firmware/main.c) / 100 MHz.
"""
import csv, gzip, json, os, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "docs", "results")
CLK = 100e6

def rel(p): return os.path.relpath(p, ROOT).replace("\\", "/")

def read_csv(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in ("label", "board_pred", "pc_int_pred", "pc_float_pred", "outputs_match",
              "cycles", "in_events", "in_changes", "hidden_events", "out_events", "host_ms"):
        if rows and k in rows[0]:
            cols[k] = np.array([float(r[k]) for r in rows])
    cols["idx"] = np.array([int(r["idx"]) for r in rows])
    return cols

def macro_f1(y, p, n_cls):
    f = []
    for c in range(n_cls):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        recl = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * prec * recl / (prec + recl) if prec + recl else 0.0)
    return float(np.mean(f))

def run_metrics(c, n_cls):
    y, p = c["label"], c["board_pred"]
    lat = c["cycles"] / CLK * 1e3
    return dict(n=int(len(y)), acc=float(np.mean(p == y)), f1=macro_f1(y, p, n_cls),
                f1_float=macro_f1(y, c["pc_float_pred"], n_cls),
                agree_int=float(np.mean(p == c["pc_int_pred"])),
                outputs_exact=float(np.mean(c["outputs_match"])),
                agree_float=float(np.mean(p == c["pc_float_pred"])),
                cycles=float(np.mean(c["cycles"])), lat_ms=float(np.mean(lat)),
                lat_p50=float(np.median(lat)), lat_p95=float(np.percentile(lat, 95)),
                lat_min=float(lat.min()), lat_max=float(lat.max()),
                thr=float(1e3 / np.mean(lat)),
                host_ms=float(np.mean(c["host_ms"])) if "host_ms" in c else None,
                in_events=float(np.mean(c["in_events"])), hidden_events=float(np.mean(c["hidden_events"])),
                out_events=float(np.mean(c["out_events"])))

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return dict(mean=float(np.mean(vals)), std=float(st.stdev(vals)) if len(vals) > 1 else 0.0, n=len(vals))

def summary_runs(path):
    if not os.path.exists(path): return {}
    d = json.load(open(path))
    return {(r["model"], r["seed"]): r for r in d["runs"]}

out = {"feedforward": {}, "streaming": {}, "T_study": {}, "vivado": {}, "paired": {}, "misc": {}}
rows_csv = []

def add_rows(ds, model, var, agg, srcs):
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            rows_csv.append([ds, model, var, k, v["mean"], v["std"], v["n"], ";".join(srcs)])

# ---------------------------------------------------------------- feedforward
TASKS = {"MNIST": ("ann_vs_snn", 10, [0, 1], ".csv"),
         "ECG": ("ann_vs_snn_ecg", 3, [0, 1, 2], ".csv.gz")}
NEURONS = {"MNIST": 128 + 10, "ECG": 64 + 3}          # LIF neurons with state (SNN)
raw = {}
for ds, (sub, ncls, seeds, ext) in TASKS.items():
    d = os.path.join(RES, sub)
    for model in ("ann", "snn"):
        for var in ("baseline", "fast", "hw", "muriscv"):
            per, srcs = [], []
            summ = summary_runs(os.path.join(d, f"summary_pynq_{var}.json"))
            for s in seeds:
                p = os.path.join(d, f"{model}_seed{s}_{var}_board_results{ext}")
                if not os.path.exists(p): continue
                c = read_csv(p); raw[(ds, model, var, s)] = c
                m = run_metrics(c, ncls)
                r = summ.get((model.upper(), s), {})
                m.update(ops_bench=r.get("ops_per_sample"), ops_exec=r.get("ops_executed"),
                         params=r.get("params"), model_bytes=r.get("model_bytes"))
                per.append(m); srcs.append(rel(p))
            if not per: continue
            agg = {k: ms([m[k] for m in per]) for k in per[0] if k != "n"}
            agg["n_samples"] = per[0]["n"]; agg["seeds"] = len(per)
            agg["state_bytes"] = 8 * NEURONS[ds] if model == "snn" else 0
            agg["sources"] = srcs
            out["feedforward"][f"{ds}/{model}/{var}"] = agg
            add_rows(ds, model, var, agg, srcs)

# MAC coprocessor level (board, docs/results/mac_coproc/board/)
MB = os.path.join(RES, "mac_coproc", "board")
for ds, (sub, ncls, seeds, ext) in TASKS.items():
    for model in ("ann", "snn"):
        per, srcs = [], []
        summ = summary_runs(os.path.join(MB, f"{ds.lower()}_summary_pynq_mac.json"))
        for s in seeds:
            p = os.path.join(MB, f"{model}_{ds.lower()}_seed{s}_board_results_mac.csv.gz")
            if not os.path.exists(p): continue
            c = read_csv(p); raw[(ds, model, "mac", s)] = c
            m = run_metrics(c, ncls)
            r = summ.get((model.upper(), s), {})
            m.update(ops_bench=r.get("ops_per_sample"), ops_exec=r.get("ops_executed"),
                     params=r.get("params"), model_bytes=r.get("model_bytes"))
            per.append(m); srcs.append(rel(p))
        if not per: continue
        agg = {k: ms([m[k] for m in per]) for k in per[0] if k != "n"}
        agg["n_samples"] = per[0]["n"]; agg["seeds"] = len(per)
        agg["state_bytes"] = 8 * NEURONS[ds] if model == "snn" else 0
        agg["sources"] = srcs
        out["feedforward"][f"{ds}/{model}/mac"] = agg
        add_rows(ds, model, "mac", agg, srcs)

# ANN on the coprocessor bitstream (hw image): regression run, seed 0
p_hw = os.path.join(RES, "snn_coprocessor/v33_regression/ann_ecg_seed0_v33_board_results_hw.csv.gz")
p_fa = os.path.join(RES, "snn_coprocessor/v33_regression/ann_ecg_seed0_v33_board_results_fast.csv.gz")
a, b = read_csv(p_hw), read_csv(p_fa)
n = min(len(a["cycles"]), len(b["cycles"]))
out["misc"]["ann_ecg_hw_vs_fast"] = dict(n=n, identical_cycles=int(np.sum(a["cycles"][:n] == b["cycles"][:n])),
    identical_pred=int(np.sum(a["board_pred"][:n] == b["board_pred"][:n])),
    lat_hw=float(np.mean(a["cycles"][:n]) / 1e5), lat_fast=float(np.mean(b["cycles"][:n]) / 1e5),
    sources=[rel(p_hw), rel(p_fa)])

# paired per-sample comparisons
def paired(ds, m1, v1, m2, v2, seeds):
    res = []
    for s in seeds:
        A, B = raw.get((ds, m1, v1, s)), raw.get((ds, m2, v2, s))
        if A is None or B is None: continue
        ia = {i: k for k, i in enumerate(A["idx"])}
        common = [i for i in B["idx"] if i in ia]
        ca = np.array([A["cycles"][ia[i]] for i in common])
        ib = {i: k for k, i in enumerate(B["idx"])}
        cb = np.array([B["cycles"][ib[i]] for i in common])
        res.append(dict(seed=s, n=len(common), frac_first_faster=float(np.mean(ca < cb)),
                        n_first_faster=int(np.sum(ca < cb)), ratio_mean=float(np.mean(ca) / np.mean(cb))))
    return res
for ds, (_, _, seeds, _) in TASKS.items():
    for (m1, v1, m2, v2) in [("snn", "fast", "ann", "fast"), ("snn", "hw", "ann", "fast"),
                             ("snn", "fast", "ann", "muriscv"), ("snn", "hw", "ann", "muriscv"),
                             ("ann", "fast", "ann", "muriscv"), ("ann", "muriscv", "ann", "baseline"),
                             ("snn", "mac", "ann", "mac"), ("ann", "mac", "ann", "fast")]:
        out["paired"][f"{ds}:{m1}_{v1} vs {m2}_{v2}"] = paired(ds, m1, v1, m2, v2, seeds)

# ---------------------------------------------------------------- T study (ECG SNN seed 0, first 2000 beats)
ts = {}
ann0 = raw[("ECG", "ann", "fast", 0)]
first = ann0["idx"] < 2000
out["T_study"]["ann_fast_first2000_ms"] = float(np.mean(ann0["cycles"][first]) / 1e5)
for lvl, folder in (("fast", "T_study"), ("hw", "T_study_hw")):
    for T in (1, 2, 5, 10, 20):
        if T == 10:
            c = raw[("ECG", "snn", lvl, 0)]; sel = c["idx"] < 2000
            src = out["feedforward"][f"ECG/snn/{lvl}"]["sources"][0]
            meta = json.load(open(os.path.join(RES, "ann_vs_snn_ecg/snn_seed0_meta.json")))
        else:
            p = os.path.join(RES, f"ann_vs_snn_ecg/{folder}/snn_seed0_T{T}_{lvl}_board_results.csv.gz")
            c = read_csv(p); sel = np.ones(len(c["idx"]), bool); src = rel(p)
            meta = json.load(open(os.path.join(RES, f"ann_vs_snn_ecg/T_study/snn_seed0_T{T}_meta.json")))
        ts.setdefault(T, {})[lvl] = dict(lat_ms=float(np.mean(c["cycles"][sel]) / 1e5), n=int(sel.sum()),
                                        exact=float(np.mean(c["outputs_match"][sel])), src=src)
        ts[T]["f1_int8_ds2"] = meta["pc_metrics_int"]["macro_f1"]
for lvl in ("fast", "hw"):
    Ts = np.array(sorted(ts)); L = np.array([ts[t][lvl]["lat_ms"] for t in Ts])
    k, b0 = np.polyfit(Ts, L, 1)
    r2 = 1 - np.sum((L - (k * Ts + b0)) ** 2) / np.sum((L - L.mean()) ** 2)
    out["T_study"][f"fit_{lvl}"] = dict(slope_ms_per_T=float(k), intercept_ms=float(b0), r2=float(r2),
        breakeven_T=float((out["T_study"]["ann_fast_first2000_ms"] - b0) / k))
out["T_study"]["points"] = {str(t): v for t, v in ts.items()}

# ---------------------------------------------------------------- streaming (GRU vs RSNN)
SD = os.path.join(RES, "ecg_streaming")
for model in ("gru", "rsnn"):
    for var, folder in (("baseline", "board_subset"), ("fast", "board_subset"), ("hw", "board_subset"),
                        ("hw_full", "board_full_hw"), ("mac", "../mac_coproc/board")):
        per, srcs = [], []
        summ = summary_runs(os.path.join(SD, folder if var == "mac" else "",
                                         f"summary_pynq_stream_{var.split('_')[0]}.json"))
        for s in (0, 1, 2):
            v = var.split("_")[0]
            p = os.path.join(SD, folder, f"{model}_ecg_seed{s}_board_results_mac.csv.gz" if var == "mac"
                             else f"{model}_seed{s}_{v}_board_results.csv.gz")
            if not os.path.exists(p): continue
            c = read_csv(p); m = run_metrics(c, 3)
            r = summ.get((model.upper(), s), {}) if var != "hw_full" else {}
            m.update(ops_exec=r.get("ops_executed"), params=r.get("params"), model_bytes=r.get("model_bytes"))
            per.append(m); srcs.append(rel(p))
        if not per: continue
        agg = {k: ms([m[k] for m in per]) for k in per[0] if k != "n"}
        agg["n_samples"] = per[0]["n"]; agg["seeds"] = len(per); agg["sources"] = srcs
        out["streaming"][f"{model}/{var}"] = agg
        add_rows("ECG-stream", model, var, agg, srcs)
p = os.path.join(SD, "board_subset/gru_seed0_hw_first50_board_results.csv.gz")
c = read_csv(p); out["streaming"]["gru/hw_first50"] = dict(n=len(c["idx"]), lat_ms=float(np.mean(c["cycles"]) / 1e5), src=rel(p))
ss = json.load(open(os.path.join(SD, "pc/stream_summary.json")))
out["streaming"]["pc_summary_keys"] = list(ss.keys()) if isinstance(ss, dict) else None
# full-DS2 int8 macro-F1 from the export metas (int8 == board, bit-exact)
for model in ("gru", "rsnn"):
    vals = []
    for s in (0, 1, 2):
        for folder in ("board_subset", "board_full_hw"):
            mp = os.path.join(SD, folder, f"{model}_seed{s}_meta.json")
            if os.path.exists(mp):
                mm = json.load(open(mp)); pi = mm.get("pc_metrics_int") or {}
                if "macro_f1" in pi: vals.append((s, pi["macro_f1"], mm.get("n_samples"))); break
    out["streaming"][f"{model}/pc_int8_meta"] = vals

# ---------------------------------------------------------------- Vivado
def parse_util(p):
    want = {"Slice LUTs": "lut", "LUT as Logic": "lut_logic", "LUT as Memory": "lut_mem", "Slice Registers": "ff",
            "Block RAM Tile": "bram", "DSPs": "dsp", "Slice": "slice"}
    r = {}
    for line in open(p):
        parts = [x.strip() for x in line.split("|")]
        if len(parts) > 3 and parts[1] in want and want[parts[1]] not in r:
            try: r[want[parts[1]]] = float(parts[2])
            except ValueError: pass
    return r
def parse_timing(p):
    L = open(p).read().splitlines()
    for i, l in enumerate(L):
        if "WNS(ns)" in l and "TNS(ns)" in l:
            v = L[i + 2].split(); return dict(wns=float(v[0]), whs=float(v[4]))
def parse_power(p):
    r = {}
    for l in open(p):
        parts = [x.strip() for x in l.split("|")]
        if len(parts) > 2:
            if parts[1] == "Total On-Chip Power (W)": r["total_w"] = float(parts[2])
            if parts[1] == "Dynamic (W)": r["dyn_w"] = float(parts[2])
            if parts[1] == "u_snn": r["u_snn_w"] = float(parts[2])
            if parts[1] == "u_mac": r["u_mac_w"] = float(parts[2])
            if parts[1] == "PS7" and "ps7_w" not in r: r["ps7_w"] = float(parts[2])
    return r
for k in ("vivado_reference", "vivado_snn"):
    d = os.path.join(RES, "snn_coprocessor", k)
    t = parse_timing(os.path.join(d, "spike_top_timing_summary_routed.rpt"))
    out["vivado"][k] = dict(**parse_util(os.path.join(d, "spike_top_utilization_placed.rpt")), **t,
                            **parse_power(os.path.join(d, "spike_top_power_routed.rpt")),
                            fmax_est_mhz=1e3 / (10.0 - t["wns"]), src=rel(d))
# ANN MAC coprocessor (docs/mac_coprocessor.md): MAC only, SNN + MAC
for k in ("vivado_mac", "vivado_both"):
    d = os.path.join(RES, "mac_coproc", k)
    if not os.path.isdir(d): continue
    t = parse_timing(os.path.join(d, "spike_top_timing_summary_routed.rpt"))
    out["vivado"][k] = dict(**parse_util(os.path.join(d, "spike_top_utilization_placed.rpt")), **t,
                            **parse_power(os.path.join(d, "spike_top_power_routed.rpt")),
                            fmax_est_mhz=1e3 / (10.0 - t["wns"]), src=rel(d))
# MAC coprocessor latencies: cycle-exact RTL simulation on random subsets
# (mac_coproc_summary.py; not board runs)
p_mac = os.path.join(RES, "mac_coproc", "summary_sim_rtl.json")
if os.path.exists(p_mac):
    out["mac_coproc_sim_rtl"] = dict(json.load(open(p_mac)), src=rel(p_mac))

# ---------------------------------------------------------------- firmware image sizes
fw = os.path.join(ROOT, "code/ann_vs_snn_mnist/pynq/firmware")
out["misc"]["fw_bin_bytes"] = {f: os.path.getsize(os.path.join(fw, f)) for f in sorted(os.listdir(fw)) if f.endswith(".bin")}

# ---------------------------------------------------------------- derived ratios (latency means)
def lat(k, src=None):
    return (src or out["feedforward"])[k]["lat_ms"]["mean"]
D = {}
for ds in ("MNIST", "ECG"):
    D[f"{ds} SNN/ANN fast"] = lat(f"{ds}/snn/fast") / lat(f"{ds}/ann/fast")
    D[f"{ds} SNN/ANN baseline"] = lat(f"{ds}/snn/baseline") / lat(f"{ds}/ann/baseline")
    D[f"{ds} SNN hw / ANN fast"] = lat(f"{ds}/snn/hw") / lat(f"{ds}/ann/fast")
    D[f"{ds} SNN speed-up fast->hw"] = lat(f"{ds}/snn/fast") / lat(f"{ds}/snn/hw")
    D[f"{ds} SNN speed-up baseline->hw"] = lat(f"{ds}/snn/baseline") / lat(f"{ds}/snn/hw")
    D[f"{ds} SNN speed-up baseline->fast"] = lat(f"{ds}/snn/baseline") / lat(f"{ds}/snn/fast")
    D[f"{ds} ANN speed-up baseline->fast"] = lat(f"{ds}/ann/baseline") / lat(f"{ds}/ann/fast")
    D[f"{ds} muRISCV / ANN fast"] = lat(f"{ds}/ann/muriscv") / lat(f"{ds}/ann/fast")
    D[f"{ds} ANN baseline / muRISCV"] = lat(f"{ds}/ann/baseline") / lat(f"{ds}/ann/muriscv")
    D[f"{ds} muRISCV / SNN fast"] = lat(f"{ds}/ann/muriscv") / lat(f"{ds}/snn/fast")
    D[f"{ds} muRISCV / SNN hw"] = lat(f"{ds}/ann/muriscv") / lat(f"{ds}/snn/hw")
    D[f"{ds} F1 SNN-ANN (fast)"] = out["feedforward"][f"{ds}/snn/fast"]["f1"]["mean"] - out["feedforward"][f"{ds}/ann/fast"]["f1"]["mean"]
S_ = out["streaming"]
D["stream GRU fast / RSNN fast"] = S_["gru/fast"]["lat_ms"]["mean"] / S_["rsnn/fast"]["lat_ms"]["mean"]
D["stream GRU fast / RSNN hw"] = S_["gru/fast"]["lat_ms"]["mean"] / S_["rsnn/hw"]["lat_ms"]["mean"]
D["stream GRU baseline / RSNN baseline"] = S_["gru/baseline"]["lat_ms"]["mean"] / S_["rsnn/baseline"]["lat_ms"]["mean"]
D["stream RSNN speed-up fast->hw"] = S_["rsnn/fast"]["lat_ms"]["mean"] / S_["rsnn/hw"]["lat_ms"]["mean"]
D["stream RSNN hw_full / ECG ANN fast"] = S_["rsnn/hw_full"]["lat_ms"]["mean"] / lat("ECG/ann/fast")
D["stream params GRU/RSNN"] = S_["gru/fast"]["params"]["mean"] / S_["rsnn/fast"]["params"]["mean"]
D["stream ops GRU/RSNN"] = S_["gru/fast"]["ops_exec"]["mean"] / S_["rsnn/fast"]["ops_exec"]["mean"]
D["T fast per-step cycles"] = out["T_study"]["fit_fast"]["slope_ms_per_T"] * 1e5
D["T hw per-step cycles"] = out["T_study"]["fit_hw"]["slope_ms_per_T"] * 1e5
D["T per-step reduction"] = out["T_study"]["fit_fast"]["slope_ms_per_T"] / out["T_study"]["fit_hw"]["slope_ms_per_T"]
vr, vs = out["vivado"]["vivado_reference"], out["vivado"]["vivado_snn"]
for k in ("lut", "lut_logic", "lut_mem", "ff", "slice", "bram", "dsp"):
    D[f"vivado delta {k}"] = vs[k] - vr[k]
D["vivado delta total power W"] = vs["total_w"] - vr["total_w"]
for kk in ("vivado_mac", "vivado_both"):
    if kk in out["vivado"]:
        for k in ("lut", "lut_mem", "ff", "dsp", "total_w"):
            D[f"{kk} delta {k} vs reference"] = out["vivado"][kk][k] - vr[k]
for ds in ("MNIST", "ECG"):
    if f"{ds}/ann/mac" in out["feedforward"]:
        D[f"{ds} ANN speed-up fast->mac"] = lat(f"{ds}/ann/fast") / lat(f"{ds}/ann/mac")
        D[f"{ds} SNN mac / ANN mac"] = lat(f"{ds}/snn/mac") / lat(f"{ds}/ann/mac")
        D[f"{ds} SNN speed-up hw->mac"] = lat(f"{ds}/snn/hw") / lat(f"{ds}/snn/mac")
if "gru/mac" in S_:
    D["stream GRU speed-up fast->mac"] = S_["gru/fast"]["lat_ms"]["mean"] / S_["gru/mac"]["lat_ms"]["mean"]
    D["stream GRU mac / RSNN mac"] = S_["gru/mac"]["lat_ms"]["mean"] / S_["rsnn/mac"]["lat_ms"]["mean"]
if "mac_coproc_sim_rtl" in out:
    M_ = out["mac_coproc_sim_rtl"]
    for t_, r_ in M_["feedforward"].items():
        D[f"{t_.upper()} (RTL sim subset) ANN speed-up fast->mac"] = r_["ann_speedup"]
        D[f"{t_.upper()} (RTL sim subset) SNN/ANN, both units"] = r_["snn_over_ann_both_units"]
        D[f"{t_.upper()} (RTL sim subset) SNN/ANN, SNN hw vs ANN fast"] = r_["snn_over_ann_sw_hw"]
    if M_.get("streaming"):
        D["stream (RTL sim subset) GRU speed-up fast->mac"] = M_["streaming"]["gru_speedup"]
        D["stream GRU mac / RSNN hw"] = M_["streaming"]["gru_mac"][0] / S_["rsnn/hw"]["lat_ms"]["mean"]
# same-image baseline vs fast ratios for MNIST (baseline ran on the first 1000 images only)
for model in ("ann", "snn"):
    r = []
    for s_ in (0, 1):
        B_, F_ = raw[("MNIST", model, "baseline", s_)], raw[("MNIST", model, "fast", s_)]
        sel = np.isin(F_["idx"], B_["idx"])
        r.append(np.mean(B_["cycles"]) / np.mean(F_["cycles"][sel]))
    D[f"MNIST {model.upper()} speed-up baseline->fast (same 1000 images)"] = float(np.mean(r))
out["derived"] = D

json.dump(out, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(HERE, "results_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["dataset", "model", "variant", "metric", "mean", "std", "n_seeds", "source_files"])
    w.writerows(rows_csv)
print("ok", len(rows_csv), "rows")

# ---------------------------------------------------------------- markdown summary
def f(v, k, d=3):
    x = v.get(k)
    if not isinstance(x, dict): return "-"
    return f"{x['mean']:.{d}f} ± {x['std']:.{d}f}"
L = ["# Report numbers (generated by compute_results.py - do not edit by hand)", "",
     "Latency = PicoRV32 cycles of the timed inference call / 100 MHz. mean ± std over seeds (std of per-seed means).", "",
     "## Feedforward (board, per-sample CSVs)", "",
     "| dataset/model/variant | seeds | samples | macro-F1 | accuracy | board==PC int8 pred | board==PC float pred | latency [ms] | cycles | throughput [1/s] | ops (benchmark def.) | ops executed | model [B] | SNN state [B] | sources |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in out["feedforward"].items():
    L.append(f"| {k} | {v['seeds']} | {v['n_samples']} | {f(v,'f1',4)} | {f(v,'acc',4)} | {f(v,'agree_int',4)} | {f(v,'agree_float',4)} | {f(v,'lat_ms',3)} | {f(v,'cycles',0)} | {f(v,'thr',1)} | {f(v,'ops_bench',0)} | {f(v,'ops_exec',0)} | {f(v,'model_bytes',0)} | {v['state_bytes']} | {'<br>'.join(v['sources'])} |")
L += ["", "## Streaming ECG (GRU vs RSNN)", "",
      "| model/variant | seeds | beats | macro-F1 (on these beats) | latency [ms] | ops executed | params | model [B] | sources |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in out["streaming"].items():
    if isinstance(v, dict) and isinstance(v.get("lat_ms"), dict):
        L.append(f"| {k} | {v['seeds']} | {v['n_samples']} | {f(v,'f1',4)} | {f(v,'lat_ms',3)} | {f(v,'ops_exec',0)} | {f(v,'params',0)} | {f(v,'model_bytes',0)} | {'<br>'.join(v['sources'])} |")
L += ["", "Note: GRU/RSNN baseline rows are the first 200 beats of the sorted subset (almost only N): latency only, F1 meaningless.", "",
      "## T study (ECG SNN seed 0, first 2000 DS2 beats)", "", "```", json.dumps(out["T_study"], indent=1), "```", "",
      "## Vivado (routed, 100 MHz clk_fpga_0)", "", "```", json.dumps(out["vivado"], indent=1), "```",
      "fmax_est_mhz = 1000/(10 ns - WNS): an estimate from slack, not a timing closure at that frequency.", "",
      "## Paired per-sample comparisons (n_first_faster / n, mean cycle ratio first/second)", "", "```", json.dumps(out["paired"], indent=1), "```", "",
      "## ANN MAC coprocessor (cycle-exact RTL simulation subsets, docs/mac_coprocessor.md)", "", "```",
      json.dumps(out.get("mac_coproc_sim_rtl", {}), indent=1), "```", "",
      "## Misc", "", "```", json.dumps(out["misc"], indent=1), "```", "", "## Derived ratios", "", "```", json.dumps(out["derived"], indent=1), "```"]
open(os.path.join(HERE, "results_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("md ok")
