#!/usr/bin/env python3
"""make_report_figures.py - report figures in plain language for the 5XPG0 final report.

Reads   docs/report_analysis/results.json           (written by compute_results.py)
        docs/results/snn_coprocessor/vivado_*/spike_top_power_routed.rpt
Writes  <out_dir>/fig*.pdf (report) and fig*.png (preview / web), energy_estimate.csv

Usage:  python3 make_report_figures.py <repo_root> <out_dir>

Labels avoid simulator settings on purpose: implementation levels are called
"plain C", "optimized C", "optimized C + coprocessor" and "optimized ANN library".
"""
import csv, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "report_analysis", "figures_report"))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "docs", "report_analysis", "results.json")))
FF, ST, TS, VIV = R["feedforward"], R["streaming"], R["T_study"], R["vivado"]

# ---- style: IEEE journal, black and white, patterns instead of colour ----
# Every figure is exactly W = 3.5 in wide (one IEEE column) and is saved WITHOUT a tight bounding box,
# so \includegraphics[width=\columnwidth] does not rescale it and the text stays at its nominal size.
# Text follows the IEEE graphics guideline (about 8 pt at final size). Fonts are TrueType (Type 42):
# Times New Roman on Windows, Liberation Serif (metric clone of Times) elsewhere; never Type 3.
K, G = "black", "0.45"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "Times",
                                           "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "serif", "mathtext.it": "serif:italic", "mathtext.bf": "serif:bold", "mathtext.cal": "serif",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "text.color": K, "axes.edgecolor": K, "axes.labelcolor": K, "xtick.color": K, "ytick.color": K,
    "axes.linewidth": 0.6, "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "grid.color": "0.8", "grid.linestyle": ":", "grid.linewidth": 0.5,
    "legend.frameon": True, "legend.edgecolor": K, "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.35, "legend.handlelength": 1.8, "legend.labelspacing": 0.3,
    "hatch.linewidth": 0.5, "hatch.color": K, "lines.linewidth": 1.0,
    "errorbar.capsize": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
# one pattern per entity, used in every figure
ANN = dict(facecolor="white", edgecolor=K, hatch="////", linewidth=0.6)      # ANN family: diagonal
SNN = dict(facecolor="0.82", edgecolor=K, hatch="xxx", linewidth=0.6)        # SNN family: cross-hatch on grey
SEG = {"ps": dict(facecolor="white", edgecolor=K, linewidth=0.6),
       "static": dict(facecolor="white", edgecolor=K, hatch="....", linewidth=0.6),
       "bram": dict(facecolor="white", edgecolor=K, hatch="////", linewidth=0.6),
       "logic": dict(facecolor=K, edgecolor=K, linewidth=0.6),
       "arith": dict(facecolor=K, edgecolor=K, linewidth=0.6),
       "weights": dict(facecolor="white", edgecolor=K, hatch="////", linewidth=0.6),
       "state": dict(facecolor="0.82", edgecolor=K, hatch="....", linewidth=0.6),
       "ref": dict(facecolor="white", edgecolor=K, linewidth=0.6),
       "cop": dict(facecolor="0.82", edgecolor=K, hatch="xxx", linewidth=0.6)}
LBL = dict(fontsize=7.5, color=K)
BOX = dict(fc="white", ec="none", pad=0.3)
TARGET = dict(color=K, lw=0.7, ls="--")
W = 3.5  # IEEE single column [in]


def save(fig, name):
    for lg in [a.get_legend() for a in fig.axes] + list(fig.legends):
        if lg is not None:
            lg.get_frame().set_linewidth(0.5)
    w, h = fig.get_size_inches()
    assert abs(w - W) < 1e-6, f"{name}: figure must be {W} in wide, is {w}"
    fig.savefig(os.path.join(OUT, name + ".pdf"))          # no bbox_inches="tight": width stays 3.5 in
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


def m(key, metric="lat_ms", src=FF):
    return src[key][metric]["mean"]


def s(key, metric="lat_ms", src=FF):
    return src[key][metric]["std"]


def fmt_ms(v):
    return f"{v:,.0f} ms" if v >= 100 else (f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms")


# =============================================================================
# helpers
# =============================================================================
def finish(ax, xgrid=True, ygrid=False):
    ax.grid(False)
    if xgrid: ax.xaxis.grid(True, which="major")
    if ygrid: ax.yaxis.grid(True, which="major")
    ax.set_axisbelow(True)


def hbar(ax, y, v, style, h, left=0.0):
    ax.barh(y, v, height=h, left=left, **style)


def vlabel(ax, x, y, txt, ha="left"):
    ax.text(x, y, txt, va="center", ha=ha, bbox=BOX, zorder=5, **LBL)


# =============================================================================
# Fig 1 - latency per input, every implementation level (MNIST, ECG beat)
# =============================================================================
# Latency targets from Table I (updated): MNIST <= 40 ms/image, ECG <= 200 ms/beat.
TGT_MNIST, TGT_ECG = 40, 200
# "+ coprocessor" = matched custom units: ANN on the MAC unit, SNN on the SNN unit (ECG SNN: first
# layer on the MAC unit, timesteps on the SNN unit; MNIST SNN is rate-coded and has no MAC layer).
LEVELS = [("baseline", "plain C"), ("muriscv", "optimized ANN\nlibrary"),
          ("fast", "optimized C"), ("mac", "optimized C\n+ coprocessor")]
fig, axes = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True)
for ax, ds, tgt, title in ((axes[0], "MNIST", TGT_MNIST, "(a) MNIST, dense image input"),
                           (axes[1], "ECG", TGT_ECG, "(b) ECG, one heartbeat per input")):
    ylabels, y = [], 0
    for lvl, lab in LEVELS:
        bars = [(mod, st) for mod, st in (("ann", ANN), ("snn", SNN)) if f"{ds}/{mod}/{lvl}" in FF]
        h = 0.36
        offs = [0] if len(bars) == 1 else [-h / 2 - 0.02, h / 2 + 0.02]
        for (mod, st), o in zip(bars, offs):
            v = m(f"{ds}/{mod}/{lvl}")
            hbar(ax, y + o, v, st, h)
            vlabel(ax, v * 1.1, y + o, fmt_ms(v))
        ylabels.append((y, lab)); y += 1
    ax.axvline(tgt, **TARGET)
    ax.set_yticks([p for p, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels])
    for tl in ax.get_yticklabels():
        tl.set_multialignment("right"); tl.set_linespacing(0.95)
    ax.tick_params(axis="y", which="both", length=0)
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(1, 3000); ax.set_ylim(len(LEVELS) - 0.45, -0.55)
    ax.set_title(title, loc="left", pad=3)
    finish(ax)
axes[1].set_xlabel("Latency per input on PicoRV32 at 100 MHz (ms)")
axes[1].legend(handles=[Patch(**ANN, label="ANN"), Patch(**SNN, label="SNN"),
                        Line2D([], [], **TARGET, label="Latency target")],
               loc="lower right", ncol=1, handleheight=1.0)
fig.tight_layout(h_pad=0.6)
save(fig, "fig1_latency_overview")

# =============================================================================
# Fig 2 - ECG: accuracy vs latency (beat-wise and streaming networks)
# =============================================================================
rs_f1 = ST["rsnn/hw_full"]["f1"]
GRU_F1 = 0.667  # int8 GRU on all test beats (PC; docs/snn_coprocessor_and_streaming.md, Sec. 3.3)
# family -> marker; hollow = optimized C, filled = + coprocessor; arrow = effect of the custom unit
FAMS = [("ANN (one beat)", "o", (m("ECG/ann/fast"), m("ECG/ann/fast", "f1"), s("ECG/ann/fast", "f1")),
                              (m("ECG/ann/mac"), m("ECG/ann/mac", "f1"), s("ECG/ann/mac", "f1"))),
        ("SNN (one beat)", "s", (m("ECG/snn/fast"), m("ECG/snn/fast", "f1"), s("ECG/snn/fast", "f1")),
                              (m("ECG/snn/mac"), m("ECG/snn/mac", "f1"), s("ECG/snn/mac", "f1"))),
        ("GRU (stream)", "^", (m("gru/fast", src=ST), GRU_F1, 0.0), (m("gru/mac", src=ST), GRU_F1, 0.0)),
        ("Recurrent SNN (stream)", "v", (m("rsnn/fast", src=ST), rs_f1["mean"], rs_f1["std"]),
                                      (m("rsnn/hw_full", src=ST), rs_f1["mean"], rs_f1["std"]))]
fig, ax = plt.subplots(figsize=(W, 2.5))
for lab, mk, sw, hw in FAMS:
    ax.annotate("", xy=(hw[0] * 1.07, hw[1]), xytext=(sw[0] / 1.07, sw[1]),
                arrowprops=dict(arrowstyle="-|>", color=K, lw=0.6, mutation_scale=7, shrinkA=3, shrinkB=3))
    for (x, yv, e), mfc in ((sw, "white"), (hw, K)):
        ax.errorbar(x, yv, yerr=e, ls="", marker=mk, ms=5.5, color=K, mfc=mfc, mec=K, mew=0.9,
                    elinewidth=0.6, zorder=3)
ax.axvline(TGT_ECG, **TARGET)
ax.set_xscale("log"); ax.set_xlim(1.0, 3000); ax.set_ylim(0.645, 0.725)
ax.set_xlabel("Latency per heartbeat (ms)")
ax.set_ylabel("Macro-F1 on unseen patients")
finish(ax, True, True)
fam_h = [Line2D([], [], ls="", marker=mk, ms=5, color=K, mfc="white", mec=K, label=lab) for lab, mk, _, _ in FAMS]
lvl_h = [Line2D([], [], ls="", marker="o", ms=5, color=K, mfc="white", mec=K, label="optimized C"),
         Line2D([], [], ls="", marker="o", ms=5, color=K, mfc=K, mec=K, label="+ coprocessor"),
         Line2D([], [], **TARGET, label="200 ms target")]
leg1 = ax.legend(handles=fam_h, loc="upper right", fontsize=7)
ax.add_artist(leg1)
ax.legend(handles=lvl_h, loc="lower right", fontsize=7)
fig.tight_layout(pad=0.3)
save(fig, "fig2_ecg_accuracy_vs_latency")

# =============================================================================
# Fig 3 - cost of SNN timesteps (ECG SNN at different T)
# =============================================================================
Ts = sorted(int(t) for t in TS["points"])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.8), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
for lvl, ls, mk, mfc, lab in (("fast", "--", "s", "white", "SNN, optimized C"),
                              ("hw", "-", "D", K, "SNN + coprocessor")):
    L = [TS["points"][str(t)][lvl]["lat_ms"] for t in Ts]
    a1.plot(Ts, L, ls, color=K, lw=1.0, marker=mk, ms=4.5, mfc=mfc, mec=K, mew=0.8, label=lab)
    f = TS[f"fit_{lvl}"]
    a1.annotate(f"+{f['slope_ms_per_T'] * 1000:.0f} µs per timestep", (Ts[-1], L[-1]), xytext=(-4, 5),
                textcoords="offset points", ha="right", **LBL)
a1.axhline(TS["ann_fast_first2000_ms"], color=K, lw=1.0, ls=":", label="ANN, optimized C")
a1.set_ylabel("Latency per beat (ms)"); a1.set_ylim(2.8, 13.5)
a1.legend(loc="upper left")
finish(a1, False, True)
F = [TS["points"][str(t)]["f1_int8_ds2"] for t in Ts]
a2.plot(Ts, F, "-", color=K, lw=1.0, marker="o", ms=4, mfc=K, mec=K)
for t, v in zip(Ts, F):
    if t == 1:   # steep rise to the right: keep this label below-right of the point
        off, ha, va = (3, -3), "left", "center"
    else:        # flat part of the curve: centred, just above the marker
        off, ha, va = (0, 3), "center", "bottom"
    a2.annotate(f"{v:.2f}", (t, v), xytext=off, textcoords="offset points",
                ha=ha, va=va, **LBL)
a2.set_ylabel("Macro-F1"); a2.set_ylim(0.25, 0.85)
a2.set_xlabel("SNN timesteps T per heartbeat")
a2.set_xticks(Ts); finish(a2, False, True)
fig.tight_layout(h_pad=0.4)
save(fig, "fig3_timestep_cost")

# =============================================================================
# Fig 4 - streaming ECG: GRU vs recurrent SNN (work and latency per heartbeat)
# Work is counted honestly: GRU = multiply-accumulates; recurrent SNN = synaptic additions PLUS the
# per-sample neuron-state updates (every neuron leaks, integrates and is thresholded at every sample).
# Speed-up ratios are not printed in the figure: they belong in the caption/text, split into the
# software-vs-software factor and the additional coprocessor factor.
# =============================================================================
N_SAMPLES_PER_BEAT, H_RSNN = 126, 64
import csv as _csv
OVN = {(r["job"], r["model"], r["hidden"], r["T"], r["firmware"]): r for r in
       _csv.DictReader(open(os.path.join(ROOT, "docs", "results", "overnight_2026-10-12", "summary.csv")))}
g16 = OVN[("C", "GRU", "16", "", "fast")]
gru16_ms, gru16_f1 = float(g16["board_ms_mean"]), float(g16["pc_int8_mean"])
gru_ops = m("gru/fast", "ops_exec", ST)
gru16_ops = N_SAMPLES_PER_BEAT * 3 * (1 + 16) * 16 + (gru_ops - N_SAMPLES_PER_BEAT * 3 * (1 + 64) * 64) * 16 / 64
rsnn_add = m("rsnn/hw", "ops_exec", ST)
rsnn_upd = N_SAMPLES_PER_BEAT * H_RSNN
lat = [("GRU-64,\noptimized C", m("gru/fast", src=ST), ANN),
       ("GRU-64\n+ coprocessor", m("gru/mac", src=ST), ANN),
       ("GRU-16,\noptimized C", gru16_ms, ANN),
       ("Rec. SNN-64,\noptimized C", m("rsnn/fast", src=ST), SNN),
       ("Rec. SNN-64\n+ coprocessor", m("rsnn/hw_full", src=ST), SNN)]
fig, (b1, b2) = plt.subplots(2, 1, figsize=(W, 3.5), gridspec_kw={"height_ratios": [2.3, 4.2]})
for i, (v, lab) in enumerate(((gru_ops, "GRU-64"), (gru16_ops, "GRU-16"))):
    hbar(b1, i, v, ANN, 0.55); vlabel(b1, v * 1.15, i, f"{v / 1e6:.2f} M")
hbar(b1, 2, rsnn_add, SNN, 0.55)
hbar(b1, 2, rsnn_upd, SEG["state"], 0.55, left=rsnn_add)
vlabel(b1, (rsnn_add + rsnn_upd) * 1.15, 2, f"{(rsnn_add + rsnn_upd) / 1e3:.1f} k")
b1.set_yticks([0, 1, 2]); b1.set_yticklabels(["GRU-64", "GRU-16", "Rec. SNN-64"]); b1.set_ylim(2.6, -0.6)
b1.set_xscale("log"); b1.set_xlim(1e2, 1e8)
b1.set_xlabel("(a) Arithmetic operations per heartbeat")
for i, (lab, v, st) in enumerate(lat):
    hbar(b2, i, v, st, 0.55); vlabel(b2, v * 1.15, i, fmt_ms(v))
b2.set_yticks(range(len(lat))); b2.set_yticklabels([l for l, _, _ in lat]); b2.set_ylim(len(lat) - 0.4, -0.6)
for tl in b2.get_yticklabels():
    tl.set_multialignment("right"); tl.set_linespacing(0.95)
b2.set_xscale("log"); b2.set_xlim(1, 1e5)
b2.axvline(TGT_ECG, **TARGET)
b2.set_xlabel("(b) Latency per heartbeat (ms)")
b2.legend(handles=[Patch(**ANN, label="GRU: multiply-accumulates"), Patch(**SNN, label="SNN: synaptic additions"),
                   Patch(**SEG["state"], label="SNN: neuron-state updates")],
          loc="lower right", handleheight=1.0, fontsize=7)
for b in (b1, b2):
    b.tick_params(axis="y", which="both", length=0); finish(b)
fig.tight_layout(h_pad=0.6, pad=0.3)
save(fig, "fig4_streaming_gru_vs_rsnn")
print(f"fig4: GRU16 {gru16_ms:.1f} ms F1 {gru16_f1:.3f} ops {gru16_ops:.0f}; work GRU64/RSNN {gru_ops / (rsnn_add + rsnn_upd):.0f}x, "
      f"GRU16/RSNN {gru16_ops / (rsnn_add + rsnn_upd):.1f}x; latency SW: GRU64/RSNN {lat[0][1] / lat[3][1]:.1f}x, "
      f"GRU16/RSNN {lat[2][1] / lat[3][1]:.1f}x; HW: GRU64mac/RSNNcop {lat[1][1] / lat[4][1]:.0f}x, GRU16sw/RSNNcop {lat[2][1] / lat[4][1]:.0f}x")


# =============================================================================
# Fig 5 - FPGA power (Vivado estimate): whole chip and programmable-logic breakdown
# =============================================================================
def power_rows(path):
    rows, on = {}, False
    for line in open(path, encoding="utf8", errors="ignore"):
        if "On-Chip Components" in line and "1.1" in line:
            on = True; continue
        if on and line.startswith("| ") and not line.startswith("|   ") and "Power (W)" not in line and "Static Power" not in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            try:
                rows[parts[0]] = float(parts[1])
            except (ValueError, IndexError):
                pass
            if parts[0] == "Total":
                break
    stat = re.search(r"\|\s*Device Static \(W\)\s*\|\s*([\d.]+)", open(path, encoding="utf8", errors="ignore").read())
    rows["Device Static"] = float(stat.group(1))
    return rows


P = {k: power_rows(os.path.join(ROOT, "docs", "results", "snn_coprocessor", d, "spike_top_power_routed.rpt"))
     for k, d in (("ref", "vivado_reference"), ("snn", "vivado_snn"))}


def fabric(rows):
    blk = rows.get("Block RAM", 0)
    logic = sum(rows.get(k, 0) for k in ("Slice Logic", "Signals", "Clocks", "DSPs"))
    other = rows.get("I/O", 0)
    return blk, logic, other


# Single panel: the Arm processing system (PS7, ~1.26 W) and device static power (~0.15 W) are
# identical in both designs and are quoted in the caption instead of being drawn as a wide, empty bar.
designs = [("ref", "PicoRV32 SoC"), ("snn", "PicoRV32 SoC + coprocessor")]
comp = [("Block RAM", "Block RAM"), ("Signals (wiring)", "Signals"),
        ("Clocks", "Clocks"), ("Logic (LUTs, FFs)", "Slice Logic"), ("DSP slices", "DSPs")]
fig, c2 = plt.subplots(figsize=(W, 2.1))
y = np.arange(len(comp)); h = 0.36
for j, (k, lab) in enumerate(designs):
    st = SEG["ref"] if j == 0 else SEG["cop"]
    v = [P[k].get(key, 0) * 1000 for _, key in comp]
    for yy, vv in zip(y, v):
        yo = yy + (j - 0.5) * (h + 0.04)
        hbar(c2, yo, vv, st, h); vlabel(c2, vv + 4, yo, f"{vv:.0f}")
c2.set_yticks(y); c2.set_yticklabels([l for l, _ in comp]); c2.set_ylim(len(comp) - 0.4, -0.6)
c2.set_xlim(0, 300); c2.set_xlabel("Estimated dynamic power (mW)")
c2.legend(handles=[Patch(**SEG["ref"], label=designs[0][1]), Patch(**SEG["cop"], label=designs[1][1])],
          loc="lower right", handleheight=1.0, fontsize=7)
c2.tick_params(axis="y", which="both", length=0); finish(c2)
fig.tight_layout(pad=0.3)
save(fig, "fig5_fpga_power")
for k, lab in designs:
    r = P[k]
    print(f"fig5 {lab}: total {r['Total']:.3f} W, PS7 {r['PS7']:.3f} W, static {r['Device Static']:.3f} W, "
          f"PL dynamic {(r['Total'] - r['PS7'] - r['Device Static']) * 1000:.0f} mW, BRAM {r['Block RAM'] * 1000:.0f} mW")

# =============================================================================
# Fig 6 - energy per inference: (a) this FPGA, (b) a dedicated 45 nm chip (estimate)
# =============================================================================
# 45 nm, 0.9 V energies after M. Horowitz, ISSCC 2014 [pJ]
E_MUL8, E_ADD32 = 0.2, 0.1
E_MAC = E_MUL8 + E_ADD32           # int8 multiply + 32-bit accumulate
E_SOP = E_ADD32                    # synaptic operation: accumulate only
E_NEURON_ARITH = 0.5               # leak multiply, add, compare, reset (approx.)


def sram64(kb):                    # 64-bit SRAM read: 8 kB 10 pJ, 32 kB 20 pJ, 1 MB 100 pJ (log interp.)
    pts_ = [(8, 10.0), (32, 20.0), (1024, 100.0)]
    if kb <= 8: return 10.0
    for (k0, e0), (k1, e1) in zip(pts_, pts_[1:]):
        if kb <= k1:
            t = math.log(kb / k0) / math.log(k1 / k0)
            return e0 * (e1 / e0) ** t
    return 100.0


def e_byte(model_bytes):           # weights streamed in 64-bit words -> cost per weight byte
    return sram64(model_bytes / 1024) / 8


E_STATE = 2 * sram64(1) / 2        # read + write one 32-bit membrane potential in a small SRAM (half a 64-bit access each)

work = []  # (label, colour, latency_ms, fabric_W, MACs, SOPs, neuron_updates, model_bytes)
mn_in = m("MNIST/snn/fast", "in_events"); mn_h = m("MNIST/snn/fast", "hidden_events")
work.append(("MNIST  ANN", "ann", m("MNIST/ann/fast"), "ref", m("MNIST/ann/fast", "ops_exec"), 0, 0, m("MNIST/ann/fast", "model_bytes")))
work.append(("MNIST  SNN", "snn", m("MNIST/snn/hw"), "snn", 0, m("MNIST/snn/hw", "ops_exec"), (128 + 10) * 25, m("MNIST/snn/hw", "model_bytes")))
ecg_first = m("ECG/snn/hw", "in_events") * 64
work.append(("ECG beat  ANN", "ann", m("ECG/ann/fast"), "ref", m("ECG/ann/fast", "ops_exec"), 0, 0, m("ECG/ann/fast", "model_bytes")))
work.append(("ECG beat  SNN", "snn", m("ECG/snn/hw"), "snn", ecg_first, m("ECG/snn/hw", "ops_exec") - ecg_first,
             (64 + 3) * 10, m("ECG/snn/hw", "model_bytes")))
work.append(("ECG stream  GRU", "ann", m("gru/fast", src=ST), "ref", m("gru/fast", "ops_exec", ST), 0, 0, m("gru/fast", "model_bytes", ST)))
work.append(("ECG stream  rec. SNN", "snn", m("rsnn/hw_full", src=ST), "snn", 0, m("rsnn/hw", "ops_exec", ST), 126 * 64,
             m("rsnn/hw", "model_bytes", ST)))

rows_out = []
for lab, col, lat_ms, d, macs, sops, upd, mb in work:
    p_fab = P[d]["Total"] - P[d]["PS7"] - P[d]["Device Static"] if "Total" in P[d] else None
    if p_fab is None or p_fab <= 0:
        p_fab = (VIV["vivado_" + ("reference" if d == "ref" else "snn")]["dyn_w"] - VIV["vivado_reference"]["ps7_w"])
    e_fpga_uJ = p_fab * lat_ms * 1e3          # W * ms = mJ -> *1e3 = uJ
    eb = e_byte(mb)
    e_arith = macs * E_MAC + sops * E_SOP + upd * E_NEURON_ARITH
    e_wmem = (macs + sops) * eb
    e_state = upd * E_STATE
    rows_out.append(dict(workload=lab, latency_ms=lat_ms, fabric_dyn_W=p_fab, E_fpga_uJ=e_fpga_uJ,
                         MACs=macs, SOPs=sops, neuron_updates=upd, model_bytes=mb, pJ_per_weight_byte=eb,
                         E45_arith_nJ=e_arith / 1e3, E45_weights_nJ=e_wmem / 1e3, E45_state_nJ=e_state / 1e3,
                         E45_total_nJ=(e_arith + e_wmem + e_state) / 1e3))
with open(os.path.join(OUT, "energy_estimate.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows_out[0])); wr.writeheader(); wr.writerows(rows_out)

# Only the 45 nm projection is plotted. "Energy on this FPGA" would be latency x one constant estimated
# power, i.e. a rescaled copy of Figs. 1 and 4, and is therefore reported in energy_estimate.csv / text only.
# The GRU is ~7x larger than every other bar: a standard broken x-axis (two panels with break marks) keeps
# the composition of the small bars readable without a non-standard "cut" bar.
segs6 = [("Arithmetic", "E45_arith_nJ", "arith"), ("Weight reads", "E45_weights_nJ", "weights"),
         ("Neuron-state access", "E45_state_nJ", "state")]
labels6 = ["MNIST, ANN", "MNIST, SNN", "ECG beat, ANN", "ECG beat, SNN", "ECG stream, GRU", "ECG stream, rec. SNN"]
small_max = max(r["E45_total_nJ"] for r in rows_out if "GRU" not in r["workload"])
gru_tot = max(r["E45_total_nJ"] for r in rows_out)
fig, (d2, d3) = plt.subplots(1, 2, figsize=(W, 2.45), sharey=True, gridspec_kw={"width_ratios": [4.2, 1]})
for d in (d2, d3):
    for i, r in enumerate(rows_out):
        left = 0
        for _, key, sk in segs6:
            if r[key] > 0:
                hbar(d, i, r[key], SEG[sk], 0.55, left)
            left += r[key]
lo_max = small_max * 1.32
d2.set_xlim(0, lo_max)
d3.set_xlim(gru_tot - 0.5 * (lo_max / 4.2), gru_tot + 0.5 * (lo_max / 4.2))
for i, r in enumerate(rows_out):
    t = r["E45_total_nJ"]
    if "GRU" in r["workload"]:
        vlabel(d2, lo_max * 0.86, i, f"{t:.0f}", ha="center")   # same unit (nJ) as the axis
    else:
        vlabel(d2, t + lo_max * 0.015, i, f"{t:.0f}")
d3.set_xticks([round(gru_tot, -2)])
# break marks
d2.spines["right"].set_visible(False); d3.spines["left"].set_visible(False)
d2.tick_params(axis="y", which="both", right=False); d3.tick_params(axis="y", which="both", left=False, right=False)
kw = dict(marker=[(-1, -2.5), (1, 2.5)], markersize=6, linestyle="none", color=K, mec=K, mew=0.6, clip_on=False)
d2.plot([1, 1], [0, 1], transform=d2.transAxes, **kw)
d3.plot([0, 0], [0, 1], transform=d3.transAxes, **kw)
d2.set_yticks(range(len(rows_out))); d2.set_yticklabels(labels6); d2.set_ylim(len(rows_out) - 0.4, -0.6)
d2.tick_params(axis="y", which="both", length=0)
d2.set_xlabel("Estimated energy per inference (nJ)", loc="right")
fig.legend(handles=[Patch(**SEG[k], label=l) for l, _, k in segs6], loc="upper center", ncol=3,
           bbox_to_anchor=(0.6, 1.0), handleheight=1.0, handlelength=1.5, columnspacing=1.0, fontsize=7)
for d in (d2, d3):
    finish(d)
fig.tight_layout(pad=0.3, w_pad=0.25, rect=(0, 0, 1, 0.9))
save(fig, "fig6_energy_estimate")

# =============================================================================
# Fig 7 - accuracy-latency trade-off from the overnight sweeps, SOFTWARE ONLY (optimized C on both sides),
# because only there every size / T exists for both networks. Board latency on a fixed seeded subset
# (ECG: 5 000 test beats, MNIST: first 2 000 test images), quality from the 8-bit PC model on the full
# test set (bit-exact with the board for all sampled inputs). Mean over seeds (ECG 3, MNIST 2).
# =============================================================================
ovs = list(_csv.DictReader(open(os.path.join(ROOT, "docs", "results", "overnight_2026-10-12", "summary.csv"))))
def pick(job, model, fw="fast"):
    return [r for r in ovs if r["job"] == job and r["model"] == model and r["firmware"] == fw]
fig, (e1, e2) = plt.subplots(2, 1, figsize=(W, 4.0), gridspec_kw={"height_ratios": [1.25, 1]})
# (a) ECG: ANN width sweep, SNN width x T sweep
ann = sorted(pick("A", "ANN"), key=lambda r: int(r["hidden"]))
e1.plot([float(r["board_ms_mean"]) for r in ann], [float(r["pc_int8_mean"]) for r in ann], "-", color=K, lw=0.8,
        marker="o", ms=4.5, mfc="white", mec=K, mew=0.8, label="ANN, hidden 8–128", zorder=3)
for r in ann:
    e1.annotate(r["hidden"], (float(r["board_ms_mean"]), float(r["pc_int8_mean"])), xytext=(0, -9),
                textcoords="offset points", ha="center", **LBL)
snn = pick("A", "SNN")
LS = {"16": ":", "32": "--", "64": "-"}
for H in ("16", "32", "64"):
    rr = sorted([r for r in snn if r["hidden"] == H], key=lambda r: int(r["T"]))
    xs = [float(r["board_ms_mean"]) for r in rr]; ys = [float(r["pc_int8_mean"]) for r in rr]
    e1.plot(xs, ys, LS[H], color=K, lw=0.8, marker="s", ms=4.5, mfc="0.82", mec=K, mew=0.8,
            label=f"SNN, hidden {H}, T = 2, 5, 10", zorder=3)
    e1.annotate(f"T={rr[-1]['T']}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                va="center", **dict(LBL, fontsize=6.5))
e1.axvline(TGT_ECG, **TARGET)
e1.set_xscale("log"); e1.set_xlim(0.7, 30); e1.set_ylim(0.605, 0.73)
e1.set_xlabel("(a) ECG: latency per heartbeat (ms)"); e1.set_ylabel("Macro-F1 (test)")
e1.legend(loc="lower right", fontsize=6.6)
finish(e1, True, True)
# (b) MNIST: SNN T sweep vs the single ANN
mn = sorted(pick("B", "SNN"), key=lambda r: int(r["T"]))
d25 = [r for r in ovs if r["job"] == "D" and r["model"] == "SNN"][0]
dann = [r for r in ovs if r["job"] == "D" and r["model"] == "ANN"][0]
pts_mn = [(float(r["board_ms_mean"]), float(r["pc_int8_mean"]) * 100, r["T"]) for r in mn] + \
         [(float(d25["board_ms_mean"]), float(d25["pc_int8_mean"]) * 100, d25["T"])]
e2.plot([p[0] for p in pts_mn], [p[1] for p in pts_mn], "-", color=K, lw=0.8, marker="s", ms=4.5, mfc="0.82",
        mec=K, mew=0.8, label="SNN, hidden 128, T = 2–25", zorder=3)
for x, yv, T in pts_mn:
    e2.annotate(f"{T}", (x, yv), xytext=(0, -9), textcoords="offset points", ha="center", **dict(LBL, fontsize=6.5))
e2.plot([float(dann["board_ms_mean"])], [float(dann["pc_int8_mean"]) * 100], ls="", marker="o", ms=5, mfc="white",
        mec=K, mew=0.9, label="ANN, hidden 128", zorder=4)
e2.axvline(TGT_MNIST, **TARGET)
e2.set_xscale("log"); e2.set_xlim(3, 100); e2.set_ylim(92, 98.6)
e2.set_xlabel("(b) MNIST: latency per image (ms)"); e2.set_ylabel("Accuracy (%)")
e2.legend(loc="lower right", fontsize=6.6)
finish(e2, True, True)
fig.tight_layout(pad=0.3, h_pad=0.6)
save(fig, "fig7_tradeoff_sweeps")
print("fig7 ECG ANN:", [(r["hidden"], round(float(r["board_ms_mean"]), 2), round(float(r["pc_int8_mean"]), 3), r["cv_macro_f1"][:5]) for r in ann])
print("fig7 ECG SNN:", [(r["hidden"], r["T"], round(float(r["board_ms_mean"]), 2), round(float(r["pc_int8_mean"]), 3), r["cv_macro_f1"][:5]) for r in snn])
print("fig7 MNIST:", [(T, round(x, 2), round(yv, 2)) for x, yv, T in pts_mn], "ANN", dann["board_ms_mean"][:6], dann["pc_int8_mean"][:6])

# =============================================================================
# Fig 8 - streaming ECG: latency per heartbeat at every implementation level (companion of Fig. 1)
# GRU-64 vs recurrent SNN-64. "+ coprocessor" = matched custom unit: GRU on the MAC unit, recurrent
# SNN on the SNN unit (full test set, as in Figs. 2 and 4). No optimized-library level: the library
# has no recurrent layers.
# =============================================================================
LEVELS_ST = [("plain C", "gru/baseline", "rsnn/baseline"), ("optimized C", "gru/fast", "rsnn/fast"),
             ("optimized C\n+ coprocessor", "gru/mac", "rsnn/hw_full")]
fig, ax = plt.subplots(figsize=(W, 2.15))
h = 0.36
for y, (lab, kg, kr) in enumerate(LEVELS_ST):
    for k, st, o in ((kg, ANN, -h / 2 - 0.02), (kr, SNN, h / 2 + 0.02)):
        v = m(k, src=ST)
        hbar(ax, y + o, v, st, h)
        vlabel(ax, v * 1.1, y + o, fmt_ms(v))
ax.axvline(TGT_ECG, **TARGET)
ax.set_yticks(range(len(LEVELS_ST))); ax.set_yticklabels([l for l, _, _ in LEVELS_ST])
for tl in ax.get_yticklabels():
    tl.set_multialignment("right"); tl.set_linespacing(0.95)
ax.tick_params(axis="y", which="both", length=0)
ax.set_xscale("log"); ax.set_xlim(1, 2e4); ax.set_ylim(len(LEVELS_ST) - 0.45, -0.55)
ax.set_xlabel("Latency per heartbeat on PicoRV32 at 100 MHz (ms)")
finish(ax)
# every bar row is occupied up to ~5 000 ms, so the legend goes above the axes
fig.legend(handles=[Patch(**ANN, label="GRU-64"), Patch(**SNN, label="Recurrent SNN-64"),
                    Line2D([], [], **TARGET, label="Latency target")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handleheight=1.0, handlelength=1.5,
           columnspacing=1.0, fontsize=7)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.87))
save(fig, "fig8_streaming_latency_overview")
print("fig8:", [(l.replace("\n", " "), round(m(kg, src=ST), 1), round(m(kr, src=ST), 2)) for l, kg, kr in LEVELS_ST])

print("figures ->", OUT)
for r in rows_out:
    print(f"{r['workload']:<22} FPGA {r['E_fpga_uJ']:>10.1f} uJ | 45nm {r['E45_total_nJ']:>9.1f} nJ "
          f"(arith {r['E45_arith_nJ']:.1f}, weights {r['E45_weights_nJ']:.1f}, state {r['E45_state_nJ']:.1f})")
