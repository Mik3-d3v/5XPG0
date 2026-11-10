#!/usr/bin/env python3
"""make_figures.py - report figures from results.json (+ per-beat CSVs for the ECDF).
Run compute_results.py first.  Usage: python3 make_figures.py [out_dir]
Default out_dir: ./figures (next to this script)."""
import json, os, sys, gzip, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
FF, ST, TS = R["feedforward"], R["streaming"], R["T_study"]

C_ANN, C_SNN, C_3, C_4, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.frameon": False,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.4})
W = 3.5

def g(key, m="lat_ms"):
    v = FF.get(key); return (v[m]["mean"], v[m]["std"]) if v else (None, None)

# ---- Fig 1: latency per variant ------------------------------------------
# Same levels and targets as fig1_latency_overview in make_report_figures.py:
# targets from Table I (updated), MNIST <= 40 ms/image, ECG <= 200 ms/beat;
# "+ coprocessor" = matched custom units (ANN on the MAC unit, SNN on the SNN unit).
TGT_MNIST, TGT_ECG = 40, 200
# Panels side by side at double-column width (7.16 in) so the level names and values fit horizontally.
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
VARS = [("baseline", "plain C"), ("muriscv", "optimized\nANN library"), ("fast", "optimized C"),
        ("mac", "optimized C\n+ coprocessor")]
for ax, ds, tgt in ((axes[0], "MNIST", TGT_MNIST), (axes[1], "ECG", TGT_ECG)):
    x = np.arange(len(VARS)); w = 0.38
    for off, model, col, hatch in ((-w / 2, "ann", C_ANN, None), (w / 2, "snn", C_SNN, "////")):
        for i, (v, _) in enumerate(VARS):
            m, s = g(f"{ds}/{model}/{v}")
            if m is None: continue
            ax.bar(x[i] + off, m, w * 0.92, color=col, edgecolor="white", linewidth=0.6, hatch=hatch)
            ax.text(x[i] + off, m * 1.12, f"{m:.3g}", ha="center", va="bottom", fontsize=5.6, color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.3))
    ax.axhline(tgt, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([l for _, l in VARS], fontsize=6.2, linespacing=0.95)
    ax.set_title(ds + (" (784-128-10)" if ds == "MNIST" else " (129-64-3)"), fontsize=7.5)
    ax.set_ylim(1, 2000 if ds == "MNIST" else 600)
    ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
axes[0].set_ylabel("latency per inference [ms]")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=C_ANN, label="ANN"), Patch(facecolor=C_SNN, hatch="////", edgecolor="white", label="SNN"),
                    plt.Line2D([], [], color=MUTED, lw=0.8, ls="--", label="latency target")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
fig.savefig(os.path.join(OUT, "fig_latency_variants.pdf"))
fig.savefig(os.path.join(OUT, "fig_latency_variants.svg"), transparent=True, bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: ECG macro-F1 vs latency --------------------------------------
fig, ax = plt.subplots(figsize=(W, 2.3))
pts = []
for model, col, mk in (("ann", C_ANN, "o"), ("snn", C_SNN, "s")):
    for v, lab in (("baseline", "base"), ("muriscv", "muRISCV-NN"), ("fast", "fast"), ("hw", "hw")):
        k = f"ECG/{model}/{v}"
        if k in FF:
            pts.append((FF[k]["lat_ms"]["mean"], FF[k]["f1"]["mean"], FF[k]["f1"]["std"], col, mk, f"{model.upper()} {lab}"))
# streaming: latency from the board subset (GRU fast) / full run (RSNN hw); F1 on all DS2 beats
rs_f1 = ST["rsnn/hw_full"]["f1"]
pts.append((ST["gru/fast"]["lat_ms"]["mean"], 0.667, 0.0, C_3, "^", "GRU fast (stream)"))   # 0.667 = int8 DS2, docs/snn_coprocessor_and_streaming.md 3.3
pts.append((ST["rsnn/fast"]["lat_ms"]["mean"], rs_f1["mean"], rs_f1["std"], C_4, "D", "RSNN fast (stream)"))
pts.append((ST["rsnn/hw_full"]["lat_ms"]["mean"], rs_f1["mean"], rs_f1["std"], C_4, "D", "RSNN hw (stream)"))
for (x, y, e, c, mk, lab) in pts:
    ax.errorbar(x, y, yerr=e, fmt=mk, ms=4.5, color=c, mec="white", mew=0.6, elinewidth=0.6, capsize=1.5)
offs = {"ANN base": (5, -2), "ANN muRISCV-NN": (0, 9), "ANN fast": (-5, -2), "SNN base": (5, -2), "SNN fast": (0, 8),
        "SNN hw": (-5, -2), "GRU fast (stream)": (0, 6), "RSNN fast (stream)": (0, -13), "RSNN hw (stream)": (-5, -2)}
for (x, y, e, c, mk, lab) in pts:
    dx, dy = offs.get(lab, (4, 3))
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, color=INK,
                ha="right" if dx < 0 else ("left" if dx > 0 else "center"))
ax.axvline(50, color=MUTED, lw=0.7, ls="--"); ax.text(54, 0.716, "50 ms target", fontsize=5.6, color=MUTED, va="center")
ax.set_xscale("log"); ax.set_xlim(1.2, 3000); ax.set_ylim(0.65, 0.72)
ax.set_xlabel("latency per beat on PicoRV32 @ 100 MHz [ms]"); ax.set_ylabel("macro-F1 (N/S/V), DS2")
ax.grid(True, which="major"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ecg_f1_vs_latency.pdf")); plt.close(fig)

# ---- Fig 3: T study (two stacked panels, shared x) ------------------------
Ts = sorted(int(t) for t in TS["points"])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
for lvl, col, mk, lab in (("fast", C_SNN, "s", "SNN fast (software)"), ("hw", C_4, "D", "SNN hw (coprocessor)")):
    L = [TS["points"][str(t)][lvl]["lat_ms"] for t in Ts]
    f = TS[f"fit_{lvl}"]
    a1.plot(Ts, L, mk + "-", color=col, ms=4, lw=1.4, mec="white", mew=0.5)
    a1.annotate(f"{lab}\n{f['intercept_ms']:.2f} + {f['slope_ms_per_T']:.3f}$\\cdot$T ms", (Ts[-1], L[-1]),
                xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=5.8, color=INK)
a1.axhline(TS["ann_fast_first2000_ms"], color=C_ANN, lw=1.2)
a1.text(20, TS["ann_fast_first2000_ms"] - 0.15, f"ANN fast {TS['ann_fast_first2000_ms']:.2f} ms", color=INK, fontsize=5.8, ha="right", va="top")
a1.set_ylabel("latency [ms/beat]"); a1.set_ylim(3.3, 13.2); a1.grid(True); a1.set_axisbelow(True)
F = [TS["points"][str(t)]["f1_int8_ds2"] for t in Ts]
a2.plot(Ts, F, "o-", color=MUTED, ms=3.5, lw=1.2, mec="white", mew=0.5)
for t, v in zip(Ts, F): a2.annotate(f"{v:.2f}", (t, v), xytext=(3, -8 if t == 1 else 2), textcoords="offset points", fontsize=5.6)
a2.set_ylabel("macro-F1\n(int8, DS2)"); a2.set_xlabel("timesteps T at inference (model trained at T = 10)")
a2.set_xticks(Ts); a2.grid(True); a2.set_ylim(0.25, 0.8); a2.set_axisbelow(True)
fig.tight_layout(h_pad=0.4); fig.savefig(os.path.join(OUT, "fig_ecg_T_study.pdf")); plt.close(fig)

# ---- Fig 4: per-beat latency ECDF (ECG, seed 0, all DS2 beats) ------------
def cyc(p):
    with gzip.open(os.path.join(ROOT, p), "rt") as f:
        return np.array([float(r["cycles"]) for r in csv.DictReader(f)]) / 1e5
series = [("ANN fast", FF["ECG/ann/fast"]["sources"][0], C_ANN, "-"),
          ("ANN muRISCV-NN", FF["ECG/ann/muriscv"]["sources"][0], C_ANN, ":"),
          ("SNN fast", FF["ECG/snn/fast"]["sources"][0], C_SNN, "-"),
          ("SNN hw", FF["ECG/snn/hw"]["sources"][0], C_4, "-"),
          ("RSNN hw (stream)", ST["rsnn/hw_full"]["sources"][0], C_3, "--")]
fig, ax = plt.subplots(figsize=(W, 1.9))
for lab, p, col, ls in series:
    v = np.sort(cyc(p)); ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=col, lw=1.4, label=lab)
ax.set_xlim(0, 12); ax.set_xlabel("latency per beat [ms] (seed 0, 49 273 DS2 beats)"); ax.set_ylabel("fraction of beats")
ax.legend(fontsize=5.8, loc="lower right"); ax.grid(True); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ecg_latency_ecdf.pdf")); plt.close(fig)

# ---- Fig 5: streaming latency per variant (GRU vs recurrent SNN) -----------
# Same levels as fig8_streaming_latency_overview in make_report_figures.py. "+ coprocessor":
# GRU on the MAC unit, recurrent SNN on the SNN unit (full DS2). No optimized-library level:
# the library has no recurrent layers.
fig, ax = plt.subplots(figsize=(W, 2.3))
VARS_ST = [("plain C", "gru/baseline", "rsnn/baseline"), ("optimized C", "gru/fast", "rsnn/fast"),
           ("optimized C\n+ coprocessor", "gru/mac", "rsnn/hw_full")]
x = np.arange(len(VARS_ST)); w = 0.38
for i, (_, kg, kr) in enumerate(VARS_ST):
    for off, k, col, hatch in ((-w / 2, kg, C_ANN, None), (w / 2, kr, C_SNN, "////")):
        v = ST[k]["lat_ms"]["mean"]
        ax.bar(x[i] + off, v, w * 0.92, color=col, edgecolor="white", linewidth=0.6, hatch=hatch)
        ax.text(x[i] + off, v * 1.12, f"{v:.0f}" if v >= 100 else f"{v:.3g}", ha="center", va="bottom", fontsize=5.6, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.3))
ax.axhline(TGT_ECG, color=MUTED, lw=0.8, ls="--", zorder=1)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([l for l, _, _ in VARS_ST], fontsize=6.2, linespacing=0.95)
ax.set_title("ECG streaming (GRU-64 vs recurrent SNN-64)", fontsize=7.5)
ax.set_ylim(1, 2e4)
ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
ax.set_ylabel("latency per beat [ms]")
fig.legend(handles=[Patch(facecolor=C_ANN, label="GRU"), Patch(facecolor=C_SNN, hatch="////", edgecolor="white", label="recurrent SNN"),
                    plt.Line2D([], [], color=MUTED, lw=0.8, ls="--", label="latency target")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_streaming_latency_variants.pdf"))
fig.savefig(os.path.join(OUT, "fig_streaming_latency_variants.png"), dpi=300); plt.close(fig)

# ---- Fig 6: slide version, all three latency panels side by side -----------
# MNIST and ECG beat (Fig. 1) next to ECG streaming (Fig. 5) on one shared log axis, sized for a
# 16:9 PowerPoint slide (13.33 x 7.5 in) with slide-readable text. Saved as PDF and 300 dpi PNG.
PANELS = [("MNIST (784-128-10)", TGT_MNIST,
           [(l, f"MNIST/ann/{v}", f"MNIST/snn/{v}") for v, l in VARS], FF),
          ("ECG, one beat (129-64-3)", TGT_ECG,
           [(l, f"ECG/ann/{v}", f"ECG/snn/{v}") for v, l in VARS], FF),
          ("ECG streaming (64 hidden)", TGT_ECG, VARS_ST, ST)]
with plt.rc_context({"font.size": 13}):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.0), sharey=True, gridspec_kw={"width_ratios": [4, 4, 3]})
    for ax, (title, tgt, rows, src) in zip(axes, PANELS):
        x = np.arange(len(rows)); w = 0.4
        for i, (_, ka, ks) in enumerate(rows):
            for off, k, col, hatch in ((-w / 2, ka, C_ANN, None), (w / 2, ks, C_SNN, "////")):
                if k not in src: continue
                v = src[k]["lat_ms"]["mean"]
                ax.bar(x[i] + off, v, w * 0.92, color=col, edgecolor="white", linewidth=0.8, hatch=hatch)
                ax.text(x[i] + off, v * 1.15, f"{v:.0f}" if v >= 100 else f"{v:.3g}", ha="center", va="bottom",
                        fontsize=11, color=INK, rotation=90,
                        bbox=dict(fc="white", ec="none", pad=0.5))
        ax.axhline(tgt, color=MUTED, lw=1.3, ls="--", zorder=1)
        ax.set_yscale("log"); ax.set_ylim(1, 3e4); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks(x); ax.set_xticklabels([l for l, _, _ in rows], fontsize=10.5, linespacing=0.95)
        ax.set_title(title, fontsize=14, pad=20)
        ax.text(0.5, 1.0, f"target ≤ {tgt} ms", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11, color=MUTED)
        ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
    axes[0].set_ylabel("latency per inference [ms]")
    fig.legend(handles=[Patch(facecolor=C_ANN, label="ANN / GRU"),
                        Patch(facecolor=C_SNN, hatch="////", edgecolor="white", label="SNN / recurrent SNN"),
                        plt.Line2D([], [], color=MUTED, lw=1.3, ls="--", label="latency target")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_latency_combined_slide.pdf"))
    fig.savefig(os.path.join(OUT, "fig_latency_combined_slide.png"), dpi=300)
    plt.close(fig)

# ---- Fig 7: estimated FPGA dynamic power per component (Vivado) ------------
# Same arrangement and text as fig5_fpga_power in make_report_figures.py, in this script's colours.
# PS7 (~1.26 W) and device static power (~0.15 W) are identical in both designs and left out.
def power_rows(path):
    rows, on = {}, False
    txt = open(path, encoding="utf8", errors="ignore").read()
    for line in txt.splitlines():
        if "On-Chip Components" in line and "1.1" in line:
            on = True; continue
        if on and line.startswith("| ") and not line.startswith("|   ") and "Power (W)" not in line and "Static Power" not in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            try: rows[parts[0]] = float(parts[1])
            except (ValueError, IndexError): pass
            if parts[0] == "Total": break
    return rows
P = {k: power_rows(os.path.join(ROOT, "docs", "results", "snn_coprocessor", d, "spike_top_power_routed.rpt"))
     for k, d in (("ref", "vivado_reference"), ("snn", "vivado_snn"))}
# Slide colours: grey = baseline SoC, theme red = the added SNN coprocessor (matches the SNN red
# on the results slide). Text in Calibri, kept as editable text in the SVG.
C_BASE, C_SNN_RED = "#8a8a8a", "#c81919"
plt.rcParams.update({"font.family": ["Calibri", "Carlito", "DejaVu Sans"], "svg.fonttype": "none",
                     "font.size": 10})  # Calibri runs small; ~1.3x the serif size to match the old figure
designs = [("ref", "PicoRV32 SoC", dict(facecolor=C_BASE, edgecolor="white", linewidth=0.6)),
           ("snn", "PicoRV32 SoC +\nSNN coprocessor", dict(facecolor=C_SNN_RED, edgecolor="white", linewidth=0.6, hatch="////"))]
comp = [("Block RAM", "Block RAM"), ("Signals (wiring)", "Signals"),
        ("Clocks", "Clocks"), ("Logic (LUTs, FFs)", "Slice Logic"), ("DSP slices", "DSPs")]
fig, ax = plt.subplots(figsize=(W, 2.4))
y = np.arange(len(comp)); h = 0.36
for j, (k, _, st) in enumerate(designs):
    for yy, (_, key) in zip(y, comp):
        vv = P[k].get(key, 0) * 1000
        yo = yy + (j - 0.5) * (h + 0.04)
        ax.barh(yo, vv, height=h, **st)
        ax.text(vv + 4, yo, f"{vv:.0f}", va="center", ha="left", fontsize=10, color=INK)
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in comp]); ax.set_ylim(len(comp) - 0.4, -0.6)
ax.set_xlim(0, 300); ax.set_xlabel("Estimated dynamic power (mW)")
ax.legend(handles=[Patch(label=lab, **st) for _, lab, st in designs], loc="lower right", handleheight=1.0, fontsize=9.5)
ax.tick_params(axis="y", which="both", length=0)
ax.xaxis.grid(True, which="major"); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
_svg = os.path.join(OUT, "fig_fpga_power.svg")
fig.savefig(_svg, transparent=True, bbox_inches="tight"); plt.close(fig)
import re
_s = open(_svg, encoding="utf-8").read()
open(_svg, "w", encoding="utf-8").write(re.sub(r"font-family:\s*[^;\"]*", "font-family: Calibri, Carlito, sans-serif", _s))
print("figures ->", OUT)
