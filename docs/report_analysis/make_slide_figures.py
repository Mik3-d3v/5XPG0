#!/usr/bin/env python3
"""make_slide_figures.py - the three latency panels for the 'Results' slide (slide 10).
Same data, colours and levels as Fig. 1 / Fig. 5 in make_figures.py, but one transparent SVG
(+ 300 dpi PNG fallback) per panel, Calibri text, no per-plot legend (the slide has a shared one),
and the best final-configuration bar outlined. Usage: python3 make_slide_figures.py [out_dir]"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
FF, ST = R["feedforward"], R["streaming"]

C_ANN, C_SNN, INK, MUTED = "#2a78d6", "#c81919", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": ["Calibri", "Carlito", "DejaVu Sans"], "font.size": 8,
    "svg.fonttype": "none", "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": "#d9d9d9", "grid.linewidth": 0.4,
    "hatch.color": "white", "hatch.linewidth": 0.8})
TGT_MNIST, TGT_ECG = 40, 200
H = 2.15  # panel height on the slide [in]

VARS = [("baseline", "plain C"), ("muriscv", "opt. ANN\nlibrary"), ("fast", "opt. C"),
        ("mac", "opt. C +\ncoproc.")]
VARS_ST = [("plain C", "gru/baseline", "rsnn/baseline"), ("opt. C", "gru/fast", "rsnn/fast"),
           ("opt. C +\ncoproc.", "gru/mac", "rsnn/hw_full")]


def lat(src, k):
    return src[k]["lat_ms"]["mean"] if k in src else None


def panel(fname, title, rows, src, tgt, width, ylim, ylabel=None):
    fig = plt.figure(figsize=(width, H))
    left = 0.62 / width
    ax = fig.add_axes([left, 0.21, 1 - left - 0.05 / width, 0.62])
    x = np.arange(len(rows)); w = 0.38
    last = len(rows) - 1
    a_last, s_last = lat(src, rows[last][1]), lat(src, rows[last][2])
    winner = "a" if a_last < s_last else "s"
    for i, (_, ka, ks) in enumerate(rows):
        va, vs = lat(src, ka), lat(src, ks)
        for which, v, off, col, hatch in (("a", va, -w / 2 if vs else 0, C_ANN, None),
                                          ("s", vs, w / 2, C_SNN, "////")):
            if v is None:
                continue
            hl = i == last and which == winner
            ax.bar(x[i] + off, v, w * 0.92, color=col, hatch=hatch, zorder=3,
                   edgecolor="white", linewidth=0)
            if hl:  # outline drawn separately so the hatch stays white on any matplotlib version
                ax.bar(x[i] + off, v, w * 0.92, fill=False, zorder=4, edgecolor=INK, linewidth=1.3)
            ax.text(x[i] + off, v * 1.13, f"{v:.0f}" if v >= 100 else f"{v:.3g}", ha="center",
                    va="bottom", fontsize=7, color=INK, fontweight="bold" if hl else "normal")
    ax.axhline(tgt, color=MUTED, lw=0.9, ls="--", zorder=2)
    ax.set_yscale("log"); ax.set_ylim(*ylim); ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7, linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(title, fontsize=9.5, color=INK, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
    svg = os.path.join(OUT, fname + ".svg")
    fig.savefig(svg, transparent=True)
    fig.savefig(os.path.join(OUT, fname + ".png"), transparent=True, dpi=300)
    plt.close(fig)
    # keep text editable and in the slide font
    s = open(svg, encoding="utf-8").read()
    s = re.sub(r"font-family:\s*[^;\"]*", "font-family: Calibri, Carlito, sans-serif", s)
    open(svg, "w", encoding="utf-8").write(s)


panel("slide_mnist", "MNIST (784-128-10)",
      [(l, f"MNIST/ann/{v}", f"MNIST/snn/{v}") for v, l in VARS], FF, TGT_MNIST, 2.55, (1, 3000),
      "latency per inference [ms]")
panel("slide_ecg_beat", "ECG, one beat (129-64-3)",
      [(l, f"ECG/ann/{v}", f"ECG/snn/{v}") for v, l in VARS], FF, TGT_ECG, 2.55, (1, 3000))
panel("slide_ecg_stream", "ECG streaming (GRU-64 vs recurrent SNN-64)", VARS_ST, ST, TGT_ECG, 3.45,
      (1, 3e4), "latency per beat [ms]")

# speed-ups shown in the slide callouts (final configuration: "+ coprocessor")
ratio = lambda a, b: a / b
print(f"MNIST   ANN faster by {ratio(FF['MNIST/snn/mac']['lat_ms']['mean'], FF['MNIST/ann/mac']['lat_ms']['mean']):.2f}x")
print(f"ECG     ANN faster by {ratio(FF['ECG/snn/mac']['lat_ms']['mean'], FF['ECG/ann/mac']['lat_ms']['mean']):.2f}x")
print(f"stream  SNN faster by {ratio(ST['gru/mac']['lat_ms']['mean'], ST['rsnn/hw_full']['lat_ms']['mean']):.1f}x")
print("slide figures ->", OUT)
