#!/usr/bin/env python3
"""compare_pynq.py — ANN-vs-SNN hardware breakdown from the board results.

Reads <export-root>/<model>_<task>_seed<k>/board_results_{fast,baseline}.csv
(made by benchmark_pynq.py / benchmark_pynq_ecg.py) and prints markdown tables:
  1. latency distribution per model (fast firmware),
  2. speed-up of the optimisations (baseline vs fast, same images),
  3. linear cost model per model: cycles ~ events,
  4. paired per-image comparison (same test image for ANN and SNN),
  5. prediction agreement between ANN and SNN and per-class accuracy,
and, if muRISCV-NN results exist (--fw muriscv, <model>_<task>_seed<k>_muriscv/
board_results_muriscv.csv):
  6. the ANN on three kernels (ours baseline, muRISCV-NN, ours fast) and the
     SNN, same images,
  7. predictions of muRISCV-NN vs our int8 ANN (exact McNemar test),
  8. cycles per multiply-accumulate of the three ANN kernels.

Run from code/ann_vs_snn_mnist/:  python pynq/compare_pynq.py [--out file.md]
Committed results instead of export/:  --results ../../docs/results/ann_vs_snn
  (files <model>_seed<k>_<fw>_board_results.csv)
ECG:  python pynq/compare_pynq.py --task ecg --export-root ../ann_vs_snn_ecg/pynq/export
"""

import argparse
import csv
import glob
import gzip
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLK_MHZ = 100.0
TASKS = {  # class names, name of a nonzero ANN input
    "mnist": ([str(k) for k in range(10)], "nonzero pixel"),
    "ecg": (["N", "S", "V"], "nonzero input"),
}
ANN_DIMS = {"mnist": (784, 128, 10), "ecg": (129, 64, 3)}   # n_in, hidden, n_out


def load(root, task, model, seed, fw, results=None):
    if results:      # committed results: <model>_seed<k>_<fw>_board_results.csv
        p = os.path.join(results, f"{model.lower()}_seed{seed}_{fw}_board_results.csv")
    elif fw == "muriscv":
        p = os.path.join(root, f"{model.lower()}_{task}_seed{seed}_muriscv",
                         "board_results_muriscv.csv")
    else:
        p = os.path.join(root, f"{model.lower()}_{task}_seed{seed}",
                         f"board_results_{fw}.csv")
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p += ".gz"                   # committed ECG results are gzipped
    if not os.path.exists(p):
        return None
    with (gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {k: col(k) for k in ("label", "board_pred", "cycles", "in_events",
                                "in_changes", "hidden_events", "outputs_match")}


def mcnemar_p(b, c):
    """exact two-sided McNemar test: b, c = discordant pair counts"""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def ms(c):
    return c / (CLK_MHZ * 1e3)


def fit(y, cols, names):
    A = np.column_stack(cols + [np.ones(len(y))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((A @ c - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    terms = " + ".join(f"{v:,.0f} · {n}" for v, n in zip(c[:-1], names))
    return f"{terms} + {c[-1]:,.0f}", r2, c


def muriscv_tables(R, seeds, w, task):
    """sections 6-8: the external ANN reference (muRISCV-NN, dense, scalar C)"""
    unit = "image" if task == "mnist" else "beat"
    n_in, hid, n_out = ANN_DIMS[task]
    w(f"\n### 6. ANN kernels: ours baseline, muRISCV-NN, ours fast (same {unit}s) "
      f"[ms/{unit}]\n")
    w("| seed | n | ANN baseline | ANN muRISCV-NN | ANN fast | SNN fast | "
      f"muRISCV-NN vs baseline | fast vs muRISCV-NN | {unit}s where SNN fast < muRISCV-NN |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        M, F, B, S = (R[(m, s, fw)] for m, fw in (("ANN", "muriscv"), ("ANN", "fast"),
                                                  ("ANN", "baseline"), ("SNN", "fast")))
        if M is None or F is None:
            continue
        n = min(len(M["cycles"]), len(F["cycles"]))
        mc, fc = M["cycles"][:n], F["cycles"][:n]
        nb = min(n, len(B["cycles"])) if B is not None else 0
        bl = f"{ms(B['cycles'][:nb]).mean():.2f} (n={nb})" if nb else "n/a"
        vb = f"{B['cycles'][:nb].mean() / M['cycles'][:nb].mean():.2f}× faster" if nb else "n/a"
        sn = f"{ms(S['cycles'][:n]).mean():.2f}" if S is not None else "n/a"
        sf = f"{int((S['cycles'][:n] < mc).sum())}/{n}" if S is not None else "n/a"
        w(f"| {s} | {n} | {bl} | {ms(mc).mean():.2f} (min {ms(mc).min():.2f}, "
          f"max {ms(mc).max():.2f}) | {ms(fc).mean():.2f} | {sn} | {vb} | "
          f"{mc.mean() / fc.mean():.2f}× faster | {sf} |")

    w(f"\n### 7. Predictions: muRISCV-NN vs our int8 ANN (same {unit}s)\n")
    w("| seed | n | muRISCV-NN acc | ours acc | both right | only ours right | "
      "only muRISCV-NN right | both wrong | same prediction | McNemar p (exact) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        M, F = R[("ANN", s, "muriscv")], R[("ANN", s, "fast")]
        if M is None or F is None:
            continue
        n = min(len(M["label"]), len(F["label"]))
        y = M["label"][:n]
        m_ok, f_ok = M["board_pred"][:n] == y, F["board_pred"][:n] == y
        b, c = int((f_ok & ~m_ok).sum()), int((~f_ok & m_ok).sum())
        w(f"| {s} | {n} | {m_ok.mean() * 100:.2f} % | {f_ok.mean() * 100:.2f} % | "
          f"{int((m_ok & f_ok).sum())} | {b} | {c} | {int((~m_ok & ~f_ok).sum())} | "
          f"{int((M['board_pred'][:n] == F['board_pred'][:n]).sum())} | "
          f"{mcnemar_p(b, c):.3f} |")

    dense = n_in * hid + hid * n_out
    w(f"\n### 8. Cycles per multiply-accumulate ({n_in}-{hid}-{n_out}: {dense:,} dense MACs)\n")
    w("| seed | ours baseline, per dense MAC | muRISCV-NN, per dense MAC | "
      f"ours fast, per executed MAC (cost-model slope / {hid}) | ours fast, per dense MAC |")
    w("|---|---|---|---|---|")
    for s in seeds:
        M, F, B = R[("ANN", s, "muriscv")], R[("ANN", s, "fast")], R[("ANN", s, "baseline")]
        if M is None or F is None:
            continue
        _, _, c = fit(F["cycles"], [F["in_events"], F["hidden_events"]], ["px", "hid"])
        bl = f"{B['cycles'].mean() / dense:.1f}" if B is not None else "n/a"
        w(f"| {s} | {bl} | {M['cycles'].mean() / dense:.1f} | {c[0] / hid:.1f} | "
          f"{F['cycles'].mean() / dense:.1f} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None)
    p.add_argument("--task", choices=sorted(TASKS), default="mnist")
    p.add_argument("--export-root", default=os.path.join(HERE, "export"))
    p.add_argument("--results", default=None,
                   help="read <model>_seed<k>_<fw>_board_results.csv from this folder")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console
    classes, nz_name = TASKS[a.task]

    if a.results:
        seeds = sorted({int(os.path.basename(f).split("_")[1][4:])
                        for f in glob.glob(os.path.join(a.results, "*_seed*_board_results.csv*"))})
    else:   # <model>_<task>_seed<k>[_muriscv]
        seeds = sorted({int(f.rsplit("seed", 1)[1].split(os.sep)[0].split("_")[0])
                        for f in glob.glob(os.path.join(a.export_root, f"*_{a.task}_seed*"))})
    R = {(m, s, fw): load(a.export_root, a.task, m, s, fw, a.results) for m in ("ANN", "SNN")
         for s in seeds for fw in ("fast", "baseline", "muriscv")}
    out = []
    w = out.append

    w("### 1. Latency distribution, fast firmware [ms/sample]\n")
    w("| model | seed | n | mean | p5 | median | p95 | min | max | bit-exact |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for m in ("ANN", "SNN"):
        for s in seeds:
            r = R[(m, s, "fast")]
            if r is None:
                continue
            l = ms(r["cycles"])
            w(f"| {m} | {s} | {len(l)} | {l.mean():.2f} | {np.percentile(l, 5):.2f} | "
              f"{np.median(l):.2f} | {np.percentile(l, 95):.2f} | {l.min():.2f} | "
              f"{l.max():.2f} | {int(r['outputs_match'].sum())}/{len(l)} |")

    w("\n### 2. Speed-up of the optimisations (same images)\n")
    w("| model | seed | n | baseline [ms] | fast [ms] | speed-up | per-image min / max |")
    w("|---|---|---|---|---|---|---|")
    for m in ("ANN", "SNN"):
        for s in seeds:
            b, f = R[(m, s, "baseline")], R[(m, s, "fast")]
            if b is None or f is None:
                continue
            n = min(len(b["cycles"]), len(f["cycles"]))
            bc, fc = b["cycles"][:n], f["cycles"][:n]
            sp = bc / fc
            w(f"| {m} | {s} | {n} | {ms(bc).mean():.2f} | {ms(fc).mean():.2f} | "
              f"{bc.mean() / fc.mean():.2f}× | {sp.min():.1f}× / {sp.max():.1f}× |")

    w("\n### 3. Cost model of the fast kernels (least squares, all images)\n")
    w("| model | seed | cycles ≈ | R² | fixed part [ms] | share of mean |")
    w("|---|---|---|---|---|---|")
    for m in ("ANN", "SNN"):
        for s in seeds:
            r = R[(m, s, "fast")]
            if r is None:
                continue
            if m == "ANN":
                eq, r2, c = fit(r["cycles"], [r["in_events"], r["hidden_events"]],
                                [nz_name, "nonzero hidden"])
            elif (r["in_changes"] == r["in_events"]).all():   # direct encoding
                eq, r2, c = fit(r["cycles"], [r["in_events"], r["hidden_events"]],
                                ["nonzero input", "hidden spike"])
            else:
                eq, r2, c = fit(r["cycles"], [r["in_changes"], r["hidden_events"]],
                                ["input change", "hidden spike"])
            w(f"| {m} | {s} | {eq} | {r2:.4f} | {ms(c[-1]):.2f} | "
              f"{c[-1] / r['cycles'].mean() * 100:.0f} % |")

    w("\n### 4. Paired per-image comparison, fast firmware (same test image)\n")
    w("| seed | n | SNN/ANN latency: mean ratio | median | min | max | "
      "corr(ANN, SNN latency) | images where SNN is faster |")
    w("|---|---|---|---|---|---|---|---|")
    for s in seeds:
        A, S = R[("ANN", s, "fast")], R[("SNN", s, "fast")]
        if A is None or S is None:
            continue
        n = min(len(A["cycles"]), len(S["cycles"]))
        ra = S["cycles"][:n] / A["cycles"][:n]
        cc = np.corrcoef(A["cycles"][:n], S["cycles"][:n])[0, 1]
        w(f"| {s} | {n} | {S['cycles'][:n].mean() / A['cycles'][:n].mean():.2f} | "
          f"{np.median(ra):.2f} | {ra.min():.2f} | {ra.max():.2f} | {cc:.3f} | "
          f"{int((ra < 1).sum())} |")

    w("\n### 5. Predictions: ANN vs SNN (fast firmware, same images)\n")
    w("| seed | n | both right | only ANN right | only SNN right | both wrong | "
      "same prediction |")
    w("|---|---|---|---|---|---|---|")
    pc = {}
    for s in seeds:
        A, S = R[("ANN", s, "fast")], R[("SNN", s, "fast")]
        if A is None or S is None:
            continue
        n = min(len(A["label"]), len(S["label"]))
        y = A["label"][:n]
        a_ok, s_ok = A["board_pred"][:n] == y, S["board_pred"][:n] == y
        w(f"| {s} | {n} | {int((a_ok & s_ok).sum())} | {int((a_ok & ~s_ok).sum())} | "
          f"{int((~a_ok & s_ok).sum())} | {int((~a_ok & ~s_ok).sum())} | "
          f"{int((A['board_pred'][:n] == S['board_pred'][:n]).sum())} |")
        for k in range(len(classes)):
            sel = y == k
            pc.setdefault(k, []).append((a_ok[sel].mean(), s_ok[sel].mean(),
                                         ms(A["cycles"][:n][sel]).mean(),
                                         ms(S["cycles"][:n][sel]).mean()))
    if pc:
        w("\nPer class (mean over seeds):\n")
        w(f"| {'digit' if a.task == 'mnist' else 'class'} | ANN acc [%] | SNN acc [%] | "
          "ANN latency [ms] | SNN latency [ms] |")
        w("|---|---|---|---|---|")
        for k in range(len(classes)):
            v = np.array(pc[k]).mean(0)
            w(f"| {classes[k]} | {v[0] * 100:.2f} | {v[1] * 100:.2f} | {v[2]:.2f} | {v[3]:.2f} |")

    if any(R[("ANN", s, "muriscv")] is not None for s in seeds):
        muriscv_tables(R, seeds, w, a.task)

    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
