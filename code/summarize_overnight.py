#!/usr/bin/env python3
"""summarize_overnight.py — tables, CSV and draft figures of overnight.sh.

Reads whatever exists (partial nights are fine) from
  code/ann_vs_snn_ecg/pynq/export_<sub>/ecgA_*, ecgC_*   (sweeps A, C)
  code/ann_vs_snn_mnist/pynq/export_<sub>/mnistB_*, mnistD_*  (B, D)
  code/ann_vs_snn_ecg/runs/<sub>/cv/cv/ecg_*.json          (E)
plus the published reference points in docs/results/, and writes to <res>:
  summary.md, summary.csv              one row per (workload, model, hidden, T,
                                       firmware level), mean ± std over seeds
  job_<X>.md                           the rows of one job (--job X)
  fig1_ecg_f1_vs_latency.png/.csv, fig2_mnist_vs_T.png/.csv,
  fig3_stream_f1_vs_latency.png/.csv
No conclusions are drawn here: only the numbers.
Run with the ECG venv (it has matplotlib):
  python code/summarize_overnight.py --res docs/results/overnight_2026-10-12 [--job A]
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUB = os.path.join(ROOT, "docs", "results")
CLK_KHZ = 100_000.0                    # cycles -> ms

COLS = ["job", "workload", "model", "hidden", "T", "firmware", "seeds", "params",
        "model_bytes", "metric", "pc_float_mean", "pc_float_std", "pc_int8_mean",
        "pc_int8_std", "pc_int8_accuracy_mean", "pc_int8_se_S_mean", "pc_int8_se_V_mean",
        "cv_macro_f1", "board_n_per_seed", "board_ms_mean", "board_ms_std_seeds",
        "board_ms_p5", "board_ms_p95", "board_cycles_mean", "bit_exact", "source"]


def read_csv(p):
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p += ".gz"
    if not os.path.exists(p):
        return None
    with (gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")) as f:
        return list(csv.DictReader(f))


def params_of(meta):
    if "params" in meta:
        return meta["params"]
    return sum(l["n_in"] * l["n_out"] + l["n_out"] for l in meta["layers"])


def ms(v, d=2):
    return "" if v is None or v == "" else f"{v:.{d}f}"


def board_stats(csvs, n_first=None):
    """csvs: per-seed board rows -> pooled stats (None if no seed has results)"""
    csvs = [r[:n_first] if n_first else r for r in csvs if r]
    if not csvs:
        return {}
    cyc = [np.array([float(x["cycles"]) for x in r]) for r in csvs]
    allc = np.concatenate(cyc)
    match = sum(int(x["outputs_match"]) for r in csvs for x in r)
    per_seed = [c.mean() / CLK_KHZ for c in cyc]
    return {"board_n_per_seed": "/".join(str(len(c)) for c in cyc),
            "board_ms_mean": float(np.mean(per_seed)),
            "board_ms_std_seeds": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
            "board_ms_p5": float(np.percentile(allc, 5) / CLK_KHZ),
            "board_ms_p95": float(np.percentile(allc, 95) / CLK_KHZ),
            "board_cycles_mean": float(allc.mean()),
            "bit_exact": f"{match}/{len(allc)}"}


def pc_stats(metas, workload):
    """mean ± std over seeds of the PC metrics (ECG: macro-F1, MNIST: accuracy)"""
    def ms_(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    if workload.startswith("ecg"):
        f = [m["pc_metrics_float"]["macro_f1"] for m in metas]
        q = [m["pc_metrics_int"]["macro_f1"] for m in metas]
        r = {"metric": "macro-F1 (all DS2)",
             "pc_int8_accuracy_mean": float(np.mean([m["pc_metrics_int"]["accuracy"] for m in metas])),
             "pc_int8_se_S_mean": float(np.mean([m["pc_metrics_int"]["se_S"] for m in metas])),
             "pc_int8_se_V_mean": float(np.mean([m["pc_metrics_int"]["se_V"] for m in metas]))}
    else:
        f = [m["pc_test_accuracy_float"] for m in metas]
        q = [m["pc_test_accuracy_int"] for m in metas]
        r = {"metric": "accuracy (10k test)"}
    (r["pc_float_mean"], r["pc_float_std"]), (r["pc_int8_mean"], r["pc_int8_std"]) = ms_(f), ms_(q)
    return r


def group(dirs):
    """export dirs <prefix>_seed<k> -> {prefix: [dirs]}"""
    g = {}
    for d in sorted(dirs):
        g.setdefault(re.sub(r"_seed\d+$", "", os.path.basename(d)), []).append(d)
    return g


def rows_for(job, workload, prefix, dirs, fws, n_first=None, extra=None):
    metas = [json.load(open(os.path.join(d, "meta.json"))) for d in dirs]
    m0 = metas[0]
    base = {"job": job, "workload": workload, "model": m0["model"],
            "hidden": m0.get("hidden", m0["layers"][0]["n_out"]),
            "T": m0.get("timesteps", 0) if m0["model"] in ("SNN", "RSNN") else "",
            "seeds": len(metas), "params": params_of(m0), "model_bytes": m0["model_bytes"],
            "source": prefix, **pc_stats(metas, workload), **(extra or {})}
    out = []
    for fw in fws:
        b = board_stats([read_csv(os.path.join(d, f"board_results_{fw}.csv")) for d in dirs], n_first)
        if b or fw == fws[0]:
            out.append({**base, "firmware": fw, **b})
    return out


def cv_scores(sub):
    s = {}
    for f in glob.glob(os.path.join(HERE, "ann_vs_snn_ecg", "runs", sub, "cv", "cv", "ecg_*.json")):
        res = json.load(open(f))["results"]
        s[os.path.basename(f)[4:-5]] = float(np.mean([r["macro_f1"] for v in res.values() for r in v]))
    return s


def collect(sub):
    ecg_x = os.path.join(HERE, "ann_vs_snn_ecg", "pynq", f"export_{sub}")
    mn_x = os.path.join(HERE, "ann_vs_snn_mnist", "pynq", f"export_{sub}")
    rows, cv = [], cv_scores(sub)
    for pre, dirs in group(glob.glob(os.path.join(ecg_x, "ecgA_*_seed*"))).items():
        cfg = pre[len("ecgA_"):]
        fws = ["fast", "hw"] if cfg.startswith("snn") else ["fast"]
        rows += rows_for("A", "ecg", pre, dirs, fws, extra={"cv_macro_f1": cv.get(cfg, "")})
    for pre, dirs in group(glob.glob(os.path.join(mn_x, "mnistB_*_seed*"))).items():
        rows += rows_for("B", "mnist", pre, dirs, ["fast", "hw"])
    for pre, dirs in group(glob.glob(os.path.join(ecg_x, "ecgC_*_seed*"))).items():
        rows += rows_for("C", "ecg-stream", pre, dirs, ["fast"])
    for pre, dirs in group(glob.glob(os.path.join(mn_x, "mnistD_*_seed*"))).items():
        rows += rows_for("D", "mnist", pre, dirs, ["fast"])
    return rows


def references(n_mnist, n_stream):
    """published points (docs/results/), board stats on comparable subsets"""
    rows = []
    a = os.path.join(PUB, "ann_vs_snn")
    for m, fws in (("ann", ["fast"]), ("snn", ["fast", "hw"])):
        metas = [json.load(open(os.path.join(a, f"{m}_seed{s}_meta.json"))) for s in (0, 1)]
        base = {"job": "ref", "workload": "mnist", "model": m.upper(), "hidden": 128,
                "T": 25 if m == "snn" else "", "seeds": 2, "params": params_of(metas[0]),
                "model_bytes": metas[0]["model_bytes"], **pc_stats(metas, "mnist")}
        for fw in fws:
            b = board_stats([read_csv(os.path.join(a, f"{m}_seed{s}_{fw}_board_results.csv"))
                             for s in (0, 1)], n_mnist)
            rows.append({**base, "firmware": fw, **b,
                         "source": f"published, first {n_mnist} images"})
    e = os.path.join(PUB, "ann_vs_snn_ecg")
    for m, fws in (("ann", ["fast"]), ("snn", ["fast", "hw"])):
        metas = [json.load(open(os.path.join(e, f"{m}_seed{s}_meta.json"))) for s in (0, 1, 2)]
        base = {"job": "ref", "workload": "ecg", "model": m.upper(), "hidden": 64,
                "T": 10 if m == "snn" else "", "seeds": 3, "params": params_of(metas[0]),
                "model_bytes": metas[0]["model_bytes"], **pc_stats(metas, "ecg")}
        for fw in fws:
            b = board_stats([read_csv(os.path.join(e, f"{m}_seed{s}_{fw}_board_results.csv"))
                             for s in (0, 1, 2)])
            rows.append({**base, "firmware": fw, **b, "source": "published, all DS2 beats"})
    st = os.path.join(PUB, "ecg_streaming", "board_subset")
    for m, fws in (("gru", ["fast"]), ("rsnn", ["fast", "hw"])):
        metas = [json.load(open(os.path.join(st, f"{m}_seed{s}_meta.json"))) for s in (0, 1, 2)]
        base = {"job": "ref", "workload": "ecg-stream", "model": m.upper(),
                "hidden": metas[0].get("hidden", metas[0]["layers"][0]["n_out"]),
                "T": "", "seeds": 3, "params": params_of(metas[0]),
                "model_bytes": metas[0]["model_bytes"], **pc_stats(metas, "ecg")}
        for fw in fws:
            b = board_stats([read_csv(os.path.join(st, f"{m}_seed{s}_{fw}_board_results.csv"))
                             for s in (0, 1, 2)])
            rows.append({**base, "firmware": fw, **b,
                         "source": f"published board subset ({metas[0]['n_samples']} beats)"})
    return rows


def fmt_table(rows):
    h = ("| job | workload | model | hidden | T | fw | seeds | params | model B | metric | "
         "PC float | PC int8 | CV macro-F1 | board n/seed | board ms (p5–p95) | bit-exact | source |")
    out = [h, "|" + "---|" * 17]
    for r in rows:
        pm = lambda k: (f"{r[k + '_mean']:.4f} ± {r[k + '_std']:.4f}"
                        if r.get(k + "_mean", "") != "" else "")
        cvv = r.get("cv_macro_f1", "")
        bm = (f"{ms(r['board_ms_mean'], 3)} ({ms(r['board_ms_p5'], 3)}–{ms(r['board_ms_p95'], 3)})"
              if r.get("board_ms_mean", "") != "" else "not run")
        flag = ""
        if r.get("bit_exact"):
            k, n = map(int, r["bit_exact"].split("/"))
            flag = "" if k == n else " **MISMATCH**"
        out.append(f"| {r['job']} | {r['workload']} | {r['model']} | {r['hidden']} | {r['T']} | "
                   f"{r['firmware']} | {r['seeds']} | {r['params']} | {r['model_bytes']} | "
                   f"{r['metric']} | {pm('pc_float')} | {pm('pc_int8')} | "
                   f"{f'{cvv:.4f}' if cvv != '' else ''} | {r.get('board_n_per_seed', '')} | {bm} | "
                   f"{r.get('bit_exact', '')}{flag} | {r['source']} |")
    return "\n".join(out)


def write_csv(path, rows, cols=COLS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def figures(res, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def done(fig, name, data):
        fig.tight_layout()
        fig.savefig(os.path.join(res, name + ".png"), dpi=150)
        plt.close(fig)
        write_csv(os.path.join(res, name + ".csv"), data)

    # 1. ECG: macro-F1 vs board latency, Pareto front over all points
    pts = [r for r in rows if r["workload"] == "ecg" and r.get("board_ms_mean", "") != ""]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        series = {("ANN", "fast"): ("ANN (fast)", "o"), ("SNN", "fast"): ("SNN (fast)", "s"),
                  ("SNN", "hw"): ("SNN (hw coprocessor)", "^")}
        for (m, fw), (lab, mk) in series.items():
            p = [r for r in pts if r["model"] == m and r["firmware"] == fw and r["job"] == "A"]
            if p:
                ax.errorbar([r["board_ms_mean"] for r in p], [r["pc_int8_mean"] for r in p],
                            yerr=[r["pc_int8_std"] for r in p], fmt=mk, capsize=3, label=lab)
                for r in p:
                    ax.annotate(f"h{r['hidden']}" + (f" T{r['T']}" if r["T"] != "" else ""),
                                (r["board_ms_mean"], r["pc_int8_mean"]), fontsize=7,
                                xytext=(3, 3), textcoords="offset points")
        a_pts = sorted((r for r in pts if r["job"] == "A"), key=lambda r: r["board_ms_mean"])
        front, best = [], -1.0
        for r in a_pts:
            if r["pc_int8_mean"] > best:
                front.append(r)
                best = r["pc_int8_mean"]
        if front:
            ax.step([r["board_ms_mean"] for r in front], [r["pc_int8_mean"] for r in front],
                    where="post", color="k", lw=1, alpha=.6, label="Pareto front")
        ax.set_xlabel("board latency [ms/beat] (5 000-beat DS2 subset)")
        ax.set_ylabel("PC int8 macro-F1, all DS2 (mean ± std, 3 seeds)")
        ax.set_title("ECG: macro-F1 vs board latency")
        ax.legend(fontsize=8)
        done(fig, "fig1_ecg_f1_vs_latency",
             [dict(r, pareto=int(r in front)) for r in pts if r["job"] == "A"])

    # 2. MNIST: accuracy and latency vs T, ANN as reference
    snn = [r for r in rows if r["workload"] == "mnist" and r["model"] == "SNN"
           and r["job"] in ("B", "ref")]
    if snn:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        p = sorted({(r["T"], r["pc_int8_mean"], r["pc_int8_std"]) for r in snn})
        a1.errorbar([x[0] for x in p], [x[1] * 100 for x in p], yerr=[x[2] * 100 for x in p],
                    fmt="o-", capsize=3, label="SNN (PC int8, 10k)")
        ann = [r for r in rows if r["workload"] == "mnist" and r["model"] == "ANN" and r["job"] == "ref"]
        for r in ann[:1]:
            a1.axhline(r["pc_int8_mean"] * 100, color="k", ls="--", label="ANN (published)")
        a1.set_xlabel("timesteps T")
        a1.set_ylabel("test accuracy [%]")
        a1.legend(fontsize=8)
        for fw, mk in (("fast", "o-"), ("hw", "^-")):
            q = sorted((r["T"], r["board_ms_mean"]) for r in snn
                       if r["firmware"] == fw and r.get("board_ms_mean", "") != "")
            if q:
                a2.plot([x[0] for x in q], [x[1] for x in q], mk, label=f"SNN {fw}")
        for r in [r for r in ann if r["firmware"] == "fast" and r.get("board_ms_mean", "") != ""][:1]:
            a2.axhline(r["board_ms_mean"], color="k", ls="--", label="ANN fast (published)")
        a2.set_xlabel("timesteps T")
        a2.set_ylabel("board latency [ms/image] (first 2 000 images)")
        a2.legend(fontsize=8)
        fig.suptitle("MNIST: accuracy and latency vs T")
        done(fig, "fig2_mnist_vs_T", snn + ann)

    # 3. streaming: macro-F1 vs latency, GRU 16/32/64 vs RSNN
    st = [r for r in rows if r["workload"] == "ecg-stream" and r.get("board_ms_mean", "") != ""]
    if st:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in st:
            lab = f"{r['model']} h{r['hidden']} {r['firmware']}"
            ax.errorbar(r["board_ms_mean"], r["pc_int8_mean"], yerr=r["pc_int8_std"],
                        fmt="o" if r["model"] == "GRU" else "s", capsize=3)
            ax.annotate(lab, (r["board_ms_mean"], r["pc_int8_mean"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("board latency [ms/beat] (log)")
        ax.set_ylabel("PC int8 macro-F1, all DS2 (mean ± std, 3 seeds)")
        ax.set_title("Streaming ECG: GRU 16 / 32 / 64 vs RSNN")
        done(fig, "fig3_stream_f1_vs_latency", st)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--res", default=os.path.join(PUB, "overnight_2026-10-12"))
    p.add_argument("--mode", default="full", choices=["full", "smoke"])
    p.add_argument("--job", default=None)
    a = p.parse_args()
    sub = "overnight_smoke" if a.mode == "smoke" else "overnight"
    os.makedirs(a.res, exist_ok=True)
    rows = collect(sub)
    if a.job:
        jr = [r for r in rows if r["job"] == a.job]
        if a.job == "E":
            jr = [r for r in rows if r["job"] == "A" and r.get("cv_macro_f1", "") != ""]
        with open(os.path.join(a.res, f"job_{a.job}.md"), "w", encoding="utf-8") as f:
            f.write(f"# overnight job {a.job} ({a.mode})\n\n" + fmt_table(jr) + "\n")
        print(f"job {a.job}: {len(jr)} rows -> job_{a.job}.md")
        return
    n_b = next((int(r["board_n_per_seed"].split("/")[0]) for r in rows
                if r["job"] == "B" and r.get("board_n_per_seed")), 2000)
    allrows = rows + references(n_b, None)
    write_csv(os.path.join(a.res, "summary.csv"), allrows)
    mism = [r for r in allrows if r.get("bit_exact") and
            int(r["bit_exact"].split("/")[0]) != int(r["bit_exact"].split("/")[1])]
    idx = {open(f, "rb").read() for f in glob.glob(os.path.join(
        HERE, "ann_vs_snn_ecg", "pynq", f"export_{sub}", "ecgA_*", "sample_idx.bin"))}
    subset = ("no A exports yet" if not idx else "identical in every A export"
              if len(idx) == 1 else f"**{len(idx)} DIFFERENT subsets**")
    timing = os.path.join(a.res, "timing.csv")
    last = {}                                  # last outcome of every step
    if os.path.exists(timing):
        for r in csv.DictReader(open(timing)):
            last[r["step"]] = r["status"]
    failed = [s for s, v in last.items() if v != "ok"]
    text = [f"# Overnight accuracy-vs-cost sweeps ({a.mode})", "",
            "Generated by `code/summarize_overnight.py` from the results of "
            "`code/overnight.sh`. Numbers only; no conclusions.", "",
            "## Held fixed (only the swept variable changes)", "",
            "- ECG (jobs A, E): final configuration of "
            "`docs/results/ann_vs_snn_ecg/pc/manifest_final.json` (10 epochs, batch 64, "
            "lr 0.001, beta 0.9, direct encoding, RR features, diff morphology, "
            "augmentation, weight decay 0.2); input scale fixed to the published "
            "s_x; swept: hidden size (ANN, SNN) and T (SNN). Board: one seeded "
            "5 000-beat DS2 subset (seed 0, `sample_idx.bin` in every export); "
            "PC metrics on all 49 273 DS2 beats. E: 5 patient folds on DS1, seed 0.",
            "- MNIST (B, D): `train_mnist.py` defaults (5 epochs, hidden 128, beta "
            "0.9); swept: T (B). Board: first 2 000 test images (B), full test set (D).",
            "- Streaming GRU (C): the tuning configs `runs/cv/gru_h16_bins3_e10` / "
            "`gru_h32_bins3_e10` (batch 256, lr 0.003, 10 epochs, 3 readout bins), "
            "trained --final. The published GRU-64 used 20 epochs. Board: 200 "
            "seeded DS2 beats; the published GRU-64 / RSNN points are on their "
            "1 000-beat board subset.",
            "- Training ran 6 processes in parallel with 3 torch threads each "
            "(the published models used other thread counts), so retrained "
            "configurations that also exist in the published results can differ "
            "slightly from them.", "",
            "## Checks", "",
            f"- board == PC int8 reference: "
            + ("all board subsets bit-exact" if not mism else
               "**MISMATCH in: " + ", ".join(f"{r['source']} {r['firmware']}" for r in mism) + "**"),
            f"- ECG board subset (`sample_idx.bin`): {subset}",
            f"- failed steps: {', '.join(failed) if failed else 'none'}", "",
            "## All points", "", fmt_table(allrows), "",
            "Board ms: mean of the per-seed means; p5 / p95 pooled over the seeds' samples. "
            "PC: mean ± std over seeds. Figures: fig1–3 (*.png, data in the *.csv "
            "next to them)."]
    with open(os.path.join(a.res, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    try:
        figures(a.res, allrows)
    except ImportError as e:
        print(f"no figures: {e}")
    print(f"{len(allrows)} rows -> {a.res}/summary.md, summary.csv")


if __name__ == "__main__":
    main()
