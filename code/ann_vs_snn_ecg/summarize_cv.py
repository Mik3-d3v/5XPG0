#!/usr/bin/env python3
"""summarize_cv.py — one table over all tagged CV runs (runs/cv/<tag>.json).

Every row is one tuning variant and model: pooled out-of-fold metrics on DS1,
mean +- std over seeds, plus the accuracy on the bundle-branch-block records
(BBB_RECORDS) and on 108 / 203, where the baseline fails.

Run:  python summarize_cv.py [tag-glob ...]  -> prints + writes runs/cv/ALL.md
"""

import glob
import json
import os
import sys

import numpy as np

from train_ecg import BBB_RECORDS, OUT_DIR

HARD = list(BBB_RECORDS) + [108, 203]
KEYS = ["macro_f1", "accuracy", "se_N", "se_S", "se_V", "ppv_S", "ppv_V"]


def ms(v, f="{:.3f}"):
    return f"{f.format(np.mean(v))} ± {f.format(np.std(v))}"


def main():
    pats = sys.argv[1:] or ["*"]
    files = sorted({f for p in pats
                    for f in glob.glob(os.path.join(OUT_DIR, "cv", p + ".json"))})
    head = ["variant", "model", "epochs", "wd", "seeds"] + KEYS + \
           [f"acc {r}" for r in HARD]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for f in files:
        d = json.load(open(f))
        a = d["args"]
        for name, runs in d["results"].items():
            row = [os.path.basename(f)[:-5], name, str(a["epochs"]),
                   str(a["weight_decay"]), str(len(runs))]
            row += [ms([r[k] for r in runs]) for k in KEYS]
            row += [f"{np.mean([r['per_record'][str(rec)]['accuracy'] for r in runs]):.2f}"
                    for rec in HARD]
            lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    print(table)
    if pats == ["*"]:
        with open(os.path.join(OUT_DIR, "cv", "ALL.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(table + "\n")


if __name__ == "__main__":
    main()
