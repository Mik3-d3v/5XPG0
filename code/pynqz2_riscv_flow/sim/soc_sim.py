#!/usr/bin/env python3
"""soc_sim.py — run a firmware image on exported samples in the RTL of the
whole SoC (tb_soc.v + rtl/, Vivado xsim), without the board.

The CPU boots once with the firmware and model.bin in BRAM; per sample the
bench writes the input and posts MB_CMD_RUN, as run_mnist_pynq.py does on
the board. Every reply (pred, cycles, aux, err) and output vector is checked
against the PC int8 reference of the export (ref_*.bin) and written to a CSV
in the board format (board_results_*.csv columns that the RTL can give).
The SoC is cycle-accurate RTL (the BRAM wait states are in spike_soc.v), so
a level that also ran on the board gives the board's cycle count exactly;
--board-csv compares with such a run, sample by sample.

  python soc_sim.py --fw <firmware.bin> --defines SNN_COPROC MAC_COPROC \\
      --export <export dir> --random 200 --csv out.csv [--board-csv x.csv.gz]

xsim does not like spaces in paths: the work directory (--work, default
C:/vb/soc_sim/<defines>) gets copies of the RTL and the bench. Use one
work directory per parallel run.
"""

import argparse
import csv
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "..", "rtl")
VIV = r"C:\AMDDesignTools\2026.1\Vivado\bin"
MODEL_BASE = 0x13000
CSV_COLS = ["idx", "label", "board_pred", "pc_int_pred", "outputs_match", "cycles",
            "hidden_events", "out_events", "outputs", "ref_cycles", "ref_outputs_match"]


def run(cmd, cwd):
    r = subprocess.run(["cmd", "/c", f"cd /d {cwd} && {cmd}"], capture_output=True, text=True)
    return r.stdout + r.stderr


def compile_soc(work, defines):
    os.makedirs(work, exist_ok=True)
    srcs = ["picorv32.v", "poisson.v", "snn_pcpi.v", "mac_pcpi.v", "spike_soc.v"]
    for f in srcs:
        shutil.copy(os.path.join(RTL, f), work)
    shutil.copy(os.path.join(HERE, "tb_soc.v"), work)
    d = " ".join(f"-d {x}" for x in defines)
    log = run(f"{VIV}\\xvlog.bat {d} {' '.join(srcs)} tb_soc.v", work)
    log += run(f"{VIV}\\xelab.bat tb -s tb -O3", work)
    if "ERROR" in log:
        sys.exit(log)


def hexwords(b):
    b = b + bytes(-len(b) % 4)
    return "".join(f"{int.from_bytes(b[i:i + 4], 'little'):08x}\n" for i in range(0, len(b), 4))


def open_csv(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fw", required=True, help="firmware .bin")
    p.add_argument("--export", required=True, help="export dir (meta.json, model.bin, ...)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--idx", type=int, nargs="+", help="sample indices")
    g.add_argument("--random", type=int, help="this many random samples (fixed seed)")
    p.add_argument("--rng-seed", type=int, default=1)
    p.add_argument("--T", type=int, default=None, help="override T (default: meta timesteps)")
    p.add_argument("--defines", nargs="*", default=[])
    p.add_argument("--board-csv", default=None,
                   help="board_results csv(.gz) of the same export to compare with")
    p.add_argument("--csv", default=None, help="write the per-sample results here")
    p.add_argument("--work", default=None)
    a = p.parse_args()

    work = a.work or "C:/vb/soc_sim/" + ("_".join(a.defines) or "stock")
    compile_soc(work, a.defines)
    meta = json.load(open(os.path.join(a.export, "meta.json")))
    nb, n_out, N = meta["input_bytes"], meta["n_out"], meta["n_samples"]
    T = a.T if a.T is not None else meta["timesteps"]
    is_snn = meta["model"] in ("SNN", "RSNN")
    idx = (a.idx if a.idx else
           sorted(random.Random(a.rng_seed).sample(range(N), a.random)) if a.random else [0])

    # BRAM image: firmware + model; inputs; parameters
    mem = bytearray(256 * 1024)
    fw = open(a.fw, "rb").read()
    model = open(os.path.join(a.export, "model.bin"), "rb").read()
    mem[:len(fw)] = fw
    mem[MODEL_BASE:MODEL_BASE + len(model)] = model
    with open(os.path.join(work, "image.hex"), "w") as f:
        f.write(hexwords(bytes(mem)))
    W = (nb + 3) // 4
    with open(os.path.join(a.export, "inputs.bin"), "rb") as fin, \
            open(os.path.join(work, "inputs.hex"), "w") as f:
        for i in idx:
            fin.seek(i * nb)
            f.write(hexwords(fin.read(nb)))
    with open(os.path.join(work, "params.hex"), "w") as f:
        f.write(f"{len(idx):08x}\n{W:08x}\n{T:08x}\n")

    t0 = time.time()
    out = run(f"{VIV}\\xsim.bat tb -R", work)
    if "DONE" not in out:
        print(out[-3000:])
        sys.exit("simulation did not finish")

    ref = open(os.path.join(a.export, meta["ref_file"]), "rb").read()
    labels = open(os.path.join(a.export, "labels.bin"), "rb").read()
    board = {}
    if a.board_csv:
        with open_csv(a.board_csv) as f:
            board = {int(r["idx"]): r for r in csv.DictReader(f)}
    rows = []
    for m in re.finditer(r"S (\d+) RESP (\d+) (\d+) (\d+) (\d+) OUT((?: -?\d+)*)", out):
        s, pred, cyc, aux, err = map(int, m.groups()[:5])
        i = idx[s]
        outs = [int(v) for v in m.group(6).split()][:n_out]
        r = [int.from_bytes(ref[4 * (i * n_out + k):4 * (i * n_out + k + 1)], "little",
                            signed=True) for k in range(n_out)]
        row = {"idx": i, "label": labels[i], "board_pred": pred,
               "pc_int_pred": r.index(max(r)),
               "outputs_match": int(outs == r and err == 0) if T == meta["timesteps"] else "",
               "cycles": cyc,
               "hidden_events": aux - sum(outs) if is_snn and meta["model"] == "SNN" else aux,
               "out_events": sum(outs) if meta["model"] == "SNN" else 0,
               "outputs": " ".join(map(str, outs)), "ref_cycles": "", "ref_outputs_match": ""}
        if i in board:
            b = board[i]
            row["ref_cycles"] = int(b["cycles"])
            row["ref_outputs_match"] = int(b["outputs"].split() == row["outputs"].split())
        rows.append(row)
    if len(rows) != len(idx):
        sys.exit(f"{len(rows)} results for {len(idx)} samples")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, CSV_COLS)
            w.writeheader()
            w.writerows(rows)
    ok = sum(1 for r in rows if r["outputs_match"] == 1)
    cyc = [r["cycles"] for r in rows]
    msg = (f"{meta['model']} {os.path.basename(a.export.rstrip('/'))} {os.path.basename(a.fw)}: "
           f"{len(rows)} samples, == PC int8 ref {ok}/{len(rows)}, "
           f"mean {sum(cyc) / len(cyc) / 1e5:.3f} ms ({sum(cyc) / len(cyc):,.0f} cycles)")
    rc = [r for r in rows if r["ref_cycles"] != ""]
    if rc:
        msg += (f" | vs board csv: mean {sum(r['ref_cycles'] for r in rc) / len(rc) / 1e5:.3f} ms, "
                f"same cycles {sum(r['ref_cycles'] == r['cycles'] for r in rc)}/{len(rc)}, "
                f"same outputs {sum(r['ref_outputs_match'] for r in rc)}/{len(rc)}")
    print(msg + f"  [{time.time() - t0:.0f} s]", flush=True)
    sys.exit(0 if T != meta["timesteps"] or ok == len(rows) else 1)


if __name__ == "__main__":
    main()
