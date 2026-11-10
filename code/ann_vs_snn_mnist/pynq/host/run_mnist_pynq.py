#!/usr/bin/env python3
"""run_mnist_pynq.py — run an exported ANN or SNN (MNIST or ECG) on the PicoRV32 (PYNQ-Z2).

Runs ON THE PYNQ-Z2 as root with the XRT environment (deploy_pynq.sh does
this). Needs spike_pynq.py of the bring-up flow next to it; stdlib only.

  1. load the bitstream (unchanged spike_top.bit of pynqz2_riscv_flow),
  2. write the firmware image with the CPU held in reset,
  3. write model.bin to MODEL_BASE, release the CPU, print its console,
  4. per test sample: read its input from inputs.bin (streamed, one sample
     at a time; meta.json "input_kind": "spikes" = rate-coded bitmaps,
     "u8" / "s8" = input bytes), write it to INPUT_BASE,
     send MB_CMD_RUN, read the outputs from RESULT_BASE (SNN: output spike
     counts, ANN: int32 logits),
  5. compare with the labels and with the PC integer reference
     (ref_counts.bin / ref_logits.bin: the board must match it exactly).

Every sample is appended to <export_dir>/board_results_<fw>.csv as soon as
it is done, so a long run can be resumed with --resume after an
interruption. At the end the summary is printed and written to
<export_dir>/board_summary_<fw>.json. <fw> is "fast" (snn_mnist_fast.bin,
default) or "baseline" (snn_mnist.bin); both images run both model types.
"hw" adds the SNN coprocessor (spike_top_snn.bit); "muriscv" adds ANN3
models on muRISCV-NN (the *_muriscv exports of muriscv_ref.py); "mac" is
"hw" plus the ANN MAC coprocessor (snn_mnist_mac.bin, spike_top_both.bit).
--tag X writes board_results_<fw>X.csv / board_summary_<fw>X.json instead,
e.g. for regression runs that must not overwrite the published results.

CSV columns (events: SNN spikes, ANN nonzero values):
  in_events      SNN input spikes          ANN / direct SNN: nonzero inputs
  in_changes     SNN input on/off changes  ANN / direct SNN: = in_events
  hidden_events  SNN hidden spikes         ANN nonzero hidden activations
  out_events     SNN output spikes         ANN 0
  outputs        SNN spike counts          ANN logits

Usage:  run_mnist_pynq.py <export_dir> [--fw fast|baseline|hw|muriscv|mac] [--n N] [--resume]
                         [--no-bit] [--tag X]
BRAM layout constants mirror firmware/snn_layout.h — keep in sync.
"""

import argparse
import csv
import json
import os
import sys
import time
from array import array

from spike_pynq import MAILBOX_BASE, Bram, Regs, load_bit

RESULT_BASE = 0x10800
INPUT_BASE = 0x11000
INPUT_MAX_BYTES = 0x2000
MODEL_BASE = 0x13000
MODEL_MAX_BYTES = 0x3F000 - MODEL_BASE
MB_CMD_ECHO = 1
MB_CMD_RUN = 16
CLK_HZ = 100_000_000
FIRMWARE = {"fast": "snn_mnist_fast.bin", "baseline": "snn_mnist.bin",
            "hw": "snn_mnist_hw.bin", "muriscv": "snn_mnist_muriscv.bin",
            "mac": "snn_mnist_mac.bin"}
# hw: SNN kernel on the SNN coprocessor (custom instructions) -> its bitstream
# muriscv: fast + ANN3 models on the unmodified muRISCV-NN library (exports of
# muriscv_ref.py), plain rv32im -> the unchanged bitstream
# mac: hw + ANN / GRU / SNN first layer on the MAC coprocessor (custom-1)
# -> the bitstream with both coprocessors
BITSTREAM = {"fast": "spike_top.bit", "baseline": "spike_top.bit",
             "hw": "spike_top_snn.bit", "muriscv": "spike_top.bit",
             "mac": "spike_top_both.bit"}

CSV_COLS = ["idx", "label", "board_pred", "pc_int_pred", "pc_float_pred",
            "outputs_match", "cycles", "in_events", "in_changes",
            "hidden_events", "out_events", "host_ms", "outputs"]


def mb_call(bram, opcode, args=(), timeout=30.0):
    """Mailbox command (like Bram.mb_cmd, with a longer timeout)."""
    a = (list(args) + [0, 0, 0])[:3]
    seq = bram.rd32(MAILBOX_BASE)
    bram.wr32(MAILBOX_BASE + 8, opcode)
    for i, v in enumerate(a):
        bram.wr32(MAILBOX_BASE + 12 + 4 * i, v)
    bram.wr32(MAILBOX_BASE, seq + 1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if bram.rd32(MAILBOX_BASE + 4) == seq + 1:
            return [bram.rd32(MAILBOX_BASE + 24 + 4 * i) for i in range(4)]
        time.sleep(0.0005)
    raise TimeoutError("mailbox command timed out - CPU not running?")


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def popcount(b):
    return bin(int.from_bytes(b, "little")).count("1")


def input_kind(meta):
    """"spikes" (SNN2 bitmaps), "u8" or "s8" (input bytes); old exports
    have no "input_kind": MNIST SNN spikes / ANN pixels."""
    return meta.get("input_kind", "spikes" if meta["model"] == "SNN" else "u8")


def input_stats(x, meta):
    """(in_events, in_changes) of one sample."""
    if input_kind(meta) != "spikes":
        zero = meta.get("input_zero", 0) & 0xFF   # ANN3: pixel 0 is int8 -128
        nz = sum(1 for v in x[:meta["n_in"]] if v != zero)
        return nz, nz
    step = 4 * meta["words_per_step"]
    prev = 0
    ev = ch = 0
    for t in range(meta["timesteps"]):
        cur = int.from_bytes(x[t * step:(t + 1) * step], "little")
        ev += bin(cur).count("1")
        ch += bin(cur ^ prev).count("1")
        prev = cur
    return ev, ch


def summarize(rows, meta, wall_s):
    n = len(rows)
    col = lambda k: [int(r[k]) for r in rows]
    mean = lambda k: sum(col(k)) / n
    label, pred, ref, fpred = (col(k) for k in
                               ("label", "board_pred", "pc_int_pred", "pc_float_pred"))
    cycles = col("cycles")
    ms = sorted(c / CLK_HZ * 1e3 for c in cycles)
    fan = [l["n_out"] for l in meta["layers"]]
    s = {
        "model": meta["model"], "checkpoint": meta["checkpoint"], "n_samples": n,
        "timesteps": meta["timesteps"], "clk_hz": CLK_HZ,
        "board_accuracy": sum(p == y for p, y in zip(pred, label)) / n,
        "pc_float_accuracy_same_samples": sum(p == y for p, y in zip(fpred, label)) / n,
        "board_eq_pc_int_pred": sum(p == r for p, r in zip(pred, ref)),
        "board_eq_pc_int_outputs": sum(col("outputs_match")),
        "board_float_pred_agree": sum(p == f for p, f in zip(pred, fpred)),
        "cycles_mean": sum(cycles) / n,
        "latency_ms_mean": sum(ms) / n,
        "latency_ms_min": ms[0], "latency_ms_median": ms[n // 2],
        "latency_ms_max": ms[-1],
        "in_events_mean": mean("in_events"),
        "in_changes_mean": mean("in_changes"),
        "hidden_events_mean": mean("hidden_events"),
        "out_events_mean": mean("out_events"),
        "host_ms_mean": sum(float(r["host_ms"]) for r in rows) / n,
        "wall_time_s_this_session": wall_s,
        "pc_test_accuracy_float_full": meta["pc_test_accuracy_float"],
        "pc_test_accuracy_int_full": meta["pc_test_accuracy_int"],
    }
    if len(fan) == 2 and input_kind(meta) != "spikes" and meta["model"] == "SNN":
        # direct encoding: dense first-layer MACs once + SOPs of the hidden spikes
        s["ops_benchmark_mean"] = meta["n_in"] * fan[0] + mean("hidden_events") * fan[1]
        s["ops_executed_fast_mean"] = mean("in_events") * fan[0] + mean("hidden_events") * fan[1]
    elif len(fan) == 2:   # hidden events are not split per layer
        if meta["model"] == "SNN":
            # benchmark.py: one add per spike per fan-out synapse
            s["ops_benchmark_mean"] = mean("in_events") * fan[0] + mean("hidden_events") * fan[1]
        else:
            # benchmark.py: dense MACs
            s["ops_benchmark_mean"] = sum(l["n_in"] * l["n_out"] for l in meta["layers"])
        # synaptic updates the fast firmware really performs
        s["ops_executed_fast_mean"] = mean("in_changes") * fan[0] + mean("hidden_events") * fan[1]
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("export_dir")
    p.add_argument("--n", type=int, default=0, help="samples to run (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="continue after the last sample in board_results_<fw>.csv")
    p.add_argument("--bit", default=None,
                   help="bitstream (default: spike_top.bit, hw: spike_top_snn.bit)")
    p.add_argument("--fw", choices=sorted(FIRMWARE), default="fast")
    p.add_argument("--no-bit", action="store_true",
                   help="skip the bitstream download (already loaded)")
    p.add_argument("--tag", default="",
                   help="suffix of the result files (board_results_<fw><tag>.csv)")
    a = p.parse_args()

    d = a.export_dir
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    if "input_bytes" not in meta:
        sys.exit("old export format - re-run export_model.py")
    is_snn = meta["model"] == "SNN"
    T, n_out = meta["timesteps"], meta["n_out"]
    sample_bytes = meta["input_bytes"]
    n = meta["n_samples"] if a.n <= 0 else min(a.n, meta["n_samples"])

    model = read_file(os.path.join(d, "model.bin"))
    labels = read_file(os.path.join(d, "labels.bin"))
    fpred = read_file(os.path.join(d, "float_pred.bin"))
    ref = array("i", read_file(os.path.join(d, meta["ref_file"])))
    if sys.byteorder != "little":
        ref.byteswap()
    if len(model) > MODEL_MAX_BYTES or sample_bytes > INPUT_MAX_BYTES:
        sys.exit("model or input does not fit the BRAM layout")

    csv_path = os.path.join(d, f"board_results_{a.fw}{a.tag}.csv")
    rows = []
    if a.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"resuming after {len(rows)} samples")
    start = len(rows)

    # 1-3: bitstream, firmware, model
    if not a.no_bit:
        load_bit(a.bit or BITSTREAM[a.fw])
    regs, bram = Regs(), Bram()
    regs.ctrl_reset(True)
    time.sleep(0.01)
    bram.write_image(read_file(FIRMWARE[a.fw]))
    bram.m[MODEL_BASE:MODEL_BASE + len(model)] = model
    regs.ctrl_reset(False)
    time.sleep(0.2)
    sys.stdout.write(bram.console_read().decode("ascii", "replace"))
    if mb_call(bram, MB_CMD_ECHO, [0xC0FFEE, 1, 2], timeout=2.0)[0] != 0xC0FFEE:
        sys.exit("mailbox echo failed")

    print(f"firmware {FIRMWARE[a.fw]}, {meta['model']} {meta['checkpoint']}"
          f"{f'  T={T}' if is_snn else ''}  running samples {start}..{n - 1} "
          f"(PC: float {meta['pc_test_accuracy_float'] * 100:.2f}%, "
          f"int8 {meta['pc_test_accuracy_int'] * 100:.2f}% on the full test set)",
          flush=True)

    # 4: samples — inputs streamed from the file, results appended per sample
    t_start = time.time()
    with open(os.path.join(d, "inputs.bin"), "rb") as fin, \
            open(csv_path, "a" if start else "w", newline="") as fout:
        w = csv.DictWriter(fout, CSV_COLS)
        if not start:
            w.writeheader()
        fin.seek(start * sample_bytes)
        for i in range(start, n):
            t0 = time.perf_counter()
            x = fin.read(sample_bytes)
            bram.m[INPUT_BASE:INPUT_BASE + sample_bytes] = x
            pred, cycles, aux, err = mb_call(bram, MB_CMD_RUN, [T])
            if err:
                sys.exit(f"sample {i}: firmware error {err} (see snn_layout.h)")
            outs = [bram.rd32(RESULT_BASE + 4 * o) for o in range(n_out)]
            if not is_snn:   # int32 logits
                outs = [v - (1 << 32) if v >= 1 << 31 else v for v in outs]
            host_ms = (time.perf_counter() - t0) * 1e3 - cycles / CLK_HZ * 1e3
            r = list(ref[i * n_out:(i + 1) * n_out])
            ev, ch = input_stats(x, meta)
            row = {"idx": i, "label": labels[i], "board_pred": pred,
                   "pc_int_pred": r.index(max(r)), "pc_float_pred": fpred[i],
                   "outputs_match": int(outs == r), "cycles": cycles,
                   "in_events": ev, "in_changes": ch,
                   "hidden_events": aux - sum(outs) if is_snn else aux,
                   "out_events": sum(outs) if is_snn else 0,
                   "host_ms": f"{host_ms:.3f}",
                   "outputs": " ".join(map(str, outs))}
            w.writerow(row)
            fout.flush()
            rows.append(row)
            if (i + 1) % 500 == 0 or i + 1 == n:
                acc = sum(int(q["board_pred"]) == int(q["label"]) for q in rows)
                same = sum(int(q["outputs_match"]) for q in rows)
                el = time.time() - t_start
                eta = el / (i + 1 - start) * (n - i - 1)
                print(f"  [{i + 1:5d}/{n}] board acc {acc / len(rows) * 100:6.2f}%  "
                      f"== PC int ref {same}/{len(rows)}  "
                      f"ETA {eta / 60:5.1f} min", flush=True)

    # 5: summary
    s = summarize(rows, meta, time.time() - t_start)
    s["firmware"] = FIRMWARE[a.fw]
    with open(os.path.join(d, f"board_summary_{a.fw}{a.tag}.json"), "w") as f:
        json.dump(s, f, indent=2)
    ev = "spikes" if is_snn else "nonzero"
    print(f"\n=== {meta['model']}, {n} samples ===")
    print(f"board accuracy        {s['board_accuracy'] * 100:.2f}%")
    print(f"PC float accuracy     {s['pc_float_accuracy_same_samples'] * 100:.2f}%"
          f"  (same samples, same inputs)")
    print(f"board == PC int ref   pred {s['board_eq_pc_int_pred']}/{n}, "
          f"exact outputs {s['board_eq_pc_int_outputs']}/{n}")
    print(f"latency on PicoRV32   {s['cycles_mean']:,.0f} cycles = "
          f"{s['latency_ms_mean']:.2f} ms/sample @ {CLK_HZ // 1_000_000} MHz "
          f"(min {s['latency_ms_min']:.1f}, max {s['latency_ms_max']:.1f})")
    print(f"host overhead         {s['host_ms_mean']:.2f} ms/sample "
          f"(file read + AXI copy + mailbox)")
    print(f"{ev:9s} in/hid/out  {s['in_events_mean']:.1f} / "
          f"{s['hidden_events_mean']:.1f} / {s['out_events_mean']:.1f} per sample"
          + (f"  (input changes {s['in_changes_mean']:.1f})" if is_snn else ""))
    if "ops_benchmark_mean" in s:
        print(f"ops/sample            {s['ops_benchmark_mean']:,.0f} "
              f"({'SOPs' if is_snn else 'dense MACs'}, benchmark.py)"
              + (", all executed (dense kernel)" if a.fw == "muriscv" else
                 f", {s['ops_executed_fast_mean']:,.0f} executed by the fast kernel"))
    print(f"results -> {csv_path}, board_summary_{a.fw}{a.tag}.json")
    regs.ctrl_reset(True)


if __name__ == "__main__":
    main()
