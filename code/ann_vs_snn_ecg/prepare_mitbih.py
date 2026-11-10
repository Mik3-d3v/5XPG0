#!/usr/bin/env python3
"""prepare_mitbih.py - MIT-BIH Arrhythmia DB -> beat dataset (inter-patient split).

Source : PhysioNet MIT-BIH Arrhythmia Database (48 records, 360 Hz, 2 leads).
         Downloaded once with wfdb (~100 MB), or point --data-dir at any folder
         that already holds the raw 100.dat / 100.hea / 100.atr ... files
         (e.g. a Kaggle copy of the raw database).

Pipeline (every step is a design choice - document it in the report):
  1. lead      : MLII (the channel is looked up by name; record 114 is swapped)
  2. filter    : 0.5-40 Hz Butterworth band-pass, zero-phase (filtfilt)
  3. resample  : 360 Hz -> 180 Hz (decimate by 2) to halve the input width
  4. segment   : fixed window around each annotated R-peak
  5. labels    : AAMI EC57 classes N / S / V  (F and Q dropped, paced records
                 excluded). --merge-f instead maps fusion beats F -> V.
  6. normalise : per-beat z-score (removes patient-specific gain / offset)
  7. RR feats  : pre-RR, post-RR, running-mean RR (causal) -> rhythm information
  8. template  : patient-relative morphology. For every beat, the sample-wise
                 median of the z-scored windows of the previous TMPL_BEATS
                 beats of the SAME record (causal, every annotated beat counts,
                 the labels are not used). The first beat of a record has no
                 history: its template is the beat itself.
  9. split     : de Chazal inter-patient DS1 (train) / DS2 (test). No patient is
                 in both sets - a random beat-level split leaks patients and
                 inflates accuracy to ~99 %.

Run:    python prepare_mitbih.py                # downloads to data/mitdb
Output: data/mitbih_ds1.npz, data/mitbih_ds2.npz
        keys: x [N, L] float32, tmpl [N, L] float32, rr [N, 3] float32,
              y [N] int64, rec [N] int32
"""

import argparse
import os

import numpy as np
import wfdb
from scipy import signal

# ---------------------------------------------------------------- constants
FS_RAW = 360
DECIM = 2                       # 360 -> 180 Hz
FS = FS_RAW // DECIM
PRE_S, POST_S = 0.25, 0.45      # window around R: 250 ms before, 450 ms after

# de Chazal et al., IEEE TBME 2004. Paced records 102, 104, 107, 217 excluded.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

CLASSES = ["N", "S", "V"]
AAMI = {s: 0 for s in "NLRej"}
AAMI.update({s: 1 for s in "AaJS"})
AAMI.update({s: 2 for s in "VE"})
# F (fusion of normal + ventricular) is <1 % of beats, concentrated in a couple
# of records: dropped by default, or merged into V with --merge-f.
# symbols not in AAMI ('/', 'f', 'Q', rhythm/noise marks, ...) are skipped

RR_AVG_BEATS = 10               # running mean over the previous N RR intervals
TMPL_BEATS = 32                 # template = median of the previous N beats


def bandpass(x, fs=FS_RAW, lo=0.5, hi=40.0, order=3):
    b, a = signal.butter(order, [lo, hi], btype="band", fs=fs)
    return signal.filtfilt(b, a, x)


def load_record(rec, data_dir):
    path = os.path.join(data_dir, str(rec))
    r = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    lead = r.sig_name.index("MLII")
    return r.p_signal[:, lead], np.asarray(ann.sample), list(ann.symbol)


def beats_from_record(rec, data_dir):
    sig, r_idx, sym = load_record(rec, data_dir)
    sig = bandpass(sig)
    sig = signal.decimate(sig, DECIM, zero_phase=True)
    r_idx = r_idx // DECIM

    # keep only beat annotations (any symbol that is a heartbeat, AAMI or not)
    # so RR intervals are computed on the true beat sequence
    beat_syms = set("NLRejAaJSVEFf/Q")
    keep = [i for i, s in enumerate(sym) if s in beat_syms]
    r_idx = r_idx[keep]
    sym = [sym[i] for i in keep]

    rr = np.diff(r_idx) / FS                      # seconds, len = n_beats - 1
    pre, post = int(PRE_S * FS), int(POST_S * FS)

    xs, ts, rrs, ys = [], [], [], []
    past = []                                     # z-scored windows, any beat
    for i in range(len(r_idx)):
        c = r_idx[i]
        if c - pre < 0 or c + post > len(sig):
            continue
        w = sig[c - pre:c + post]
        w = ((w - w.mean()) / (w.std() + 1e-6)).astype(np.float32)
        s = sym[i]
        # need a previous and next beat for the RR features
        if 1 <= i < len(r_idx) - 1 and s in AAMI:
            pre_rr, post_rr = rr[i - 1], rr[i]
            hist = rr[max(0, i - RR_AVG_BEATS):i]  # causal: past intervals only
            avg_rr = hist.mean()
            # causal template: previous beats only, labels not used
            t = np.median(past[-TMPL_BEATS:], axis=0) if past else w

            xs.append(w)
            ts.append(t.astype(np.float32))
            rrs.append([pre_rr, post_rr, avg_rr])
            ys.append(AAMI[s])
        past.append(w)
    return xs, ts, rrs, ys


def build(records, data_dir):
    X, T, RR, Y, R = [], [], [], [], []
    for rec in records:
        xs, ts, rrs, ys = beats_from_record(rec, data_dir)
        X += xs; T += ts; RR += rrs; Y += ys; R += [rec] * len(ys)
        counts = np.bincount(ys, minlength=len(CLASSES))
        print(f"  {rec}: {len(ys):5d} beats  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    return (np.stack(X), np.stack(T), np.asarray(RR, np.float32),
            np.asarray(Y, np.int64), np.asarray(R, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/mitdb")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--merge-f", action="store_true",
                    help="label fusion beats (F) as V instead of dropping them")
    args = ap.parse_args()
    if args.merge_f:
        AAMI["F"] = 2

    if not os.path.exists(os.path.join(args.data_dir, "100.hea")):
        print(f"downloading MIT-BIH from PhysioNet to {args.data_dir} ...")
        wfdb.dl_database("mitdb", dl_dir=args.data_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, recs in (("ds1", DS1), ("ds2", DS2)):
        print(f"{name.upper()} ({len(recs)} records)")
        x, tmpl, rr, y, rec = build(recs, args.data_dir)
        out = os.path.join(args.out_dir, f"mitbih_{name}.npz")
        np.savez_compressed(out, x=x, tmpl=tmpl, rr=rr, y=y, rec=rec,
                            classes=np.array(CLASSES), fs=FS)
        counts = np.bincount(y, minlength=len(CLASSES))
        print(f"-> {out}: x{list(x.shape)}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)) + "\n")


if __name__ == "__main__":
    main()