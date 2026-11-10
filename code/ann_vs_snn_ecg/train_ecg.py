#!/usr/bin/env python3
"""train_ecg.py — ANN vs SNN heartbeat classification on MIT-BIH (N / S / V).

Mirrors train_mnist.py: same training loop, checkpoint format, manifest and
TensorBoard layout, so benchmark_ecg.py and benchmark.py produce comparable
output. Needs data/mitbih_ds1.npz + data/mitbih_ds2.npz (prepare_mitbih.py).

Trains TWO 2-layer MLPs under the SAME budget:
  * an ANN  (F -> H -> C, ReLU)
  * an SNN  (F -> H LIF -> C LIF, snnTorch)
F = 126 beat samples (+3 RR features with --use-rr), C = 3 classes.

PROTOCOL — three modes. Patients differ more than classes do, so every split
is by record (patient), never by beat:
  1. tuning  (default)  patient-level K-fold cross-validation on DS1.
       Every DS1 beat is predicted exactly once by a model that never saw its
       patient; metrics are computed on these pooled out-of-fold predictions
       (a per-fold F1 is meaningless for folds with almost no S beats).
       Folds are balanced by S/V content; the bundle-branch-block records
       (BBB_RECORDS: AAMI "N" but wide QRS, easily mistaken for V) are spread
       over different folds so no fold loses all examples of that shape.
       -> runs/cv_summary.md / .json, TensorBoard <model>_seed<s>_fold<k>.
       Use this to choose epochs, width, weight decay, encoding, T, beta ...
  2. final   (--final)    train on ALL of DS1 with the chosen settings, save
       checkpoints + runs/manifest.json. DS2 is only touched by
       benchmark_ecg.py, once, after tuning is finished.
  3. holdout (--val-records r1 r2 ...)  quick check on a fixed set of records.

What is held constant between the two models (per seed/fold):
  topology/width, optimiser (AdamW + cosine LR schedule), learning rate,
  weight decay, batch size, number of epochs, weight initialisation seed,
  data order (dedicated loader generator), class weights and the splits.
Models are kept after the LAST epoch (fixed budget, as for MNIST).

ECG-specific choices (vs MNIST):
  * class-weighted cross-entropy (N is ~90 % of the beats); weights are
    sqrt-inverse class frequency of the training split, stored in the config
  * macro-F1 and per-class Se / +P are logged next to accuracy
  * SNN input encoding (--encoding):
      direct : the analog feature vector is a constant input current every
               step -> first layer = one dense MAC pass (reused over T steps),
               the rest is spike-driven
      rate   : signed ON/OFF rate code, 2 channels per feature (positive /
               negative part), spike prob = clip(|v| / 3, 0, 1) per step ->
               fully spike-driven, input width doubles

Learning curves go to TensorBoard (runs/tb/<timestamp>/<run>):
  train/loss, train/accuracy, train/macro_f1, val/loss, val/accuracy,
  val/macro_f1, lr, epoch_time_s.

Run:   python train_ecg.py --models ANN --seeds 0 --use-rr      # fast tuning
       python train_ecg.py --use-rr --epochs 5 --weight-decay 0.05
       python train_ecg.py --use-rr --epochs 5 --weight-decay 0.05 --final
       python benchmark_ecg.py
View:  tensorboard --logdir runs/tb
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 20
SEEDS = (0, 1, 2)
BATCH = 64
LR = 1e-3
N_HIDDEN = 64
N_TIMESTEPS = 25      # SNN simulation steps
BETA = 0.9            # LIF membrane decay
ENCODING = "direct"
WEIGHT_DECAY = 0.0    # AdamW; 0.0 == plain Adam
CV_FOLDS = 5          # patient-level folds over the 22 DS1 records
# bundle branch block records in DS1: labelled N (AAMI) but QRS looks like V
BBB_RECORDS = (109, 118, 124, 207)
DATA_DIR = "data"
OUT_DIR = "runs"
MANIFEST = os.path.join(OUT_DIR, "manifest.json")


# ---------------------------------------------------------------- data
def load_components(split, data_dir=DATA_DIR):
    """-> comp [N, 2L+3] = [beat | template | RR features], y, rec, classes.

    The model input is built from these by make_features(); training keeps
    the components so augmentation can act on the beat before the features
    (e.g. beat - template) are formed.
    """
    d = np.load(os.path.join(data_dir, f"mitbih_{split}.npz"))
    x, rr = d["x"], d["rr"]
    tmpl = d["tmpl"] if "tmpl" in d else x      # old files: no template
    pre, post, avg = rr[:, 0], rr[:, 1], rr[:, 2]
    # ratios to the patient's own recent rhythm generalise across
    # patients; scaled to roughly the range of the z-scored beat
    feats = np.stack([(pre / avg - 1) * 3, (post / avg - 1) * 3,
                      (avg - 0.8) * 3], axis=1).astype(np.float32)
    comp = np.concatenate([x, tmpl, feats], axis=1)
    classes = [str(c) for c in d["classes"]]
    return torch.from_numpy(comp), torch.from_numpy(d["y"]), d["rec"], classes


def make_features(comp, use_rr, morph):
    """[N, 2L+3] components -> [N, F] model input.

    morph: raw  = the z-scored beat (L)
           diff = beat - causal patient template (L)
           both = beat and beat - template (2L)
    use_rr appends the 3 RR features.
    """
    L = (comp.shape[1] - 3) // 2
    beat, tmpl, rr = comp[:, :L], comp[:, L:2 * L], comp[:, 2 * L:]
    parts = {"raw": [beat], "diff": [beat - tmpl],
             "both": [beat, beat - tmpl]}[morph]
    if use_rr:
        parts = parts + [rr]
    return torch.cat(parts, dim=1)


def load_split(split, use_rr, data_dir=DATA_DIR, morph="raw"):
    """-> x [N, F] float32, y [N] int64, rec [N] (numpy), class names."""
    comp, y, rec, classes = load_components(split, data_dir)
    return make_features(comp, use_rr, morph), y, rec, classes


# light augmentation of the (already z-scored) beat; the template and the RR
# features are left alone, so beat - template sees the same perturbation
AUG_SCALE = 0.1       # amplitude x U(1 - s, 1 + s)
AUG_SHIFT = 3         # +- samples (at 180 Hz: +- 17 ms R-peak jitter)
AUG_NOISE = 0.05      # additive Gaussian noise, std
AUG_OFFSET = 0.1      # baseline offset U(-o, o)


def augment(comp, gen):
    """Randomly perturb the beat part of a [B, 2L+3] component batch."""
    B = comp.shape[0]
    L = (comp.shape[1] - 3) // 2
    beat = comp[:, :L]
    shift = torch.randint(-AUG_SHIFT, AUG_SHIFT + 1, (B, 1), generator=gen)
    idx = (torch.arange(L).unsqueeze(0) - shift).clamp(0, L - 1)
    beat = beat.gather(1, idx)                   # shift, edge-padded
    scale = 1 + AUG_SCALE * (2 * torch.rand(B, 1, generator=gen) - 1)
    offset = AUG_OFFSET * (2 * torch.rand(B, 1, generator=gen) - 1)
    noise = AUG_NOISE * torch.randn(B, L, generator=gen)
    beat = beat * scale + offset + noise
    return torch.cat([beat, comp[:, L:]], dim=1)


def split_by_records(x, y, rec, val_recs, classes, n_features):
    """Train / val datasets (val = the given records) + config metadata.

    x holds the components (load_components), features are built per batch.
    """
    m = torch.from_numpy(np.isin(rec, list(val_recs)))
    train = TensorDataset(x[~m], y[~m])
    val = TensorDataset(x[m], y[m])
    counts = torch.bincount(y[~m], minlength=len(classes)).float().clamp(min=1)
    w = (counts.sum() / counts).sqrt()
    w = (w / w.mean()).tolist()
    return train, val, {"classes": classes,
                        "val_records": sorted(int(r) for r in val_recs),
                        "class_weights": w, "n_features": n_features}


def get_ecg(args):
    """--final: all of DS1 for training; --val-records: fixed holdout."""
    x, y, rec, classes = load_components("ds1", args.data_dir)
    val_recs = [] if args.final else args.val_records
    n_features = make_features(x[:1], args.use_rr, args.morph).shape[1]
    return split_by_records(x, y, rec, val_recs, classes, n_features)


def make_folds(rec, y, k, seed, n_classes):
    """Split the DS1 records into k folds, balanced by S / V content.

    BBB records go first (one per fold), then the records with the most
    abnormal beats; each goes to the fold with the fewest records, ties
    broken by the lowest abnormal load (serpentine assignment). The seed
    only shuffles records that tie (e.g. all-N records).
    """
    recs = np.unique(rec)
    counts = {int(r): np.bincount(y[rec == r], minlength=n_classes)
              for r in recs}
    tot = np.maximum(sum(counts.values()), 1)
    frac = {r: (c / tot)[1:].sum() for r, c in counts.items()}  # S + V share
    order = [int(r) for r in np.random.default_rng(seed).permutation(recs)]
    order.sort(key=lambda r: (r not in BBB_RECORDS, -frac[r]))
    folds, load = [[] for _ in range(k)], np.zeros(k)
    for r in order:
        f = min(range(k), key=lambda i: (len(folds[i]), load[i]))
        folds[f].append(r)
        load[f] += frac[r]
    return [sorted(f) for f in folds]


# ----------------------------------------------------------------------
def encode(x, n_steps, encoding):
    """[B, F] features -> [T, B, F'] SNN input."""
    if encoding == "direct":
        return x.unsqueeze(0).expand(n_steps, *x.shape)
    if encoding == "rate":
        p = torch.cat([x.clamp(min=0), (-x).clamp(min=0)], dim=1) / 3.0
        p = p.clamp(0, 1)
        rnd = torch.rand(n_steps, *p.shape, device=x.device, dtype=x.dtype)
        return (rnd < p).float()
    raise ValueError(encoding)


class ANN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)

    def macs_per_sample(self):
        """Dense multiply-accumulates for one inference."""
        return self.n_in * self.n_hidden + self.n_hidden * self.n_out


class SNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, beta, encoding):
        super().__init__()
        self.encoding = encoding
        n_enc = 2 * n_in if encoding == "rate" else n_in
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(n_enc, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, return_hidden=False):
        # x: [T, B, F'] (encode() output)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur_const = self.fc1(x[0]) if self.encoding == "direct" else None
        rec, rec_hidden = [], []
        for step in range(x.shape[0]):
            cur1 = cur_const if cur_const is not None else self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            rec.append(spk2)
            rec_hidden.append(spk1)
        out = torch.stack(rec, dim=0)      # [T, B, C]
        if return_hidden:
            return out, torch.stack(rec_hidden, dim=0)   # [T, B, H]
        return out


# ---------------------------------------------------------------- streaming
# The streaming models read the beat sample by sample instead of as one
# vector. Their input is the same feature vector as the ANN's with
# --morph diff: x = [d_0 .. d_{L-1}, rr_0, rr_1, rr_2], where d_t is the beat
# minus the causal patient template (near 0 for a beat that looks like the
# patient's recent beats) and rr are the 3 RR features. Both models consume
# d_t one sample per step and get the RR features at the readout.
N_RR = 3


def delta_encode(d, theta, levels):
    """Level-crossing (send-on-delta) encoding of [B, T] signals.

    A tracker r follows d in steps of theta; at every sample the tracker
    moves by n = clip(round((d_t - r) / theta), -levels, levels) steps and
    emits |n| spikes on a thermometer code: UP_k fires if n >= k, DOWN_k if
    n <= -k (k = 1..levels). No change -> no spike. Deterministic, no
    gradient needed (input encoding). -> [T, B, 2*levels] 0/1 spikes.
    """
    B, T = d.shape
    r = torch.zeros(B, dtype=d.dtype, device=d.device)
    ks = torch.arange(1, levels + 1, dtype=d.dtype, device=d.device)
    out = []
    for t in range(T):
        n = torch.clamp(torch.round((d[:, t] - r) / theta), -levels, levels)
        r = r + n * theta
        out.append(torch.cat([(n.unsqueeze(1) >= ks).to(d.dtype),
                              (n.unsqueeze(1) <= -ks).to(d.dtype)], dim=1))
    return torch.stack(out)


class GRUNet(nn.Module):
    """GRU over the d_t stream (1 input per step), readout on [h, rr].
    bins > 1: the hidden state at the end of each of the bins time windows
    (the GRU counterpart of the RSNN's per-window spike counts)."""

    def __init__(self, n_in, n_hidden, n_out, bins=1):
        super().__init__()
        self.L = n_in - N_RR
        self.n_hidden, self.n_out, self.bins = n_hidden, n_out, bins
        self.gru = nn.GRU(1, n_hidden, batch_first=True)
        self.head = nn.Linear(bins * n_hidden + N_RR, n_out)
        # last step of every window (same windows as RSNN.bin_of)
        self.ends = [max(t for t in range(self.L) if t * bins // self.L == k)
                     for k in range(bins)]

    def forward(self, x):
        seq, rr = x[:, :self.L].unsqueeze(-1), x[:, self.L:]
        out, _ = self.gru(seq)                       # [B, L, H]
        h = out[:, self.ends].reshape(len(x), -1)    # [B, bins * H]
        return self.head(torch.cat([h, rr], dim=1))

    def macs_per_sample(self):
        """Dense MACs: 3 gates x (input + recurrent) per step, plus the head."""
        H = self.n_hidden
        return self.L * 3 * (1 + H) * H + (self.bins * H + N_RR) * self.n_out


class RSNN(nn.Module):
    """Recurrent spiking network over the delta-encoded d_t stream.

    hidden: LIF neurons (snnTorch Leaky, reset by subtraction, threshold 1)
            with input I_t = W_in s_in_t + W_rec s_h_{t-1}, no bias: with no
            input events and no recurrent spikes a neuron only decays, so the
            network does no work for an unchanging input;
    readout: logits = W_o (hidden spike counts) + W_r rr + b, a non-spiking
            linear layer, evaluated once per beat. bins > 1: the counts are
            kept per time window (bins equal windows of the beat), so the
            readout also sees WHEN a neuron fired; still one increment per
            spike.
    """

    def __init__(self, n_in, n_hidden, n_out, beta, theta, levels, bins=1):
        super().__init__()
        self.L = n_in - N_RR
        self.theta, self.levels, self.bins = theta, levels, bins
        self.n_hidden, self.n_out = n_hidden, n_out
        self.fc_in = nn.Linear(2 * levels, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.head = nn.Linear(bins * n_hidden + N_RR, n_out)

    def bin_of(self, t):
        """time window of step t (bins equal parts of the L steps)."""
        return t * self.bins // self.L

    def encode(self, x):
        return delta_encode(x[:, :self.L], self.theta, self.levels)

    def forward(self, x, return_spikes=False):
        s_in = self.encode(x)                        # [T, B, 2K]
        B = x.shape[0]
        mem = torch.zeros(B, self.n_hidden, dtype=x.dtype, device=x.device)
        spk = torch.zeros_like(mem)
        counts = [torch.zeros_like(mem) for _ in range(self.bins)]
        for t in range(s_in.shape[0]):
            spk, mem = self.lif(self.fc_in(s_in[t]) + self.fc_rec(spk), mem)
            k = self.bin_of(t)
            counts[k] = counts[k] + spk
        counts = torch.cat(counts, dim=1)              # [B, bins * H]
        logits = self.head(torch.cat([counts, x[:, self.L:]], dim=1))
        if return_spikes:
            return logits, s_in, counts
        return logits


STREAM_MODELS = ("GRU", "RSNN")


def build_model(config):
    """Model for a checkpoint config (see run_one)."""
    n_in, h, c = config["n_features"], config["hidden"], len(config["classes"])
    if config["model"] == "SNN":
        return SNN(n_in, h, c, config["beta"], config["encoding"])
    if config["model"] == "GRU":
        return GRUNet(n_in, h, c, config.get("readout_bins", 1))
    if config["model"] == "RSNN":
        return RSNN(n_in, h, c, config["beta"], config["delta_theta"],
                    config["delta_levels"], config.get("readout_bins", 1))
    return ANN(n_in, h, c)


def forward_logits(model, x, is_snn, n_steps, encoding):
    """Class scores: ANN logits, or SNN output spike counts over T steps."""
    if is_snn:
        return model(encode(x, n_steps, encoding)).sum(dim=0)
    return model(x)


def confusion(y, pred, n_classes):
    return torch.bincount(y * n_classes + pred, minlength=n_classes ** 2
                          ).view(n_classes, n_classes)


def cm_metrics(cm, classes):
    """accuracy, macro-F1, per-class Se (recall) and +P (precision)."""
    tp = cm.diag().float()
    se = tp / cm.sum(1).clamp(min=1)
    ppv = tp / cm.sum(0).clamp(min=1)
    f1 = 2 * se * ppv / (se + ppv).clamp(min=1e-9)
    m = {"accuracy": (tp.sum() / cm.sum()).item(),
         "macro_f1": f1.mean().item()}
    for i, c in enumerate(classes):
        m[f"se_{c}"] = se[i].item()
        m[f"ppv_{c}"] = ppv[i].item()
    return m


def train_epoch(model, loader, opt, sched, loss_fn, is_snn, n_steps,
                encoding, classes, feat, aug_gen=None):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n = 0
    cm = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for c, y in loader:
        if aug_gen is not None:
            c = augment(c, aug_gen)
        out = forward_logits(model, feat(c), is_snn, n_steps, encoding)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]
        cm += confusion(y, out.argmax(1), len(classes))
    return total_loss / n, cm_metrics(cm, classes), time.time() - t0


@torch.no_grad()
def evaluate(model, loader, loss_fn, is_snn, n_steps, encoding, classes,
             feat):
    model.eval()
    total_loss = 0.0
    n = 0
    cm = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for c, y in loader:
        out = forward_logits(model, feat(c), is_snn, n_steps, encoding)
        total_loss += loss_fn(out, y).item() * y.shape[0]
        n += y.shape[0]
        cm += confusion(y, out.argmax(1), len(classes))
    return total_loss / n, cm_metrics(cm, classes)


def run_one(name, seed, args, data, log_dir, tag="", save=True):
    is_snn = name == "SNN"
    train_ds, val_ds, meta = data
    config = {"model": name, "seed": seed, "epochs": args.epochs,
              "batch": args.batch, "lr": args.lr, "hidden": args.hidden,
              "timesteps": args.timesteps, "beta": args.beta,
              "weight_decay": args.weight_decay,
              "encoding": args.encoding if is_snn else "none",
              "use_rr": args.use_rr, "morph": args.morph,
              "delta_theta": args.delta_theta, "delta_levels": args.delta_levels,
              "readout_bins": args.readout_bins,
              "augment": args.augment, "data_dir": args.data_dir, **meta}
    classes = meta["classes"]
    # same seed -> same init (for equal shapes), same data order for ANN & SNN
    torch.manual_seed(seed)
    model = build_model(config)
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              generator=loader_gen)
    val_loader = (DataLoader(val_ds, batch_size=1024, shuffle=False)
                  if len(val_ds) else None)

    feat = lambda c: make_features(c, args.use_rr, args.morph)
    # dedicated generator: ANN and SNN of a seed see the same augmentations
    aug_gen = torch.Generator().manual_seed(seed) if args.augment else None
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(meta["class_weights"]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))

    run_dir = os.path.join(log_dir, f"{name.lower()}_seed{seed}{tag}")
    writer = SummaryWriter(run_dir)
    train_time = 0.0
    for epoch in range(1, args.epochs + 1):
        lr = opt.param_groups[0]["lr"]
        tr_loss, tr_m, secs = train_epoch(model, train_loader, opt, sched,
                                          loss_fn, is_snn, args.timesteps,
                                          args.encoding, classes, feat,
                                          aug_gen)
        train_time += secs
        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/accuracy", tr_m["accuracy"], epoch)
        writer.add_scalar("train/macro_f1", tr_m["macro_f1"], epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("epoch_time_s", secs, epoch)
        line = (f"  {name} seed={seed}{tag} epoch {epoch}/{args.epochs}: "
                f"train_loss={tr_loss:.4f} train_f1={tr_m['macro_f1']:.3f}")
        val_m = {"accuracy": None, "macro_f1": None}
        if val_loader is not None:
            val_loss, val_m = evaluate(model, val_loader, loss_fn, is_snn,
                                       args.timesteps, args.encoding, classes,
                                       feat)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_m["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val_m["macro_f1"], epoch)
            line += (f"  val_loss={val_loss:.4f} "
                     f"val_acc={val_m['accuracy']*100:.2f}% "
                     f"val_f1={val_m['macro_f1']:.3f}")
        print(line + f"  ({secs:.1f}s)")
    writer.close()

    info = {"log_dir": run_dir}
    if save:
        path = os.path.join(OUT_DIR, f"{name.lower()}_ecg_seed{seed}.pt")
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "train_time_s": train_time,
                    "val_accuracy": val_m["accuracy"],
                    "val_macro_f1": val_m["macro_f1"]}, path)
        info["checkpoint"] = path
        print(f"{name} seed={seed}: train_time={train_time:.1f}s -> {path}")
    return info, model


@torch.no_grad()
def predict(model, x, is_snn, n_steps, encoding, bs=1024):
    model.eval()
    return torch.cat([forward_logits(model, x[i:i + bs], is_snn, n_steps,
                                     encoding).argmax(1)
                      for i in range(0, len(x), bs)])


def run_cv(args, log_dir):
    """Patient-level K-fold CV on DS1, metrics on pooled out-of-fold preds."""
    comp, y, rec, classes = load_components("ds1", args.data_dir)
    x = make_features(comp, args.use_rr, args.morph)
    n_cls = len(classes)
    per_seed = {name: [] for name in args.models}
    for seed in args.seeds:
        folds = make_folds(rec, y.numpy(), args.cv_folds, seed, n_cls)
        print(f"seed {seed}: {args.cv_folds} folds {folds}")
        oof = {name: torch.full_like(y, -1) for name in args.models}
        for k, val_recs in enumerate(folds):
            data = split_by_records(comp, y, rec, val_recs, classes,
                                    x.shape[1])
            m = torch.from_numpy(np.isin(rec, val_recs))
            for name in args.models:
                _, model = run_one(name, seed, args, data, log_dir,
                                   tag=f"_fold{k}", save=False)
                torch.manual_seed(seed)      # reproducible rate encoding
                oof[name][m] = predict(model, x[m], name == "SNN",
                                       args.timesteps, args.encoding)
        for name in args.models:
            pred = oof[name]
            met = cm_metrics(confusion(y, pred, n_cls), classes)
            per_rec = {}
            for r in np.unique(rec):
                k = torch.from_numpy(rec == r)
                per_rec[int(r)] = {
                    "accuracy": (pred[k] == y[k]).float().mean().item(),
                    "cm": confusion(y[k], pred[k], n_cls).tolist()}
            per_seed[name].append({"seed": seed, **met, "per_record": per_rec})
            print(f"{name} seed={seed} out-of-fold: "
                  f"macro_f1={met['macro_f1']:.3f} "
                  f"acc={met['accuracy']*100:.2f}%  " +
                  "  ".join(f"Se_{c}={met['se_' + c]:.3f}" for c in classes))

    # ---- summary: mean +- std over seeds of the pooled out-of-fold metrics
    keys = ["macro_f1", "accuracy"] + [f"se_{c}" for c in classes] + \
           [f"ppv_{c}" for c in classes]
    names = list(per_seed)
    a = args
    lines = [f"CV ({a.cv_folds} patient folds on DS1, pooled out-of-fold), "
             f"epochs={a.epochs}, seeds={a.seeds}, batch={a.batch}, "
             f"lr={a.lr}, wd={a.weight_decay}, hidden={a.hidden}, "
             f"timesteps={a.timesteps}, beta={a.beta}, "
             f"encoding={a.encoding}, use_rr={a.use_rr}, morph={a.morph}, "
             f"augment={a.augment}", "",
             "| metric (mean +- std over seeds) | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for key in keys:
        cells = []
        for name in names:
            v = np.array([r[key] for r in per_seed[name]])
            cells.append(f"{v.mean():.3f} +- {v.std():.3f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "| record | " + " | ".join(f"{n} acc" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for r in sorted(per_seed[names[0]][0]["per_record"]):
        cells = [f"{np.mean([s['per_record'][r]['accuracy'] for s in per_seed[n]]):.3f}"
                 for n in names]
        mark = " (BBB)" if r in BBB_RECORDS else ""
        lines.append(f"| {r}{mark} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    base = os.path.join(OUT_DIR, "cv_summary")
    if args.tag:              # keep one summary per tuning variant
        os.makedirs(os.path.join(OUT_DIR, "cv"), exist_ok=True)
        base = os.path.join(OUT_DIR, "cv", args.tag)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(base + ".json", "w") as f:
        json.dump({"log_dir": log_dir, "args": vars(args),
                   "results": per_seed}, f, indent=1)
    with SummaryWriter(os.path.join(log_dir, "cv_summary")) as w:
        w.add_text("cv_summary", table)
    print(f"\nwrote {base}.md")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--hidden", type=int, default=N_HIDDEN)
    p.add_argument("--timesteps", type=int, default=N_TIMESTEPS)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--encoding", choices=["direct", "rate"], default=ENCODING)
    p.add_argument("--use-rr", action="store_true",
                   help="append the 3 RR-interval features")
    p.add_argument("--morph", choices=["raw", "diff", "both"], default="raw",
                   help="beat morphology input: raw beat, beat - causal "
                        "patient template, or both")
    p.add_argument("--augment", action="store_true",
                   help="light augmentation of the training beats")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--cv-folds", type=int, default=CV_FOLDS,
                   help="patient-level folds for the tuning mode (default)")
    p.add_argument("--final", action="store_true",
                   help="train on all of DS1, save for benchmark_ecg.py")
    p.add_argument("--val-records", type=int, nargs="+", default=None,
                   help="quick holdout on these DS1 records instead of CV")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--manifest", default=None,
                   help="--final / holdout: where to list the checkpoints "
                        "(default: <out-dir>/manifest.json)")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="checkpoints, TensorBoard logs and CV summaries (default: runs)")
    p.add_argument("--tag", default="",
                   help="CV mode: write runs/cv/<tag>.md/.json")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = torch default)")
    p.add_argument("--models", nargs="+", default=["ANN", "SNN"],
                   choices=["ANN", "SNN"] + list(STREAM_MODELS),
                   help="GRU / RSNN: streaming models (need --morph diff --use-rr)")
    p.add_argument("--delta-theta", type=float, default=0.2,
                   help="RSNN: level-crossing step of the delta encoder")
    p.add_argument("--readout-bins", type=int, default=1,
                   help="streaming readout per time window (1 = whole beat): "
                        "RSNN spike counts / GRU hidden state at window ends")
    p.add_argument("--delta-levels", type=int, default=2,
                   help="RSNN: max. steps per sample (spikes per polarity)")
    return p.parse_args()


def main():
    global OUT_DIR
    args = parse_args()
    OUT_DIR = args.out_dir                # read by run_one / run_cv
    if args.manifest is None:
        args.manifest = os.path.join(OUT_DIR, "manifest.json")
    if set(args.models) & set(STREAM_MODELS):
        assert args.morph == "diff" and args.use_rr, \
            "streaming models read [beat - template, rr]: use --morph diff --use-rr"
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if not args.final and not args.val_records:
        run_cv(args, os.path.join(OUT_DIR, "tb", stamp + "_cv"))
        return

    log_dir = os.path.join(OUT_DIR, "tb",
                           stamp + ("_final" if args.final else "_holdout"))
    runs = []
    data = get_ecg(args)
    print(f"{len(data[0])} train / {len(data[1])} val beats, val records "
          f"{data[2]['val_records']}, {data[2]['n_features']} features, "
          f"classes {data[2]['classes']}")
    for seed in args.seeds:
        for name in args.models:
            info, _ = run_one(name, seed, args, data, log_dir)
            runs.append(info)

    with open(args.manifest, "w") as f:
        json.dump({"log_dir": log_dir, "args": vars(args), "runs": runs},
                  f, indent=2)
    print(f"\nwrote {args.manifest} - now run:  python benchmark_ecg.py")
    print(f"TensorBoard:  tensorboard --logdir {os.path.join(OUT_DIR, 'tb')}")


if __name__ == "__main__":
    main()