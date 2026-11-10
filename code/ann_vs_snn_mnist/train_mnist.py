#!/usr/bin/env python3
"""train_mnist.py — ANN vs SNN on MNIST (GROUP 4).

Trains TWO 2-layer MLPs on MNIST under the SAME budget and saves their weights:
  * an ANN  (784 -> H -> 10, ReLU)
  * an SNN  (784 -> H LIF -> 10 LIF, rate-encoded input, snnTorch)

What is held constant between the two models (per seed):
  topology/width, optimiser (Adam + cosine LR schedule), learning rate,
  batch size, number of epochs, weight initialisation seed, data order and
  the train/validation split.

Every run is repeated for several seeds so results can be reported as
mean +- std. 5 000 training images are held out as a validation set: use it
for tuning. The test set is only touched by benchmark.py.

Learning curves go to TensorBoard (runs/tb/<timestamp>/<model>_seed<k>):
  train/loss, train/accuracy, val/loss, val/accuracy, lr, epoch_time_s.
Runs with the same tag are overlaid, so ANN and SNN appear in one chart.
benchmark.py adds the test metrics and the comparison to the same log dir.

Key design choices still left to you:
  1. INPUT ENCODING   — rate, latency, or delta encoding for the SNN.
  2. TOPOLOGY         — width/depth, recurrence, beta, threshold.
  3. TRAINING         — loss on spike counts vs membrane, surrogate gradient.
  4. BENCHMARK        — configure benchmark.py and analyse the results.

Run:   python train_mnist.py                      # defaults below
       python train_mnist.py --epochs 10 --seeds 0 1 2 3 4
View:  tensorboard --logdir runs/tb
Output: runs/<model>_mnist_seed<k>.pt   (weights + config, every seed)
        runs/manifest.json              (what benchmark.py evaluates)
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import surrogate
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 5
SEEDS = (0, 1)
BATCH = 64
LR = 1e-3
N_HIDDEN = 128
N_TIMESTEPS = 25      # SNN simulation steps
BETA = 0.9            # LIF membrane decay
N_VAL = 5000          # training images held out for validation
OUT_DIR = "runs"
MANIFEST = os.path.join(OUT_DIR, "manifest.json")


def get_mnist(n_val, seed):
    tf = torchvision.transforms.ToTensor()
    full = torchvision.datasets.MNIST("./data", train=True, download=True,
                                      transform=tf)
    # the split depends only on `seed`, so it is identical for ANN and SNN
    train, val = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))
    return train, val


# ----------------------------------------------------------------------
def rate_encode(x, n_steps):
    """Binary rate encoding of a normalized image -> [T, B, 784] spikes.

    Each pixel (in [0,1]) becomes a Bernoulli spike train whose expected
    number of spikes over T steps is proportional to the pixel value.
    """
    x = x.reshape(x.shape[0], -1)          # [B, 784]
    rnd = torch.rand(n_steps, *x.shape, device=x.device, dtype=x.dtype)
    return (rnd < x).float()


class ANN(nn.Module):
    def __init__(self, n_hidden=N_HIDDEN):
        super().__init__()
        self.n_hidden = n_hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 10),
        )

    def forward(self, x):
        return self.net(x)

    def macs_per_sample(self):
        """Dense multiply-accumulates for one inference."""
        return 784 * self.n_hidden + self.n_hidden * 10


class SNN(nn.Module):
    def __init__(self, n_hidden=N_HIDDEN, beta=BETA):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(784, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, return_hidden=False):
        # x: [T, B, 784]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        rec, rec_hidden = [], []
        for step in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            rec.append(spk2)
            rec_hidden.append(spk1)
        out = torch.stack(rec, dim=0)      # [T, B, 10]
        if return_hidden:
            return out, torch.stack(rec_hidden, dim=0)   # [T, B, H]
        return out


def build_model(config):
    """Model for a checkpoint config (see run_one)."""
    if config["model"] == "SNN":
        return SNN(config["hidden"], config["beta"])
    return ANN(config["hidden"])


def forward_logits(model, x, is_snn, n_steps):
    """Class scores: ANN logits, or SNN output spike counts over T steps."""
    if is_snn:
        return model(rate_encode(x, n_steps)).sum(dim=0)
    return model(x)


def train_epoch(model, loader, opt, sched, is_snn, n_steps):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    total_loss = 0.0
    correct = n = 0
    for x, y in loader:
        out = forward_logits(model, x, is_snn, n_steps)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total_loss += loss.item() * y.shape[0]
        correct += (out.argmax(1) == y).sum().item()
        n += y.shape[0]
    return total_loss / n, correct / n, time.time() - t0


@torch.no_grad()
def evaluate(model, loader, is_snn, n_steps):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    correct = n = 0
    for x, y in loader:
        out = forward_logits(model, x, is_snn, n_steps)
        total_loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.shape[0]
    return total_loss / n, correct / n


def run_one(name, seed, args, data, log_dir):
    is_snn = name == "SNN"
    train_ds, val_ds = data
    config = {"model": name, "seed": seed, "epochs": args.epochs,
              "batch": args.batch, "lr": args.lr, "hidden": args.hidden,
              "timesteps": args.timesteps, "beta": args.beta}
    # same seed -> same init (for equal shapes), same data order for ANN & SNN
    torch.manual_seed(seed)
    model = build_model(config)
    loader_gen = torch.Generator().manual_seed(seed)
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=shuffle, num_workers=2,
        generator=loader_gen if shuffle else None,
        persistent_workers=True)
    train_loader = mk(train_ds, True)
    val_loader = mk(val_ds, False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))

    run_dir = os.path.join(log_dir, f"{name.lower()}_seed{seed}")
    writer = SummaryWriter(run_dir)
    train_time = 0.0
    for epoch in range(1, args.epochs + 1):
        lr = opt.param_groups[0]["lr"]
        tr_loss, tr_acc, secs = train_epoch(model, train_loader, opt, sched,
                                            is_snn, args.timesteps)
        train_time += secs
        val_loss, val_acc = evaluate(model, val_loader, is_snn, args.timesteps)

        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/accuracy", tr_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("epoch_time_s", secs, epoch)
        print(f"  {name} seed={seed} epoch {epoch}/{args.epochs}: "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}%  "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%  "
              f"({secs:.1f}s)")
    writer.close()

    path = os.path.join(OUT_DIR, f"{name.lower()}_mnist_seed{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "config": config,
                "train_time_s": train_time, "val_accuracy": val_acc}, path)
    print(f"{name} seed={seed}: val_acc={val_acc*100:.2f}%  "
          f"train_time={train_time:.1f}s  -> {path}")
    return {"checkpoint": path, "log_dir": run_dir}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--hidden", type=int, default=N_HIDDEN)
    p.add_argument("--timesteps", type=int, default=N_TIMESTEPS)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--models", nargs="+", default=["ANN", "SNN"],
                   choices=["ANN", "SNN"])
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="checkpoints and TensorBoard logs (default: runs)")
    p.add_argument("--manifest", default=None,
                   help="where to list the checkpoints (default: <out-dir>/manifest.json)")
    return p.parse_args()


def main():
    global OUT_DIR, MANIFEST
    args = parse_args()
    OUT_DIR = args.out_dir                # read by run_one
    MANIFEST = args.manifest or os.path.join(OUT_DIR, "manifest.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    log_dir = os.path.join(OUT_DIR, "tb", time.strftime("%Y%m%d-%H%M%S"))

    runs = []
    for seed in args.seeds:
        data = get_mnist(N_VAL, seed)
        for name in args.models:
            runs.append(run_one(name, seed, args, data, log_dir))

    with open(MANIFEST, "w") as f:
        json.dump({"log_dir": log_dir, "args": vars(args), "runs": runs},
                  f, indent=2)
    print(f"\nwrote {MANIFEST} - now run:  python benchmark.py")
    print(f"TensorBoard:  tensorboard --logdir {os.path.join(OUT_DIR, 'tb')}")


if __name__ == "__main__":
    main()
