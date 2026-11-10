#!/bin/bash
# build_fw.sh — build firmware/snn_mnist{,_fast,_hw,_muriscv}.bin (ANN + SNN,
# MNIST + ECG). Uses riscv64-unknown-elf-gcc if it is on PATH, otherwise
# builds inside a Docker container (ubuntu:22.04 + gcc-riscv64-unknown-elf
# 10.2.0), so Windows users only need Docker Desktop. Run from Git Bash / Linux.
# snn_mnist_muriscv.bin needs the muRISCV-NN submodule (third_party/) and the
# picolibc headers (picolibc-riscv64-unknown-elf, installed in the container).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

if [ ! -f "$ROOT/third_party/muriscv-nn/Include/muriscv_nn_functions.h" ]; then
    echo "muRISCV-NN missing - run: git submodule update --init third_party/muriscv-nn" >&2
    exit 1
fi

if command -v riscv64-unknown-elf-gcc >/dev/null; then
    make -C "$HERE/firmware"
    exit 0
fi

# mount the repository root (the Makefile reaches ../../../pynqz2_riscv_flow
# and ../../../../third_party)
ROOT_W="$(cd "$ROOT" && (pwd -W 2>/dev/null || pwd))"
MSYS_NO_PATHCONV=1 docker run --rm -v "$ROOT_W:/repo" \
    -w /repo/code/ann_vs_snn_mnist/pynq/firmware ubuntu:22.04 bash -c \
    "apt-get update -qq && apt-get install -y -qq gcc-riscv64-unknown-elf \
     picolibc-riscv64-unknown-elf make >/dev/null && make"
