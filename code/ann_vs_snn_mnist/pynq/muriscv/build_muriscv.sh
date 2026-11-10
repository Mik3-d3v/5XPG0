#!/bin/bash
# build_muriscv.sh — build the muRISCV-NN pieces of the "muriscv" level, in
# Docker (ubuntu:22.04, the same gcc-riscv64-unknown-elf 10.2.0 as
# build_fw.sh), from the UNMODIFIED sources in third_party/muriscv-nn:
#   build/host_ref                 PC reference (x86 gcc, -O2): same library
#                                  sources + glue as the firmware
#   build/rv32im_O{2,3}/*.o        the library + glue for PicoRV32 (rv32im),
#                                  for code size and the disassembly check
#   build/rv32im_O{2,3}/check.txt  sizes + count of non-RV32IM instructions
# Only the plain scalar C path is built: USE_VEXT, USE_PEXT, USE_COREV,
# USE_MNN and the requantisation inline-assembly option stay undefined.
# Library sources get the firmware's -march / -mabi / -O flags but not
# -Werror (third-party code we may not modify); our glue keeps -Werror.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && (pwd -W 2>/dev/null || pwd))"

MSYS_NO_PATHCONV=1 docker run --rm -v "$REPO:/repo" -w /repo ubuntu:22.04 bash -c '
set -e
apt-get update -qq >/dev/null
apt-get install -y -qq gcc gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
    picolibc-riscv64-unknown-elf >/dev/null 2>&1
M=third_party/muriscv-nn
G=code/ann_vs_snn_mnist/pynq/muriscv
B=$G/build
mkdir -p $B
LIB="$M/Source/FullyConnectedFunctions/muriscv_nn_fully_connected_s8.c
     $M/Source/FullyConnectedFunctions/muriscv_nn_fully_connected_get_buffer_sizes_s8.c
     $M/Source/NNSupportFunctions/muriscv_nn_vec_mat_mult_t_s8.c"
INC="-I$M/Include -I$G"

# ---- host reference (x86)
gcc -O2 -std=gnu11 $INC -o $B/host_ref $G/host_ref.c $G/ann_muriscv.c $LIB -lm
echo "host_ref: $(gcc --version | head -1)"

# ---- RV32IM objects, firmware flags (see firmware/Makefile)
PICO=/usr/lib/picolibc/riscv64-unknown-elf/include
for O in O2 O3; do
  D=$B/rv32im_$O; mkdir -p $D
  F="-march=rv32im -mabi=ilp32 -$O -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns -nostdlib -g"
  riscv64-unknown-elf-gcc $F -Wall -Wextra -Werror $INC -isystem $PICO -c -o $D/ann_muriscv.o $G/ann_muriscv.c
  for s in $LIB; do
    riscv64-unknown-elf-gcc $F $INC -isystem $PICO -c -o $D/$(basename $s .c).o $s
  done
  {
    echo "== $O: riscv64-unknown-elf-gcc $(riscv64-unknown-elf-gcc -dumpversion), $F"
    riscv64-unknown-elf-size $D/*.o
    # every instruction must be RV32IM: decode with a plain rv32im objdump
    riscv64-unknown-elf-objdump -d -M no-aliases,numeric --no-show-raw-insn \
        $D/*.o > $D/disasm.txt
    echo "instructions: $(grep -cP "^\s+[0-9a-f]+:\t" $D/disasm.txt)"
    echo "mnemonics used:"
    grep -oP "^\s+[0-9a-f]+:\t\K[a-z.0-9]+" $D/disasm.txt | sort | uniq -c | sort -rn | tr "\n" " "; echo
  } > $D/check.txt
  cat $D/check.txt
done
'
