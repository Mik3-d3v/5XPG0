#!/usr/bin/env python3
"""gen_vectors_mac.py — random instruction stream + expected results for
tb_mac_pcpi.v, from a Python model of mac_pcpi.v (the arithmetic of
mac_layer() in firmware/ann.c: acc = bias + sum x * w, int32 wrap-around).
Writes vectors_mac.hex: per instruction 5 words (insn, rs1, rs2, kind,
expected rd); kind 0 = no result, 1 = check rd, 2 = not ours: the unit must
not claim it (e.g. a custom-0 SNN instruction in the combined SoC)."""
import random

N = 256
M = 0xFFFFFFFF
def s32(x): x &= M; return x - (1 << 32) if x >> 31 else x
def insn(f7, f3): return (f7 << 25) | (f3 << 12) | (10 << 7) | 0b0101011

rng = random.Random(1)
acc = [0] * N
X = 0
vec = []
def emit(i, a, b, rd=None, kind=None):
    vec.append((i, a & M, b & M, kind if kind is not None else int(rd is not None),
                0 if rd is None else rd & M))

emit(insn(0, 0), 0, N)                      # clear everything
for _ in range(4000):
    op = rng.choice([0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5])
    if op == 0:                             # mac.clr
        f = rng.randrange(N); c = rng.randint(0, min(40, N - f))
        for n in range(f, f + c): acc[n] = 0
        emit(insn(0, 0), f, c)
    elif op == 1:                           # mac.setx: int8 / uint8 / edge values
        x = rng.choice([rng.randint(-128, 127), rng.randint(0, 255),
                        -128, 127, 255, 0, 1, -1])
        X = x                               # the unit keeps the low 9 bits, signed
        # upper bits of rs1 are ignored: sometimes pass garbage there
        hi = rng.choice([0, 0, rng.randrange(1 << 23)]) << 9
        emit(insn(1, rng.randrange(8)), (x & 0x1FF) | hi, rng.randrange(1 << 32))
    elif op == 2:                           # mac.acc4
        n = rng.randrange(N // 4) * 4; f3 = rng.randrange(8)
        w = [rng.randrange(256) for _ in range(4)]
        for k in range(4):
            wk = w[k] - 128 if f3 & 1 else (w[k] - 256 if w[k] > 127 else w[k])
            acc[n + k] = s32(acc[n + k] + X * wk)
        emit(insn(2, f3), n, w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24)
    elif op == 3:                           # mac.get
        n = rng.randrange(N)
        emit(insn(3, rng.randrange(8)), n, 0, acc[n])
    elif op == 5:                           # foreign: custom-0 or unused funct7
        if rng.random() < 0.5:
            i = (rng.randrange(6) << 25) | (rng.randrange(8) << 12) | (10 << 7) | 0b0001011
        else:
            i = insn(rng.randint(5, 127), rng.randrange(8))
        emit(i, rng.randrange(1 << 32), rng.randrange(1 << 32), kind=2)
    else:                                   # mac.setacc (bias, incl. near-wrap values)
        n = rng.randrange(N)
        v = s32(rng.choice([rng.randint(-50000, 50000), rng.randrange(1 << 32)]))
        acc[n] = v
        emit(insn(4, 0), n, v)
for n in range(N):                          # read back every accumulator
    emit(insn(3, 0), n, 0, acc[n])
with open("vectors_mac.hex", "w") as f:
    for v in vec:
        f.write("".join(f"{x:08x}\n" for x in v))
print(len(vec), "instructions,", sum(v[3] == 1 for v in vec), "checked results,",
      sum(v[3] == 2 for v in vec), "foreign")
