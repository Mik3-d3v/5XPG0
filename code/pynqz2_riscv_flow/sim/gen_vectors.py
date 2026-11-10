#!/usr/bin/env python3
"""gen_vectors.py — random instruction stream + expected results for
tb_snn_pcpi.v, from a Python model of snn_pcpi.v (the LIF is the one of
quant.int_forward_snn). Writes vectors.hex: per instruction 4 words
(insn, rs1, rs2, expected rd or 0xFFFFFFFF = no result)."""
import random

N = 256
M = 0xFFFFFFFF
def s32(x): x &= M; return x - (1 << 32) if x >> 31 else x
def insn(f7, f3): return (f7 << 25) | (f3 << 12) | (10 << 7) | 0b0001011

rng = random.Random(1)
mem, cur = [0] * N, [0] * N
beta, thr = [0] * 4, [1] * 4
vec = []
def emit(i, a, b, rd=None): vec.append((i, a & M, b & M, M if rd is None else rd & M))

for slot in range(4):                       # configure all slots first
    beta[slot] = rng.choice([0, 1, 32768, 58982, 62259, 65536])
    thr[slot] = rng.randint(1, 20000)
    emit(insn(0, slot), beta[slot], thr[slot])
emit(insn(1, 0), 0, N)                      # clear everything
for _ in range(4000):
    op = rng.choice([1, 2, 3, 3, 3, 4, 4, 4, 5, 0])
    if op == 0:
        slot = rng.randrange(4)
        beta[slot] = rng.choice([0, 1, 32768, 58982, 62259, 65536, rng.randrange(65537)])
        thr[slot] = rng.randint(1, 30000)
        emit(insn(0, slot), beta[slot], thr[slot])
    elif op == 1:
        f = rng.randrange(N); c = rng.randint(0, min(40, N - f))
        for n in range(f, f + c): mem[n] = cur[n] = 0
        emit(insn(1, 0), f, c)
    elif op == 2:
        n = rng.randrange(N); v = s32(rng.randint(-50000, 50000))
        cur[n] = v; emit(insn(2, 0), n, v)
    elif op == 3:
        n = rng.randrange(N // 4) * 4; f3 = rng.randrange(4)
        w = [rng.randrange(256) for _ in range(4)]
        for k in range(4):
            x = w[k] - 128 if f3 & 1 else (w[k] - 256 if w[k] > 127 else w[k])
            cur[n + k] = s32(cur[n + k] - x if f3 & 2 else cur[n + k] + x)
        emit(insn(3, f3), n, w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24)
    elif op == 4:
        slot = rng.randrange(4); f = rng.randrange(N); c = rng.randint(0, min(32, N - f))
        mask = 0
        for k in range(c):
            n = f + k; mo = mem[n]
            v = s32(((beta[slot] * mo) >> 16) + cur[n] - (thr[slot] if mo > thr[slot] else 0))
            mem[n] = v
            if v > thr[slot]: mask |= 1 << k
        emit(insn(4, slot), f, c, mask)
    else:
        n = rng.randrange(N); f3 = rng.randrange(2)
        emit(insn(5, f3), n, 0, cur[n] if f3 else mem[n])
with open("vectors.hex", "w") as f:
    for v in vec:
        f.write("".join(f"{x:08x}\n" for x in v))
print(len(vec), "instructions,", sum(v[3] != M for v in vec), "checked results")
