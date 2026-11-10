#!/usr/bin/env python3
"""loops.py — list the loops (backward branches) of one function in an
objdump -d -M no-aliases,numeric listing, with their instruction mix.
Usage: loops.py <disasm.txt> <function>"""
import re
import sys
from collections import Counter

path, fn = sys.argv[1:3]
ins, on = [], False
for l in open(path):
    if re.match(r"^[0-9a-f]+ <[^.]", l):     # function symbol (not .L labels)
        on = l.strip().endswith(f"<{fn}>:")
        continue
    m = re.match(r"\s+([0-9a-f]+):\t(\S+)\t?(.*)", l)
    if on and m:
        ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
for a, mn, ops in ins:
    t = re.search(r"([0-9a-f]+) <", ops)
    if mn in ("bne", "blt", "bge", "bltu", "bgeu", "beq", "jal") and t and int(t.group(1), 16) < a:
        ta = int(t.group(1), 16)
        body = [x for x in ins if ta <= x[0] <= a]
        mix = Counter(x[1] for x in body)
        print(f"loop {ta:x}-{a:x}: {len(body):3d} instr, loads {mix['lw'] + mix['lb'] + mix['lbu'] + mix['lh'] + mix['lhu']}, "
              f"stores {mix['sw'] + mix['sb'] + mix['sh']}, mul {mix['mul'] + mix['mulh']}  {dict(mix)}")
