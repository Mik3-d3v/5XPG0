#!/bin/bash
# start_demo.sh — start the live ANN vs SNN demo (board side + PC side).
#
# Usage (Git Bash, from anywhere):  demo/start_demo.sh [board-ip] [--offline]
#   1. uploads the demo server, firmware, bitstreams and the two MNIST
#      exports to the board (deploy_pynq.sh does the uploads, md5-checked),
#   2. (re)starts demo_server_pynq.py on the board in the background
#      (it keeps spike_top_snn.bit loaded; log: ~/mnist_snn/demo.log),
#   3. starts demo_server.py on this PC and opens http://localhost:8000.
# --offline skips the board: PC int8 models only, no latency.
# Needs the exports pynq/export/{ann,snn}_mnist_seed0 (export_model.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
FLOW="$ROOT/../pynqz2_riscv_flow"
OFFLINE=0
POS=()
for a in "$@"; do [ "$a" = "--offline" ] && OFFLINE=1 || POS+=("$a"); done
IP="${POS[0]:-$(cat "$FLOW/.board-ip" 2>/dev/null || echo 192.168.2.99)}"
PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/.venv/bin/python"

if [ "$OFFLINE" = 0 ]; then
    B="xilinx@$IP"; R=/home/xilinx/mnist_snn
    echo "uploading to $IP ..."
    # deploy_pynq.sh uploads bit/firmware/runner + an export; --n 1 = one
    # quick run that also proves the board works
    for m in ann_mnist_seed0 snn_mnist_seed0; do
        bash "$ROOT/pynq/deploy_pynq.sh" "$IP" "$m" -- --n 1 >/dev/null
    done
    scp -q "$ROOT/pynq/host/demo_server_pynq.py" "$B:$R/"
    # stop an old instance (own ssh call: the start command below contains
    # the script name and would match the pattern itself)
    ssh "$B" "sudo pkill -f '[d]emo_server_pynq.py' || true"
    # -f -n: do not keep the ssh session open for the background server
    ssh -f -n "$B" "cd $R && nohup sudo bash -c \
'source /etc/profile.d/xrt_setup.sh && /usr/local/share/pynq-venv/bin/python3 -u demo_server_pynq.py' \
> demo.log 2>&1 < /dev/null &"
    for i in $(seq 30); do
        ssh "$B" "grep -q serving $R/demo.log" 2>/dev/null && break
        sleep 1
    done
    ssh "$B" "tail -n 2 $R/demo.log"
fi

cd "$ROOT"
ARGS=(); [ "$OFFLINE" = 1 ] && ARGS+=(--offline)
(sleep 2; start "" http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null || true) &
exec "$PY" demo/demo_server.py --board "$IP" "${ARGS[@]}"
