#!/bin/bash
# start_demo.sh — start the live ECG demo (board side + PC side).
#
# Usage (Git Bash, from anywhere):  demo/start_demo.sh [board-ip] [--offline]
#   1. uploads the demo server, firmware, bitstreams and the four ECG exports
#      (ANN, SNN, GRU, recurrent SNN; md5-checked) to the board,
#   2. (re)starts demo_server_pynq.py on the board (keeps spike_top_snn.bit
#      loaded; log: ~/mnist_snn/demo.log; the MNIST demo uses the same one),
#   3. starts ecg_demo_server.py on this PC and opens http://localhost:8001.
# --offline (or an unreachable board) skips the board: software mode, PC
# int8 models only. A board that comes up later is connected automatically.
# Needs pynq/export/{ann,snn}_ecg_seed0 (export_ecg.py) and
# pynq/export_stream/{gru,rsnn}_ecg_seed0 (export_stream.py).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
MNIST="$ROOT/../ann_vs_snn_mnist"
FLOW="$ROOT/../pynqz2_riscv_flow"
OFFLINE=0
POS=()
for a in "$@"; do [ "$a" = "--offline" ] && OFFLINE=1 || POS+=("$a"); done
IP="${POS[0]:-$(cat "$FLOW/.board-ip" 2>/dev/null || echo 192.168.2.99)}"
PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
ARGS=(--board "$IP")

if [ "$OFFLINE" = 0 ]; then
    B="xilinx@$IP"; R=/home/xilinx/mnist_snn
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$B" true 2>/dev/null; then
        echo "uploading to $IP ..."
        for d in "$ROOT/pynq/export/ann_ecg_seed0" "$ROOT/pynq/export/snn_ecg_seed0" \
                 "$ROOT/pynq/export_stream/gru_ecg_seed0" "$ROOT/pynq/export_stream/rsnn_ecg_seed0"; do
            bash "$MNIST/pynq/deploy_pynq.sh" "$IP" "$d" -- --n 1 >/dev/null 2>&1 \
                || echo "  warning: upload of $(basename "$d") failed"
        done
        scp -q "$MNIST/pynq/host/demo_server_pynq.py" "$B:$R/"
        # stop an old instance (own ssh call: the start command below contains
        # the script name and would match the pattern itself); -f -n: do not
        # keep the ssh session open for the background server
        ssh "$B" "sudo pkill -f '[d]emo_server_pynq.py' || true"
        ssh -f -n "$B" "cd $R && nohup sudo bash -c \
'source /etc/profile.d/xrt_setup.sh && /usr/local/share/pynq-venv/bin/python3 -u demo_server_pynq.py' \
> demo.log 2>&1 < /dev/null &"
        for i in $(seq 30); do
            ssh "$B" "grep -q serving $R/demo.log" 2>/dev/null && break
            sleep 1
        done
        ssh "$B" "tail -n 1 $R/demo.log"
    else
        echo "board $IP not reachable: starting anyway, it connects when the board is up"
    fi
else
    ARGS+=(--offline)
fi

cd "$ROOT"
(sleep 4; start "" http://localhost:8001 2>/dev/null || xdg-open http://localhost:8001 2>/dev/null || true) &
exec "$PY" demo/ecg_demo_server.py "${ARGS[@]}"
