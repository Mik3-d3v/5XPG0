#!/bin/bash
# status_pynq.sh — from the PC: show the state of a (detached) board run and
# copy its results back to export/<name>/.
#
# Usage: ./status_pynq.sh [board-ip] [export-name] [-f]
#   -f   follow the log until the run ends (Ctrl-C only stops following)
#   export-name may also be a path to an export folder (see deploy_pynq.sh)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FLOW="$HERE/../../pynqz2_riscv_flow"

POS=()
FOLLOW=0
for arg in "$@"; do
    case "$arg" in
        -f) FOLLOW=1 ;;
        *) POS+=("$arg") ;;
    esac
done
IP="${POS[0]:-$(cat "$FLOW/.board-ip" 2>/dev/null || true)}"
NAME="${POS[1]:-snn_mnist_seed0}"
EXP="$HERE/export/$NAME"
if [[ "$NAME" == */* || "$NAME" == *\\* ]]; then mkdir -p "$NAME"; EXP="$(cd "$NAME" && pwd)"; NAME="$(basename "$EXP")"; fi
BOARD="xilinx@$IP"
DIR=/home/xilinx/mnist_snn/$NAME

if [ "$FOLLOW" = 1 ]; then
    ssh "$BOARD" "tail -n 20 -f --pid=\$(pgrep -of '[r]un_mnist_pynq.py' || echo 1) $DIR/run.log" || true
else
    ssh "$BOARD" "tail -n 20 $DIR/run.log 2>/dev/null; \
        pgrep -f '[r]un_mnist_pynq.py' >/dev/null && echo '# still running' || echo '# not running'; \
        for f in $DIR/board_results_*.csv; do echo \"# \$(basename \$f): \$((\$(wc -l < \$f) - 1)) samples\"; done"
fi
mkdir -p "$EXP"
scp -q "$BOARD:$DIR/board_*" "$BOARD:$DIR/run.log" "$EXP/" 2>/dev/null || true
echo "results copied to $EXP/"
