#!/bin/bash
# deploy_pynq.sh — from the PC: copy an exported ANN or SNN to the PYNQ-Z2 and run it.
#
# Usage: ./deploy_pynq.sh [board-ip] [export-name] [--detach] [-- run_mnist_pynq.py args]
#   board-ip     default: ../../pynqz2_riscv_flow/.board-ip
#   export-name  default: snn_mnist_seed0  (a folder in export/), or a path
#                to an export folder elsewhere (e.g. the ECG exports)
#   --detach     run in the background ON THE BOARD (nohup, log in
#                <export-name>/run.log) and return at once; for long runs such
#                as the full test set. Follow / fetch with ./status_pynq.sh.
# Examples:
#   ./deploy_pynq.sh                                       # seed0, all exported samples
#   ./deploy_pynq.sh 192.168.2.99 snn_mnist_seed1 -- --n 100
#   ./deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 --detach
#   ./deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 --detach -- --resume
#   ./deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 -- --fw baseline --n 100
# run_mnist_pynq.py args: --fw fast|baseline|hw|muriscv|mac (default fast), --n N,
#   --resume, --tag X
#   (hw: SNN coprocessor firmware, loads spike_top_snn.bit; muriscv: ANN on
#   muRISCV-NN, needs an ann_*_muriscv export of muriscv_ref.py; mac: SNN and
#   ANN MAC coprocessors, loads spike_top_both.bit)
#
# Files are only uploaded when their md5 differs from the copy on the board
# (inputs.bin of the full test set is 25 MB).
# Same prerequisites as pynqz2_riscv_flow/install.sh (SSH key auth for
# xilinx@<board-ip>, passwordless sudo). Needs export/<name>/ (export_model.py)
# and firmware/snn_mnist{,_fast}.bin (build_fw.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FLOW="$HERE/../../pynqz2_riscv_flow"

POS=()
DETACH=0
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
        --detach) DETACH=1 ;;
        *) POS+=("$1") ;;
    esac
    shift
done
[ "${1:-}" = "--" ] && shift
IP="${POS[0]:-$(cat "$FLOW/.board-ip" 2>/dev/null || true)}"
NAME="${POS[1]:-snn_mnist_seed0}"
[ -n "$IP" ] || { echo "usage: $0 <board-ip> [export-name] [--detach] [-- args]" >&2; exit 2; }
EXP="$HERE/export/$NAME"
if [[ "$NAME" == */* || "$NAME" == *\\* ]]; then EXP="$(cd "$NAME" && pwd)"; NAME="$(basename "$EXP")"; fi
[ -d "$EXP" ] || { echo "no $EXP - run the exporter" >&2; exit 1; }
[ -f "$HERE/firmware/snn_mnist_fast.bin" ] || { echo "no firmware - run build_fw.sh" >&2; exit 1; }

BOARD="xilinx@$IP"
REMOTE=/home/xilinx/mnist_snn
if ssh "$BOARD" "pgrep -f '[r]un_mnist_pynq.py' >/dev/null"; then
    echo "a run_mnist_pynq.py run is still active on the board - see ./status_pynq.sh" >&2
    exit 1
fi
ssh "$BOARD" "mkdir -p $REMOTE/$NAME"

# upload <remote-dir> <local files...>: only files whose md5 differs
upload() {
    local dir="$1"; shift
    local remote_sums todo=() f
    remote_sums="$(ssh "$BOARD" "cd $dir && md5sum $(for f in "$@"; do printf '%q ' "$(basename "$f")"; done) 2>/dev/null" || true)"
    for f in "$@"; do
        local sum
        sum="$(md5sum < "$f" | cut -d' ' -f1)"
        grep -q "^$sum  $(basename "$f")\$" <<<"$remote_sums" || todo+=("$f")
    done
    if [ ${#todo[@]} -gt 0 ]; then
        echo "upload -> $dir: $(for f in "${todo[@]}"; do printf '%s ' "$(basename "$f")"; done)"
        scp -q "${todo[@]}" "$BOARD:$dir/"
    fi
}
upload "$REMOTE" "$FLOW/vivado/spike_top.bit" "$FLOW/host/spike_pynq.py" \
    "$HERE/host/run_mnist_pynq.py" "$HERE/firmware/snn_mnist.bin" "$HERE/firmware/snn_mnist_fast.bin"
# SNN coprocessor level (--fw hw): its firmware and bitstream, the
# muRISCV-NN level (--fw muriscv, ANN3 exports) and the SNN + MAC
# coprocessor level (--fw mac), if built
for f in "$HERE/firmware/snn_mnist_hw.bin" "$FLOW/vivado/spike_top_snn.bit"          "$HERE/firmware/snn_mnist_muriscv.bin"          "$HERE/firmware/snn_mnist_mac.bin" "$FLOW/vivado/spike_top_both.bit"; do
    if [ -f "$f" ]; then upload "$REMOTE" "$f"; fi
done
upload "$REMOTE/$NAME" "$EXP"/{meta.json,model.bin,inputs.bin,labels.bin,float_pred.bin} "$EXP"/ref_*.bin

RUN="source /etc/profile.d/xrt_setup.sh && \
/usr/local/share/pynq-venv/bin/python3 -u run_mnist_pynq.py $NAME $*"
if [ "$DETACH" = 1 ]; then
    ssh "$BOARD" "cd $REMOTE && nohup sudo bash -c '$RUN' > $NAME/run.log 2>&1 < /dev/null &"
    echo "started on the board in the background (log: $REMOTE/$NAME/run.log)"
    echo "follow / fetch results:  $HERE/status_pynq.sh $IP $NAME"
    exit 0
fi
ssh -t "$BOARD" "cd $REMOTE && sudo bash -c '$RUN'"
scp -q "$BOARD:$REMOTE/$NAME/board_*" "$EXP/" || true
