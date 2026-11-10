#!/bin/bash
# overnight.sh — unattended accuracy-vs-cost sweeps (ANN vs SNN at equal
# accuracy). Resumable: every step leaves <res>/state/<step>.done and is
# skipped on a rerun; a failed step is logged and the run goes on.
#
#   ./overnight.sh smoke     1 seed, 1 epoch, 50 board samples, separate
#                            folders (*_smoke), prints a time estimate
#   ./overnight.sh full      the real run
#   ./overnight.sh full A B  only these jobs (default: A B C D E)
#
# Jobs (in this order; drop from the bottom if short of time):
#   A  ECG width x T: ANN hidden {8,16,32,64,128}, SNN hidden {16,32,64} x
#      T {2,5,10}; final config otherwise; 3 seeds; board fast (+ hw for
#      SNNs) on one seeded 5 000-beat DS2 subset
#   B  MNIST SNN T {2,4,6,8,12}, seeds 0 1; board fast + hw, first 2 000 images
#   C  streaming GRU hidden 16, 32 (tuning configs), 3 seeds; board fast, 200 beats
#   D  MNIST seed 2, ANN + SNN (T = 25), full test set, board fast
#   E  ECG CV macro-F1 (5 patient folds, seed 0) for every A configuration
# Nothing published is touched: checkpoints -> code/*/runs/overnight/,
# exports -> code/*/pynq/export_overnight/, results -> docs/results/overnight_2026-10-12/.
set -o pipefail      # no -e / -u: one failing step must not end the night
# run from a frozen copy: bash reads scripts while running, so editing
# overnight.sh during the night would otherwise corrupt the running run
if [ -z "${OVERNIGHT_HERE:-}" ]; then
    export OVERNIGHT_HERE="$(cd "$(dirname "$0")" && pwd)"
    cp "$0" "$OVERNIGHT_HERE/.overnight.running.sh"
    exec bash "$OVERNIGHT_HERE/.overnight.running.sh" "$@"
fi
MODE="${1:-full}"; shift || true
JOBS="${*:-A B C D E}"

HERE="$OVERNIGHT_HERE"                           # code/
ROOT="$(cd "$HERE/.." && pwd)"
ECG="$HERE/ann_vs_snn_ecg"
MN="$HERE/ann_vs_snn_mnist"
PY_E="$ECG/.venv/Scripts/python.exe"; [ -x "$PY_E" ] || PY_E="$ECG/.venv/bin/python"
PY_M="$MN/.venv/Scripts/python.exe"; [ -x "$PY_M" ] || PY_M="$MN/.venv/bin/python"
DEPLOY="$MN/pynq/deploy_pynq.sh"
IP="$(cat "$HERE/pynqz2_riscv_flow/.board-ip")"
# Windows Python needs C:/... paths, not Git Bash /c/... (relative paths are fine)
winpath() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }
ROOT_W="$(winpath "$ROOT")"
# published ECG input scale: every sweep model uses it (only the swept variable changes)
S_X="$("$PY_E" -c "import json;print(json.load(open(r'$ROOT_W/docs/results/ann_vs_snn_ecg/ann_seed0_meta.json'))['input_scale'])")"
[ -n "$S_X" ] || { echo "could not read the published input scale" >&2; exit 1; }

if [ "$MODE" = smoke ]; then
    SUB=overnight_smoke; RES="$ROOT/docs/results/overnight_2026-10-12/smoke"
    S3="0"; S2="0"; EP=1; NA=50; NB=50; NC=50; ND=50; PAR=8
else
    SUB=overnight; RES="$ROOT/docs/results/overnight_2026-10-12"
    S3="0 1 2"; S2="0 1"; EP=""; NA=5000; NB=2000; NC=200; ND=10000; PAR=6
fi
THREADS=3
STATE="$RES/state"; LOG="$RES/logs"; TIMING="$RES/timing.csv"
mkdir -p "$STATE" "$LOG"
RES_W="$(winpath "$RES")"
[ -f "$TIMING" ] || echo "step,seconds,status,end" > "$TIMING"
ep() { [ -n "$EP" ] && echo "--epochs $EP" || echo "${1:+--epochs $1}"; }   # smoke: 1 epoch

ECG_CFGS="ann_h8 ann_h16 ann_h32 ann_h64 ann_h128"
for h in 16 32 64; do for T in 2 5 10; do ECG_CFGS+=" snn_h${h}_T${T}"; done; done
MNIST_T="2 4 6 8 12"
GRU_H="16 32"
if [ "$MODE" = smoke ]; then      # board smoke on the most expensive configurations
    BOARD_A="ann_h128 snn_h64_T10"; BOARD_B="12"
else
    BOARD_A="$ECG_CFGS"; BOARD_B="$MNIST_T"
fi
# ECG final configuration (docs/results/ann_vs_snn_ecg/pc/manifest_final.json)
ECG_FINAL="--batch 64 --lr 0.001 --beta 0.9 --encoding direct --use-rr --morph diff --augment --weight-decay 0.2"
# GRU tuning configuration (runs/cv/gru_h16_bins3_e10.json, gru_h32_bins3_e10.json)
GRU_CFG="--batch 256 --lr 0.003 --beta 0.9 --encoding direct --use-rr --morph diff --augment --weight-decay 0.2 --readout-bins 3"

# step <id> <command...>: skip if done, time it, record the outcome
step() {
    local id="$1"; shift
    if [ -f "$STATE/$id.done" ]; then echo "skip  $id (done)"; return 0; fi
    local t0; t0=$(date +%s)
    echo "[$(date '+%F %T')] start $id"
    if "$@"; then
        local t=$(( $(date +%s) - t0 ))
        echo "$id,$t,ok,$(date '+%F %T')" >> "$TIMING"; touch "$STATE/$id.done"
        echo "[$(date '+%F %T')] done  $id (${t} s)"
    else
        echo "$id,$(( $(date +%s) - t0 )),FAILED,$(date '+%F %T')" >> "$TIMING"
        echo "[$(date '+%F %T')] FAILED $id"; return 1
    fi
}
# bg <id> <command...>: step in the background, at most $PAR at a time
bg() { while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done; step "$@" & }

# export into a temporary folder, then give every export a unique name
# (deploy_pynq.sh names the board-side folder after it): <prefix>_seed<k>
rename_exports() {      # rename_exports <tmp> <dest-root> <prefix>
    local tmp="$1" dst="$2" pre="$3" d
    [ -d "$tmp" ] || { echo "no $tmp"; return 1; }
    for d in "$tmp"/*_seed[0-9]*/; do
        d="${d%/}"; [ -d "$d" ] || continue
        local k="${d##*_seed}" i
        [[ "$k" =~ ^[0-9]+$ ]] || { echo "unexpected export name $d"; return 1; }
        rm -rf "${dst:?}/${pre}_seed$k"
        # OneDrive briefly locks freshly written folders: retry, then copy
        for i in 1 2 3 4 5 6; do mv "$d" "$dst/${pre}_seed$k" 2>/dev/null && break; sleep 5; done
        if [ -d "$d" ]; then
            cp -r "$d" "$dst/${pre}_seed$k" || return 1
            rm -rf "$d"
        fi
    done
    rm -rf "$tmp"
}
export_mnist() {        # export_mnist <tmp name> <checkpoint glob> <n images> <prefix>
    (cd "$MN" && rm -rf "pynq/export_$SUB/$1" &&
        "$PY_M" pynq/export_model.py $2 --n-samples "$3" --out "pynq/export_$SUB/$1") &&
    rename_exports "$MN/pynq/export_$SUB/$1" "$MN/pynq/export_$SUB" "$4"
}
export_gru() {          # export_gru <hidden>
    (cd "$ECG" && rm -rf "pynq/export_$SUB/tmp_gru_h$1" &&
        "$PY_E" pynq/export_stream.py runs/$SUB/gru_h$1/gru_ecg_seed*.pt \
            --sample "$NC" --sample-seed 0 --out "pynq/export_$SUB/tmp_gru_h$1") &&
    rename_exports "$ECG/pynq/export_$SUB/tmp_gru_h$1" "$ECG/pynq/export_$SUB" "ecgC_gru_h$1"
}
board() {               # board <export dir> <fw> <n>
    [ -d "$1" ] || { echo "no export $1"; return 1; }
    timeout 7200 bash "$DEPLOY" "$IP" "$1" -- --fw "$2" --n "$3" 2>&1 | grep -v "^  \["
    [ "${PIPESTATUS[0]}" = 0 ] && [ -f "$1/board_results_$2.csv" ]
}

# ---------------------------------------------------------------- A
train_ecg() {           # train_ecg <cfg> <seeds> [extra args]
    local cfg="$1" seeds="$2"; shift 2
    local m="${cfg%%_*}" h="${cfg#*_h}"; h="${h%%_*}"
    local T=10; [[ "$cfg" == *_T* ]] && T="${cfg##*_T}"
    (cd "$ECG" && "$PY_E" train_ecg.py --final --models "${m^^}" --hidden "$h" \
        --timesteps "$T" --seeds $seeds $(ep 10) $ECG_FINAL --threads $THREADS \
        --out-dir "runs/$SUB/ecg_$cfg" "$@" > "$LOG/train_ecg_$cfg.log" 2>&1)
}
export_ecg() {
    local cfg="$1"
    local tmp="$ECG/pynq/export_$SUB/tmp_$cfg"
    mkdir -p "$ECG/pynq/export_$SUB"; rm -rf "$tmp"
    (cd "$ECG" && "$PY_E" pynq/export_ecg.py --manifest "runs/$SUB/ecg_$cfg/manifest.json" \
        --input-scale "$S_X" --sample "$NA" --sample-seed 0 --out "pynq/export_$SUB/tmp_$cfg") &&
    rename_exports "$tmp" "$ECG/pynq/export_$SUB" "ecgA_$cfg"
}
job_A() {
    for c in $ECG_CFGS; do bg "A_train_$c" train_ecg "$c" "$S3"; done; wait
    for c in $ECG_CFGS; do step "A_export_$c" export_ecg "$c"; done
    # the board subset, saved once (int32 DS2 indices; every export carries a copy)
    local f; for f in "$ECG/pynq/export_$SUB"/ecgA_*_seed0/sample_idx.bin; do
        [ -f "$f" ] && cp "$f" "$RES/ecg_ds2_subset_${NA}_seed0.idx_int32.bin"; break
    done
    for c in $BOARD_A; do for s in $S3; do
        local e="$ECG/pynq/export_$SUB/ecgA_${c}_seed$s"
        step "A_board_${c}_s${s}_fast" board "$e" fast "$NA"
        [[ "$c" == snn_* ]] && step "A_board_${c}_s${s}_hw" board "$e" hw "$NA"
    done; done
    return 0
}
# ---------------------------------------------------------------- B
job_B() {
    for T in $MNIST_T; do
        bg "B_train_T$T" bash -c "cd '$MN' && '$PY_M' train_mnist.py --models SNN --timesteps $T \
            --seeds $S2 $(ep) --out-dir runs/$SUB/mnist_snn_T$T > '$LOG/train_mnist_T$T.log' 2>&1"
    done; wait
    for T in $MNIST_T; do
        step "B_export_T$T" export_mnist "tmp_T$T" "runs/$SUB/mnist_snn_T$T/snn_mnist_seed*.pt" \
            "$NB" "mnistB_snn_T$T"
    done
    for T in $BOARD_B; do for s in $S2; do for fw in fast hw; do
        step "B_board_T${T}_s${s}_$fw" board "$MN/pynq/export_$SUB/mnistB_snn_T${T}_seed$s" "$fw" "$NB"
    done; done; done
    return 0
}
# ---------------------------------------------------------------- C
job_C() {
    for h in $GRU_H; do
        bg "C_train_h$h" bash -c "cd '$ECG' && '$PY_E' train_ecg.py --final --models GRU \
            --hidden $h --seeds $S3 $(ep 10) $GRU_CFG --threads $THREADS \
            --out-dir runs/$SUB/gru_h$h > '$LOG/train_gru_h$h.log' 2>&1"
    done; wait
    for h in $GRU_H; do step "C_export_h$h" export_gru "$h"; done
    for h in $GRU_H; do for s in $S3; do
        step "C_board_h${h}_s$s" board "$ECG/pynq/export_$SUB/ecgC_gru_h${h}_seed$s" fast "$NC"
    done; done
    return 0
}
# ---------------------------------------------------------------- D
job_D() {
    step "D_train" bash -c "cd '$MN' && '$PY_M' train_mnist.py --seeds 2 $(ep) \
        --out-dir runs/$SUB/mnist_seed2 > '$LOG/train_mnist_seed2.log' 2>&1"
    for m in ann snn; do
        step "D_export_$m" export_mnist "tmp_D_$m" "runs/$SUB/mnist_seed2/${m}_mnist_seed2.pt" \
            "$ND" "mnistD_$m"
        step "D_board_$m" board "$MN/pynq/export_$SUB/mnistD_${m}_seed2" fast "$ND"
    done
    return 0
}
# ---------------------------------------------------------------- E
job_E() {
    local cfgs="$ECG_CFGS"; [ "$MODE" = smoke ] && cfgs="ann_h64 snn_h64_T10"
    for c in $cfgs; do
        bg "E_cv_$c" bash -c "cd '$ECG' && c=$c && m=\${c%%_*} && h=\${c#*_h} && h=\${h%%_*} && \
            T=10 && [[ \$c == *_T* ]] && T=\${c##*_T}; \
            '$PY_E' train_ecg.py --models \${m^^} --hidden \$h --timesteps \$T --seeds 0 \
            --cv-folds 5 $(ep 10) $ECG_FINAL --threads $THREADS --tag ecg_$c \
            --out-dir runs/$SUB/cv > '$LOG/cv_$c.log' 2>&1"
    done; wait
    return 0
}

echo "overnight.sh $MODE, jobs: $JOBS, results: $RES, board $IP, s_x = $S_X"
for j in $JOBS; do
    {
        echo "=== job $j start $(date '+%F %T')"
        "job_$j"
        echo "=== job $j end $(date '+%F %T')"
    } >> "$LOG/$j.log" 2>&1
    "$PY_E" "$HERE/summarize_overnight.py" --res "$RES_W" --mode "$MODE" --job "$j" \
        >> "$LOG/$j.log" 2>&1 || echo "summary of job $j failed (see $LOG/$j.log)"
    echo "job $j finished $(date '+%F %T')"
done
"$PY_E" "$HERE/summarize_overnight.py" --res "$RES_W" --mode "$MODE" >> "$LOG/summary.log" 2>&1
echo "all done $(date '+%F %T')"
