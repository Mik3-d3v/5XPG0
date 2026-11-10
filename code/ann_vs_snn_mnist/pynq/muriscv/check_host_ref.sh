#!/bin/bash
# check_host_ref.sh — run build/host_ref (the muRISCV-NN library + glue built
# for the PC by build_muriscv.sh) on ANN3 exports and check that its logits
# equal ref_logits.bin (the Python model of the library) byte for byte.
# Runs host_ref in Docker (ubuntu:22.04, where it was built).
# Usage: check_host_ref.sh <export_dir>...   (paths inside the repository)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
[ $# -gt 0 ] || { echo "usage: $0 <export_dir>..." >&2; exit 2; }
rel=()
for d in "$@"; do
    a="$(cd "$d" && pwd)"
    rel+=("${a#"$REPO"/}")
done
MSYS_NO_PATHCONV=1 docker run -i --rm -v "$REPO_W:/repo" -w /repo ubuntu:22.04 \
    bash -s -- "${rel[@]}" <<'EOF'
fail=0
for d in "$@"; do
    code/ann_vs_snn_mnist/pynq/muriscv/build/host_ref "$d" >/dev/null
    if cmp -s "$d/host_outputs.bin" "$d/ref_logits.bin"; then
        echo "OK    $d: host_ref logits == ref_logits.bin ($(( $(stat -c %s "$d/ref_logits.bin") / 4 )) values)"
    else
        echo "DIFF  $d: host_ref logits differ from ref_logits.bin"; fail=1
    fi
done
exit $fail
EOF
