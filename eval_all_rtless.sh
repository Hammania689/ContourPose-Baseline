#!/usr/bin/env bash
# Evaluate paper checkpoints on the RT-Less test dataset for all 10 objects.
#
# Usage:
#   bash eval_all_rtless.sh              # PECP enabled (default)
#   bash eval_all_rtless.sh --no_pecp   # disable PECP

set -euo pipefail

BOP_ROOT="${BOP_ROOT:-data/RTLESS_BOP}"
# Keypoints and Valid3D live in the repo's local data copy, not under BOP_ROOT
DATA_ROOT="${DATA_ROOT:-data/RTLESS_BOP}"
CKPT_ROOT="${CKPT_ROOT:-model/paper_checkpoints}"
GIN_CONFIG="configs/contourpose_bop.gin"
OUTPUT_ROOT="results/rtless_paper"

# Pass --no_pecp to disable PECP refinement
PECP_FLAG=""
if [[ "${1:-}" == "--no_pecp" ]]; then
    PECP_FLAG="--no_pecp"
    OUTPUT_ROOT="results/rtless_paper_no_pecp"
fi

# class_type → BOP obj_id
declare -A OBJ_IDS=(
    [obj1]=1
    [obj2]=2
    [obj3]=3
    [obj6]=6
    [obj7]=7
    [obj13]=13
    [obj16]=16
    [obj18]=18
    [obj21]=21
    [obj32]=32
)

echo "=================================================="
echo "  RT-Less evaluation — paper checkpoints"
echo "  BOP root : $BOP_ROOT"
echo "  Data root: $DATA_ROOT"
echo "  Output   : $OUTPUT_ROOT"
echo "  PECP     : ${PECP_FLAG:-(enabled)}"
echo "=================================================="

FAILED=()

for CLASS_TYPE in obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32; do
    OBJ_ID="${OBJ_IDS[$CLASS_TYPE]}"
    CKPT="${CKPT_ROOT}/${CLASS_TYPE}/150.pkl"
    OUT_DIR="${OUTPUT_ROOT}/${CLASS_TYPE}"

    echo ""
    echo "--------------------------------------------------"
    echo "  $CLASS_TYPE  (obj_id=${OBJ_ID})"
    echo "--------------------------------------------------"

    if [[ ! -f "$CKPT" ]]; then
        echo "  [SKIP] checkpoint not found: $CKPT"
        FAILED+=("$CLASS_TYPE (no checkpoint)")
        continue
    fi

    python test.py \
        --bop_root   "$BOP_ROOT" \
        --data_root  "$DATA_ROOT" \
        --class_type "$CLASS_TYPE" \
        --obj_id     "$OBJ_ID" \
        --checkpoint_path "$CKPT" \
        --output_dir "$OUT_DIR" \
        --gin_config "$GIN_CONFIG" \
        --img_size 480 640 \
        --no_wandb \
        ${PECP_FLAG} \
        || { echo "  [FAILED] $CLASS_TYPE"; FAILED+=("$CLASS_TYPE"); }
done

echo ""
echo "=================================================="
echo "  Done. Results in: $OUTPUT_ROOT"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED: ${FAILED[*]}"
else
    echo "  All objects completed successfully."
fi
echo "=================================================="
