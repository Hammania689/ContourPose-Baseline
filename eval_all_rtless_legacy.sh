#!/usr/bin/env bash
# Evaluate paper checkpoints using the legacy eval.py pipeline (no BOP format).
# Uses test_legacy_eval.py + original ContourPose dataset layout.
#
# Usage:
#   bash eval_all_rtless_legacy.sh              # standard PnP (no PECP)
#   bash eval_all_rtless_legacy.sh --pecp       # enable PECP

set -euo pipefail

DATA_PATH="${DATA_PATH:-data/ContourPose_Original}"
CKPT_ROOT="${CKPT_ROOT:-model/paper_checkpoints}"
OUTPUT_ROOT="results/rtless_legacy"

# Pass --pecp to enable PECP refinement (off by default to match original main.py)
PECP_FLAG="--no_pecp"
if [[ "${1:-}" == "--pecp" ]]; then
    PECP_FLAG=""
    OUTPUT_ROOT="results/rtless_legacy_pecp"
fi

# class_type → scene and object index (from sceneObjs.yml / README)
# Format: "scene:index"  (index = 0-based position of object in scene's list)
#
# Several objects appear in multiple scenes; the choice below is the first
# scene in sceneObjs.yml where the object appears, except obj1/obj2/obj3
# which all share scene 13 (confirmed by README). Alternatives:
#   obj2  → 18:1, 25:3
#   obj3  → 7:0, 24:1, 32:2
#   obj6  → 23:1, 24:2, 31:0
#   obj7  → 1:3, 31:1
#   obj13 → 25:0, 32:0
#   obj16 → 8:0
#   obj18 → 21:2, 25:2
#   obj21 → 29:0
#   obj32 → (only scene 30)
# If your previous experiments used a different scene, override via DATA_PATH
# and adjust SCENE_INDEX here to match.
declare -A SCENE_INDEX=(
    [obj1]="13:2"
    [obj2]="13:1"
    [obj3]="13:0"
    [obj6]="3:0"
    [obj7]="26:2"
    [obj13]="10:2"
    [obj16]="4:0"
    [obj18]="11:2"
    [obj21]="5:2"
    [obj32]="30:1"
)

echo "=================================================="
echo "  RT-Less evaluation — legacy eval.py pipeline"
echo "  Data path: $DATA_PATH"
echo "  Output   : $OUTPUT_ROOT"
echo "  PECP     : ${PECP_FLAG:-(enabled)}"
echo "=================================================="

FAILED=()

for CLASS_TYPE in obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32; do
    SCENE_IDX="${SCENE_INDEX[$CLASS_TYPE]}"
    SCENE="${SCENE_IDX%%:*}"
    INDEX="${SCENE_IDX##*:}"
    CKPT="${CKPT_ROOT}/${CLASS_TYPE}/150.pkl"
    OUT_DIR="${OUTPUT_ROOT}/${CLASS_TYPE}"

    echo ""
    echo "--------------------------------------------------"
    echo "  $CLASS_TYPE  (scene=${SCENE}, index=${INDEX})"
    echo "--------------------------------------------------"

    if [[ ! -f "$CKPT" ]]; then
        echo "  [SKIP] checkpoint not found: $CKPT"
        FAILED+=("$CLASS_TYPE (no checkpoint)")
        continue
    fi

    python test_legacy_eval.py \
        --class_type      "$CLASS_TYPE" \
        --checkpoint_path "$CKPT" \
        --data_path       "$DATA_PATH" \
        --scene           "$SCENE" \
        --index           "$INDEX" \
        --output_dir      "$OUT_DIR" \
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
