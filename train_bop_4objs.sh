#!/bin/bash
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
# Dataset root: contains train_pbr/{obj_id:06d}/, keypoints/, Valid3D/, models/.
# train_bop.py appends train_pbr/{obj_id:06d} internally, so point at the root.
BOP_ROOT="/contourpose-baseline-4090/data/RTLESS_BOP"
GIN_CONFIG="configs/contourpose_bop.gin"

BACKGROUND_DIR="/data/SUN2012pascalformat/JPEGImages"

# ── Gin parameter overrides ────────────────────────────────────────────────────
# Uncomment for a smoke run before committing to a full 150-epoch schedule.
GIN_PARAMS=(
    # "main.epochs = 2"
    # "main.val_interval = 1"
    # "main.viz_interval = 1"
)

# ── Object list ────────────────────────────────────────────────────────────────
# Format: "class_type obj_id"
OBJECTS=(
    "obj2   2"
    "obj6   6"
    "obj13  13"
    "obj18  18"
)

# ── Build --gin_param flags ────────────────────────────────────────────────────
GIN_PARAM_FLAGS=()
if [ ${#GIN_PARAMS[@]} -gt 0 ]; then
    GIN_PARAM_FLAGS=(--gin_param)
    for param in "${GIN_PARAMS[@]}"; do
        GIN_PARAM_FLAGS+=("$param")
    done
fi

# ── Run ────────────────────────────────────────────────────────────────────────
total=${#OBJECTS[@]}
idx=0

for entry in "${OBJECTS[@]}"; do
    idx=$((idx + 1))
    class_type=$(echo "$entry" | awk '{print $1}')
    obj_id=$(echo "$entry"     | awk '{print $2}')

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Object ${idx}/${total}: ${class_type}  (obj_id=${obj_id})"
    echo "════════════════════════════════════════════════════════════════"

    python train_bop.py \
        --class_type   "$class_type" \
        --obj_id       "$obj_id" \
        --bop_root     "$BOP_ROOT" \
        --gin_config   "$GIN_CONFIG" \
        ${BACKGROUND_DIR:+--background_dir "$BACKGROUND_DIR"} \
        "${GIN_PARAM_FLAGS[@]}" \
        "$@"
done

echo ""
echo "All ${total} objects finished."
