#!/bin/bash
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
BOP_ROOT="/data/RTLESS_BOP"
GIN_CONFIG="configs/contourpose_bop.gin"

# Set to the directory containing background images (JPEGs/PNGs).
# The original ContourPose paper uses SUN2012pascalformat/JPEGImages/.
# Leave empty to skip background mixing (object renders on black background).
BACKGROUND_DIR="/data/SUN2012pascalformat/JPEGImages"

# ── Gin parameter overrides ────────────────────────────────────────────────────
# Add one entry per override. These are applied to EVERY object run.
# Syntax matches gin-config bindings: "function.param = value"
# Leave the array empty to use defaults from the gin config file.
GIN_PARAMS=(
    #"main.epochs = 1"
    #"main.val_interval = 1"
    #"main.viz_interval = 1"
    )
# "main.lr_step_size = 20"
    # "main.lr_gamma = 0.5"
    # "main.batch_size = 8"
    # "main.lr = 0.1"


# ── Object list ────────────────────────────────────────────────────────────────
# Format: "class_type obj_id"
# These are the 10 trainable objects (keypoints/ and Valid3D/ entries).
OBJECTS=(
    # "obj1   1"
    # "obj2   2"
    # "obj3   3"
    # "obj6   6"
    # "obj7   7"
    "obj13  13"
    "obj16  16"
    "obj18  18"
    "obj21  21"
    "obj32  32"
)

# ── Build --gin_param flags ────────────────────────────────────────────────────
# argparse nargs="+" only keeps the last occurrence when the flag is repeated,
# so all params must be passed as a single --gin_param flag with multiple values.
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
