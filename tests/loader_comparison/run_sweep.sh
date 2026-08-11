#!/usr/bin/env bash
# ============================================================================
# Legacy MyDataset vs BOP DALI train-loader comparison sweep
# ============================================================================
#
# For each object, samples N poses/K/images from BOTH loaders and produces a
# distributional comparison report. Configures each loader to match how it's
# actually used at training time:
#   - Legacy: MyDataset(data_path=CONTOURPOSE_ROOT). Reads SUN2012 images from
#     $CONTOURPOSE_ROOT/SUN2012pascalformat/JPEGImages internally.
#   - BOP DALI: BOPDALIDataset with --background_dir pointing at the SAME
#     SUN2012 JPEG dir so load-time background compositing runs (which
#     matches the user's training config).
#
# Outputs land under:
#   {SWEEP_ROOT}/{obj}/legacy.npz
#   {SWEEP_ROOT}/{obj}/bop_dali.npz
#   {SWEEP_ROOT}/{obj}/report/comparison_report.txt
#   {SWEEP_ROOT}/{obj}/report/*.png
#   {SWEEP_ROOT}/sweep.log
#
# Env var overrides (defaults tuned for this container layout):
#   CONTOURPOSE_ROOT                 /data/Datasets/ContourPose
#   LEGACY_RENDER_DIR_TEMPLATE       $CONTOURPOSE_ROOT/Train_Scenes/Real/renders/{obj}
#   LEGACY_RENDER_EDGE_DIR_TEMPLATE  $CONTOURPOSE_ROOT/Train_Scenes/Real/gtEdge/{obj}
#   PBR_ROOT_TEMPLATE                /data/ContourPose_PBR/{obj}_train_pbr
#     PBR substitutions:
#       {obj}    → obj1, obj2, ...   (bare class name)
#       {obj_id} → 000001, 000002... (zero-padded 6-digit BOP obj id)
#   BOP_KEYPOINTS_ROOT               data/RTLESS_BOP
#   LEGACY_KP_DIR                    keypoints
#   NUM_SAMPLES                      1000
#   BATCH_SIZE                       8
#   NUM_WORKERS                      4
#   SEED                             0
#   OBJECTS                          "obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32"
#   SWEEP_ROOT                       results/loader_compare/{ts}
#
# Usage:
#   bash tests/loader_comparison/run_sweep.sh
#   OBJECTS="obj1" bash tests/loader_comparison/run_sweep.sh   # single-obj smoke
# ============================================================================

set -euo pipefail

CONTOURPOSE_ROOT="${CONTOURPOSE_ROOT:-/data/Datasets/ContourPose}"

# Assign defaults in two steps — bash's ${var:-default} terminates at the first
# `}`, which corrupts defaults containing `{obj}`/`{obj_id}` placeholders.
_pbr_default='/data/ContourPose_PBR/{obj}_train_pbr'
_legacy_render_default="${CONTOURPOSE_ROOT}/Train_Scenes/Real/renders/{obj}"
_legacy_edge_default="${CONTOURPOSE_ROOT}/Train_Scenes/Real/gtEdge/{obj}"
PBR_ROOT_TEMPLATE="${PBR_ROOT_TEMPLATE:-$_pbr_default}"
LEGACY_RENDER_DIR_TEMPLATE="${LEGACY_RENDER_DIR_TEMPLATE:-$_legacy_render_default}"
LEGACY_RENDER_EDGE_DIR_TEMPLATE="${LEGACY_RENDER_EDGE_DIR_TEMPLATE:-$_legacy_edge_default}"
BOP_KEYPOINTS_ROOT="${BOP_KEYPOINTS_ROOT:-data/RTLESS_BOP}"
LEGACY_KP_DIR="${LEGACY_KP_DIR:-keypoints}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
OBJECTS="${OBJECTS:-obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32}"

TIMESTAMP="$(date +%y%m%d_%H%M)"
SWEEP_ROOT="${SWEEP_ROOT:-results/loader_compare/${TIMESTAMP}}"

# SUN2012 JPEGs — legacy reads them via CONTOURPOSE_ROOT/SUN2012pascalformat;
# BOP DALI needs the explicit --background_dir argument to layer them at load.
SUN_JPEG_DIR="${CONTOURPOSE_ROOT}/SUN2012pascalformat/JPEGImages"

mkdir -p "$SWEEP_ROOT"
LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================="
echo "  Loader comparison sweep — legacy synth vs BOP DALI PBR (synth-only)"
echo "  Sweep root                : $SWEEP_ROOT"
echo "  ContourPose root          : $CONTOURPOSE_ROOT"
echo "  Legacy render template    : $LEGACY_RENDER_DIR_TEMPLATE"
echo "  Legacy edge template      : $LEGACY_RENDER_EDGE_DIR_TEMPLATE"
echo "  BOP PBR template          : $PBR_ROOT_TEMPLATE"
echo "  BOP keypoints root        : $BOP_KEYPOINTS_ROOT"
echo "  SUN2012 JPEG dir          : $SUN_JPEG_DIR"
echo "  Samples per loader        : $NUM_SAMPLES"
echo "  Objects                   : $OBJECTS"
echo "=================================================="

if [[ ! -d "$SUN_JPEG_DIR" ]]; then
    echo "  [warn] SUN2012 JPEG dir not found: $SUN_JPEG_DIR"
    echo "  [warn] Legacy loader will crash on random_background(); BOP DALI will skip bg compositing."
fi

FAILED=()

for OBJ in $OBJECTS; do
    echo ""
    echo "--------------------------------------------------"
    echo "  $OBJ"
    echo "--------------------------------------------------"

    OBJ_DIR="${SWEEP_ROOT}/${OBJ}"
    mkdir -p "$OBJ_DIR"

    OBJ_ID="${OBJ#obj}"
    OBJ_ID_PAD=$(printf "%06d" "$OBJ_ID")
    PBR_ROOT="${PBR_ROOT_TEMPLATE//\{obj\}/$OBJ}"
    PBR_ROOT="${PBR_ROOT//\{obj_id\}/$OBJ_ID_PAD}"
    LEGACY_KP="${LEGACY_KP_DIR}/${OBJ}.txt"
    BOP_KP="${BOP_KEYPOINTS_ROOT}/${OBJ}/keypoints/${OBJ}_mm.txt"

    # ---- Legacy sample ----
    if [[ ! -f "$LEGACY_KP" ]]; then
        echo "  [SKIP] legacy keypoints missing: $LEGACY_KP"
        FAILED+=("${OBJ} (legacy kp missing)")
        continue
    fi

    LEGACY_RENDER_DIR="${LEGACY_RENDER_DIR_TEMPLATE//\{obj\}/$OBJ}"
    LEGACY_RENDER_EDGE_DIR="${LEGACY_RENDER_EDGE_DIR_TEMPLATE//\{obj\}/$OBJ}"

    if [[ ! -d "$LEGACY_RENDER_DIR" ]]; then
        echo "  [SKIP] legacy render dir missing: $LEGACY_RENDER_DIR"
        FAILED+=("${OBJ} (legacy render dir missing)")
        continue
    fi

    echo ""
    echo "  [1/3] sampling legacy (synth renders only)..."
    echo "        render_dir      = $LEGACY_RENDER_DIR"
    echo "        render_edge_dir = $LEGACY_RENDER_EDGE_DIR"
    python tests/loader_comparison/sample_poses.py \
        --loader           legacy \
        --class_type       "$OBJ" \
        --data_path        "$CONTOURPOSE_ROOT" \
        --render_dir       "$LEGACY_RENDER_DIR" \
        --render_edge_dir  "$LEGACY_RENDER_EDGE_DIR" \
        --sun_path         "${CONTOURPOSE_ROOT}/SUN2012pascalformat" \
        --keypoints_file   "$LEGACY_KP" \
        --n                "$NUM_SAMPLES" \
        --batch_size       "$BATCH_SIZE" \
        --num_workers      "$NUM_WORKERS" \
        --seed             "$SEED" \
        --out              "${OBJ_DIR}/legacy.npz" \
        || { echo "  [FAILED] legacy sample for $OBJ"; FAILED+=("${OBJ} (legacy)"); continue; }

    # ---- BOP DALI sample ----
    if [[ ! -f "$BOP_KP" ]]; then
        echo "  [SKIP] BOP _mm keypoints missing: $BOP_KP"
        FAILED+=("${OBJ} (bop kp missing)")
        continue
    fi
    if [[ ! -d "$PBR_ROOT" ]]; then
        echo "  [SKIP] PBR train dir missing: $PBR_ROOT"
        FAILED+=("${OBJ} (pbr dir missing)")
        continue
    fi

    echo ""
    echo "  [2/3] sampling BOP DALI (with SUN2012 background compositing)..."
    python tests/loader_comparison/sample_poses.py \
        --loader        bop_dali \
        --class_type    "$OBJ" \
        --bop_root      "$PBR_ROOT" \
        --background_dir "$SUN_JPEG_DIR" \
        --keypoints_file "$BOP_KP" \
        --n             "$NUM_SAMPLES" \
        --batch_size    "$BATCH_SIZE" \
        --num_workers   "$NUM_WORKERS" \
        --seed          "$SEED" \
        --out           "${OBJ_DIR}/bop_dali.npz" \
        || { echo "  [FAILED] bop_dali sample for $OBJ"; FAILED+=("${OBJ} (bop_dali)"); continue; }

    # ---- Compare ----
    echo ""
    echo "  [3/3] comparing..."
    python tests/loader_comparison/compare_distributions.py \
        --legacy   "${OBJ_DIR}/legacy.npz" \
        --bop_dali "${OBJ_DIR}/bop_dali.npz" \
        --out_dir  "${OBJ_DIR}/report" \
        || { echo "  [FAILED] comparison for $OBJ"; FAILED+=("${OBJ} (compare)"); continue; }
done

echo ""
echo "=================================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '%s\n' "${FAILED[@]}" > "${SWEEP_ROOT}/skipped.log"
    echo "  FAILED/SKIPPED (see ${SWEEP_ROOT}/skipped.log):"
    printf '    %s\n' "${FAILED[@]}"
else
    echo "No objects were skipped." > "${SWEEP_ROOT}/skipped.log"
    echo "  All objects completed successfully."
fi
echo "=================================================="
echo ""
echo "  Reports: ${SWEEP_ROOT}/{obj}/report/comparison_report.txt"
echo "  Plots  : ${SWEEP_ROOT}/{obj}/report/*.png"
echo "=================================================="
