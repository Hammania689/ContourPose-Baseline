#!/usr/bin/env bash
# ============================================================================
# RT-Less BOP/DALI evaluation — SEPARATE from eval_rtless_authors.sh
# ============================================================================
#
# This script:
#   - Uses the BOP-format DALI test loader (test.py → eval_spectra_pose.py)
#   - Points at per-object BOP roots produced by scripts/rtless_test_to_bop.py
#     (data/RTLESS_BOP/{obj}/test/{scene_id}/rgb/... etc.)
#   - Loops over all 10 trainable objects, one test.py invocation per object
#   - Writes results to results/rtless_bop/ (or results/rtless_bop_pecp/)
#
# Contrast with eval_rtless_authors.sh:
#   - eval_rtless_authors.sh   → LEGACY pipeline (RT-Less native, no DALI)
#                                Uses eval_legacy_all_scenes.py + eval.py + MyDataset
#                                Reads photo_cut/, mask/, gt.yml, Intrinsic.yml
#   - eval_rtless_bop.sh (this) → BOP/DALI pipeline (converted data)
#                                Uses test.py + eval_spectra_pose.py + BOPTestDALIDataset
#                                Reads rgb/, mask_visib/, scene_gt.json, scene_camera.json
#
# The two produce SEPARATE result trees so numbers can be compared directly.
#
# Usage:
#   bash eval_rtless_bop.sh              # no PECP (default)
#   bash eval_rtless_bop.sh --pecp       # with PECP refinement
#
# Prerequisite:
#   Per-object BOP roots must already exist. Run first:
#     python scripts/rtless_test_to_bop.py --all
# ============================================================================

set -euo pipefail

BOP_ROOT_BASE="${BOP_ROOT_BASE:-data/RTLESS_BOP}"     # per-object subdirs live here
CKPT_ROOT="${CKPT_ROOT:-model/paper_checkpoints}"
GIN_CONFIG="${GIN_CONFIG:-configs/contourpose_bop.gin}"
OUTPUT_ROOT="results/rtless_bop"

PECP_FLAG="--no_pecp"
PECP_SUFFIX=""
if [[ "${1:-}" == "--pecp" ]]; then
    PECP_FLAG=""
    PECP_SUFFIX="_pecp"
fi

# One timestamped run dir per invocation. Layout mirrors the legacy
# results/rtless/authors_checkpoints/{ts}_pecp/ tree so downstream tooling
# (table generation, plots) works unchanged.
TIMESTAMP="$(date +%y%m%d_%H%M)"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}${PECP_SUFFIX}"
mkdir -p "$RUN_DIR"
RUN_LOG="${RUN_DIR}/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=================================================="
echo "  RT-Less evaluation — BOP/DALI pipeline"
echo "  Per-object BOP roots under: $BOP_ROOT_BASE"
echo "  Checkpoints              : $CKPT_ROOT"
echo "  Run dir                  : $RUN_DIR"
echo "  PECP                     : ${PECP_FLAG:-(enabled)}"
echo "=================================================="

FAILED=()

for CLASS_TYPE in obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32; do
    OBJ_ID="${CLASS_TYPE#obj}"
    BOP_ROOT="${BOP_ROOT_BASE}/${CLASS_TYPE}"
    CKPT="${CKPT_ROOT}/${CLASS_TYPE}/150.pkl"
    OUT_DIR="${RUN_DIR}/${CLASS_TYPE}"

    echo ""
    echo "--------------------------------------------------"
    echo "  $CLASS_TYPE  (obj_id=${OBJ_ID})  bop_root=${BOP_ROOT}"
    echo "--------------------------------------------------"

    if [[ ! -d "$BOP_ROOT" ]]; then
        echo "  [SKIP] per-object BOP root missing: $BOP_ROOT"
        echo "         Run: python scripts/rtless_test_to_bop.py --object $CLASS_TYPE"
        FAILED+=("$CLASS_TYPE (no BOP root)")
        continue
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "  [SKIP] checkpoint not found: $CKPT"
        FAILED+=("$CLASS_TYPE (no checkpoint)")
        continue
    fi

    python test.py \
        --bop_root        "$BOP_ROOT" \
        --data_root       "$BOP_ROOT" \
        --class_type      "$CLASS_TYPE" \
        --obj_id          "$OBJ_ID" \
        --checkpoint_path "$CKPT" \
        --output_dir      "$OUT_DIR" \
        --gin_config      "$GIN_CONFIG" \
        --img_size 480 640 \
        --no_wandb \
        ${PECP_FLAG} \
        || { echo "  [FAILED] $CLASS_TYPE"; FAILED+=("$CLASS_TYPE"); }
done

echo ""
echo "=================================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '%s\n' "${FAILED[@]}" > "${RUN_DIR}/skipped.log"
    echo "  FAILED/SKIPPED (see ${RUN_DIR}/skipped.log):"
    printf '    %s\n' "${FAILED[@]}"
else
    echo "No objects were skipped." > "${RUN_DIR}/skipped.log"
    echo "  All objects completed successfully."
fi
echo "=================================================="

# Aggregate obj*/masked_detailed.csv → results.csv + summary.txt
echo ""
echo "Aggregating results..."
python scripts/aggregate_bop_results.py --run_dir "$RUN_DIR" || \
    echo "  [warn] aggregation failed — raw obj*/ CSVs are still under $RUN_DIR"

echo ""
echo "  Done. Run dir: $RUN_DIR"
echo "  To compare against the legacy pipeline:"
echo "    results/rtless/authors_checkpoints/260609_0543_pecp/ (legacy PECP)"
echo "    $RUN_DIR/                                            (BOP/DALI)"
echo "=================================================="
