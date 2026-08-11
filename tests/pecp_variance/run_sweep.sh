#!/usr/bin/env bash
# ============================================================================
# PECP variance characterization — 10 seeds × 10 objects
# ============================================================================
#
# Runs test.py per (object, seed) pair to measure run-to-run variance of the
# ADD(-S) metric introduced by PECP's unseeded random.choice subset sampling
# and OpenCV's unseeded solvePnPRansac. Uses --eval_seed to pin each run's
# RNG state so individual runs are reproducible; does NOT alter the eval
# algorithm, metric, or PECP iteration count.
#
# Layout:
#   {SWEEP_ROOT}/seed{k}/obj{N}/rgb_bop.csv
#                              rgb_detailed.csv
#                              metadata.json
#
# obj21 (paper obj9) is evaluated twice:
#   - obj21             (all scenes: 5 + 29)  — default
#   - obj21_excl_scene29 (scene 5 only)       — --scene_ids 000005
#
# Total: 10 seeds × (10 objects + 1 excl-29 variant) = 110 test.py invocations.
#
# Usage:
#   bash tests/pecp_variance/run_sweep.sh                # 10 seeds, PECP on
#   NUM_SEEDS=3 bash tests/pecp_variance/run_sweep.sh    # smaller sweep
# ============================================================================

set -euo pipefail

BOP_ROOT_BASE="${BOP_ROOT_BASE:-data/RTLESS_BOP}"
CKPT_ROOT="${CKPT_ROOT:-model/paper_checkpoints}"
GIN_CONFIG="${GIN_CONFIG:-configs/contourpose_bop.gin}"
NUM_SEEDS="${NUM_SEEDS:-10}"
SWEEP_ROOT="${SWEEP_ROOT:-results/rtless_bop_variance/$(date +%y%m%d_%H%M)_pecp}"

mkdir -p "$SWEEP_ROOT"
LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================="
echo "  PECP variance sweep"
echo "  Sweep root : $SWEEP_ROOT"
echo "  Seeds      : 0..$((NUM_SEEDS - 1))"
echo "  BOP roots  : $BOP_ROOT_BASE"
echo "  Checkpoints: $CKPT_ROOT"
echo "=================================================="

OBJECTS="obj1 obj2 obj3 obj6 obj7 obj13 obj16 obj18 obj21 obj32"
FAILED=()

for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED_DIR="${SWEEP_ROOT}/seed${SEED}"
    mkdir -p "$SEED_DIR"

    echo ""
    echo "=================================================="
    echo "  SEED ${SEED} / $((NUM_SEEDS - 1))"
    echo "=================================================="

    for CLASS_TYPE in $OBJECTS; do
        OBJ_ID="${CLASS_TYPE#obj}"
        BOP_ROOT="${BOP_ROOT_BASE}/${CLASS_TYPE}"
        CKPT="${CKPT_ROOT}/${CLASS_TYPE}/150.pkl"
        OUT_DIR="${SEED_DIR}/${CLASS_TYPE}"

        echo ""
        echo "  seed=${SEED}  ${CLASS_TYPE}  (obj_id=${OBJ_ID})"

        if [[ ! -d "$BOP_ROOT" ]] || [[ ! -f "$CKPT" ]]; then
            echo "    [SKIP] missing BOP root or checkpoint"
            FAILED+=("seed${SEED}/${CLASS_TYPE}")
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
            --eval_seed       "$SEED" \
            || { echo "    [FAILED] $CLASS_TYPE seed=${SEED}"; FAILED+=("seed${SEED}/${CLASS_TYPE}"); }
    done

    # obj21 excl-scene-29 (scene 5 only)
    EXCL_OUT_DIR="${SEED_DIR}/obj21_excl_scene29"
    echo ""
    echo "  seed=${SEED}  obj21_excl_scene29  (scene 5 only)"
    if [[ -d "${BOP_ROOT_BASE}/obj21" ]] && [[ -f "${CKPT_ROOT}/obj21/150.pkl" ]]; then
        python test.py \
            --bop_root        "${BOP_ROOT_BASE}/obj21" \
            --data_root       "${BOP_ROOT_BASE}/obj21" \
            --class_type      obj21 \
            --obj_id          21 \
            --checkpoint_path "${CKPT_ROOT}/obj21/150.pkl" \
            --output_dir      "$EXCL_OUT_DIR" \
            --gin_config      "$GIN_CONFIG" \
            --img_size 480 640 \
            --no_wandb \
            --scene_ids       000005 \
            --eval_seed       "$SEED" \
            || { echo "    [FAILED] obj21_excl_scene29 seed=${SEED}"; FAILED+=("seed${SEED}/obj21_excl_scene29"); }
    else
        echo "    [SKIP] obj21 missing"
        FAILED+=("seed${SEED}/obj21_excl_scene29")
    fi
done

echo ""
echo "=================================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '%s\n' "${FAILED[@]}" > "${SWEEP_ROOT}/skipped.log"
    echo "  FAILED/SKIPPED (see ${SWEEP_ROOT}/skipped.log):"
    printf '    %s\n' "${FAILED[@]}"
else
    echo "No runs were skipped." > "${SWEEP_ROOT}/skipped.log"
    echo "  All ${NUM_SEEDS} × 11 runs completed successfully."
fi
echo "=================================================="

echo ""
echo "Aggregating variance across seeds..."
python tests/pecp_variance/aggregate.py --sweep_root "$SWEEP_ROOT" || \
    echo "  [warn] aggregation failed — raw seed*/obj*/ CSVs are still under $SWEEP_ROOT"

echo ""
echo "  Done. Sweep root: $SWEEP_ROOT"
echo "=================================================="
