#!/usr/bin/env bash
set -e

run_full_layer_pipeline() {
    local DATASET_ID="$1"
    local LAYER_ID="$2"
    local POOL_DIR="$3"

    bash run_1_auto_golden.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_2a_train_fab_no_gt.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_2b_infer_fab_no_gt.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_3_split_eval_results.sh "${DATASET_ID}" "${LAYER_ID}"
}

# Layer 28
run_full_layer_pipeline W639 TMQB28 "/workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status"

# Layer 50 (TMTN86)
run_full_layer_pipeline W639 TMTN86 "/workspace/dataset/DA_Wafer_use/W639/TMTN86/other_status"

# Layer 86 (TMUN87)
run_full_layer_pipeline W639 TMUN87 "/workspace/dataset/DA_Wafer_use/W639/TMUN87/other_status"

echo "✅ All three layers completed (Golden -> Train -> Infer -> Split Eval)."