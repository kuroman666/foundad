#!/usr/bin/env bash
# 檔案位置: FoundAD/run_all_layers.sh
# 用途: 依序執行三個 Layer 的完整四階段流程 (支援在下方直接設定每個 Layer 要拆成幾包與每包張數)
set -e

run_full_layer_pipeline() {
    local DATASET_ID="$1"
    local LAYER_ID="$2"
    local POOL_DIR="$3"
    local NUM_PACKS="${4:-3}"
    local PACK_COUNTS="${5:-}"

    bash run_1_auto_golden.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_2_train_fab_no_gt.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_3_infer_fab_no_gt.sh "${DATASET_ID}" "${LAYER_ID}" "${POOL_DIR}"
    bash run_4_split_eval_results.sh "${DATASET_ID}" "${LAYER_ID}" "" "${NUM_PACKS}" "${PACK_COUNTS}"
}

# 參數說明: run_full_layer_pipeline <DATASET_ID> <LAYER_ID> <POOL_DIR> [切成幾包 NUM_PACKS] [每包張數 PACK_COUNTS (留空 "" 則自動均分)]
# Layer 28 (TMQB28)
run_full_layer_pipeline W639 TMQB28 "/workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status" 3 ""

# Layer 50 (TMTN86)
run_full_layer_pipeline W639 TMTN86 "/workspace/dataset/DA_Wafer_use/W639/TMTN86/other_status" 3 ""

# Layer 86 (TMUN87)
run_full_layer_pipeline W639 TMUN87 "/workspace/dataset/DA_Wafer_use/W639/TMUN87/other_status" 3 ""

echo "✅ All three layers completed (Golden -> Train -> Infer -> Split Eval)."