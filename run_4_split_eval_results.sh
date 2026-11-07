#!/usr/bin/env bash
# 檔案位置: FoundAD/run_3_split_eval_results.sh
# 用途: 單獨執行 3_split_eval_results.py (將推論結果拆分為多份獨立 Eval 報表與圖片包)
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
# 若傳入第 3 個參數則直接作為來源目錄，否則自動抓取該 Layer 的推論輸出根目錄 (程式會自動找底下最新時間戳記)
SRC_DIR="${3:-./wafer_escape_results/${DATASET_ID}/${LAYER_ID}}"

# 拆分後輸出目錄 (留空 "" 則預設輸出在該次時間戳記目錄下的 split_eval_3packs/)
OUT_DIR=""

# 拆分數量與比例/張數設定 (請注意：COUNTS 與 RATIOS 請擇一填寫，勿同時填寫！)
NUM_PACKS=3              # 拆分成幾包 Eval 結果 (預設 3 包)
COUNTS=""                # 依【每包張數】拆分：例如填 "50" (每包50張) 或 "40,40,20"；若用比例拆分請留空 ""
RATIOS="1:1:1"           # 依【比例】拆分：例如 "1:1:1" (均分3份) 或 "2:1:1" (若已填 COUNTS，請將 RATIOS 設為 "")

# 抽樣與檔案處理策略
SPLIT_ORDER="stratified" # 支援: "stratified"(分層抽樣確保異常/正常比例一致)、"random"(隨機)、"rank"(依分數排名分段)、"filename"(依檔名)
SEED=42                  # 亂數種子
EXCLUDE_GOLDEN=""        # 若想在拆分前先排除 Golden 訓練圖，設為 "--exclude_golden"
USE_SYMLINK=""           # 若想用軟連結取代實體複製圖片與 .npy 以節省空間，設為 "--symlink"
# ==============================================================================

if [ ! -d "${SRC_DIR}" ]; then
    echo "❌ [設定錯誤] 找不到推論結果來源目錄 SRC_DIR: ${SRC_DIR}"
    echo "   請確認是否已先執行 run_2b_infer_fab_no_gt.sh 完成推論！"
    exit 1
fi

# 自動組合選填參數陣列
EXTRA_ARGS=()
if [ -n "${OUT_DIR}" ]; then
    EXTRA_ARGS+=(--out_dir "${OUT_DIR}")
fi

if [ -n "${COUNTS}" ]; then
    EXTRA_ARGS+=(--counts "${COUNTS}")
elif [ -n "${RATIOS}" ]; then
    EXTRA_ARGS+=(--ratios "${RATIOS}")
fi

echo "============================================================"
echo " [Step 3: 推論結果拆分器啟動] 來源目錄: ${SRC_DIR}"
echo " [拆分設定] NUM_PACKS=${NUM_PACKS} | ORDER=${SPLIT_ORDER} | COUNTS=${COUNTS:-無} | RATIOS=${RATIOS:-預設}"
echo "============================================================"

python 4_split_eval_results.py \
    --src_dir "${SRC_DIR}" \
    --num_packs "${NUM_PACKS}" \
    --split_order "${SPLIT_ORDER}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    ${EXCLUDE_GOLDEN} ${USE_SYMLINK}