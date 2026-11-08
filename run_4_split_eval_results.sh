#!/usr/bin/env bash
# 檔案位置: FoundAD/run_4_split_eval_results.sh (相容 run_3_split_eval_results.sh)
# 用途: 單獨執行 4_split_eval_results.py (支援指定拆分成幾包、每包張數、總數不符嚴格報錯，以及各包獨立重算動態門檻)
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
# 若傳入第 3 個參數則直接作為來源目錄，否則自動抓取該 Layer 的推論輸出根目錄 (程式會自動找底下最新時間戳記)
SRC_DIR="${3:-./wafer_escape_results/${DATASET_ID}/${LAYER_ID}}"

# [新增] 支援直接從命令列第 4、第 5 個參數傳入「切成幾包 (NUM_PACKS)」與「每包張數 (COUNTS)」，亦可直接在下方修改預設值
# 範例 1 (直接在下方填寫或命令列傳入): bash run_4_split_eval_results.sh W639 TMQB28 "" 3 "700,700,600"
# 範例 2 (每包固定 500 張，共 4 包)  : bash run_4_split_eval_results.sh W639 TMQB28 "" 4 "500"
NUM_PACKS="${4:-3}"      # [功能 1] 選擇將資料集切成幾包 Eval 結果 (預設 3 包)
COUNTS="${5:-}"          # [功能 2] 選擇每包 Eval 的圖片數量：例如填 "500" (每包500張) 或 "700,700,600"
                         # ⚠️ 注意：若有填寫 COUNTS，各包數量加總必須「完全等於」資料集內的可拆分圖片總數，否則程式會立即報錯並顯示雙方數量！
RATIOS="1:1:1"           # 若 COUNTS 留空 ""，才會改用比例拆分 (當 NUM_PACKS 改變且 RATIOS 為預設 1:1:1 時，會自動依 NUM_PACKS 均分)

# 拆分後輸出目錄 (留空 "" 則預設輸出在該次時間戳記目錄下的 split_eval_${NUM_PACKS}packs/)
OUT_DIR=""

# [新增] 各包獨立重算動態門檻開關 (徹底消除老師對「先推論再拆分是否偷看全池門檻」的質疑)
RECOMPUTE_PACK_THRESH="--recompute_pack_thresh"  # 設為 "--recompute_pack_thresh" 讓每包獨立重算動態門檻；設為 "" 則沿用全池統一門檻
SIGMA_K=3.5              # 各包獨立重算門檻時的標準差倍數 k (與 Step 3 一致)
AUTO_PERCENTILE=95.0     # 各包獨立重算門檻時的分位數 (如 95.0 代表 P95)
MIN_THRESHOLD=0.0        # 動態門檻最低安全底線

# 抽樣與檔案處理策略
SPLIT_ORDER="stratified" # 支援: "stratified"(分層抽樣確保異常/正常比例一致)、"random"(隨機)、"rank"(依分數排名分段)、"filename"(依檔名)
SEED=42                  # 亂數種子
EXCLUDE_GOLDEN=""        # 若想在拆分前先排除 Golden 訓練圖 (只拆非 Golden 測試圖)，設為 "--exclude_golden"
USE_SYMLINK=""           # 若想用軟連結取代實體複製圖片與 .npy 以節省空間，設為 "--symlink"
# ==============================================================================

if [ ! -d "${SRC_DIR}" ]; then
    echo "❌ [設定錯誤] 找不到推論結果來源目錄 SRC_DIR: ${SRC_DIR}"
    echo "   請確認是否已先執行 run_3_infer_fab_no_gt.sh 完成推論！"
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
    # 若使用者修改了 NUM_PACKS (例如改成 4 包)，而 RATIOS 仍停留在預設的 "1:1:1"，自動擴展為 NUM_PACKS 等份均分
    if [ "${RATIOS}" = "1:1:1" ] && [ "${NUM_PACKS}" -ne 3 ]; then
        AUTO_RATIOS=$(printf "1:%.0s" $(seq 1 "${NUM_PACKS}") | sed 's/:$//')
        EXTRA_ARGS+=(--ratios "${AUTO_RATIOS}")
    else
        EXTRA_ARGS+=(--ratios "${RATIOS}")
    fi
fi

echo "============================================================"
echo " [Step 4: 推論結果拆分器啟動] 來源目錄: ${SRC_DIR}"
echo " [拆分設定] NUM_PACKS=${NUM_PACKS} | ORDER=${SPLIT_ORDER} | COUNTS=${COUNTS:-無(依比例均分)} | RATIOS=${RATIOS:-預設}"
echo " [門檻策略] RECOMPUTE_PACK_THRESH=${RECOMPUTE_PACK_THRESH:-OFF(沿用全池門檻)} (SIGMA_K=${SIGMA_K})"
echo "============================================================"

python 4_split_eval_results.py \
    --src_dir "${SRC_DIR}" \
    --num_packs "${NUM_PACKS}" \
    --split_order "${SPLIT_ORDER}" \
    --seed "${SEED}" \
    --sigma_k "${SIGMA_K}" \
    --auto_percentile "${AUTO_PERCENTILE}" \
    --min_threshold "${MIN_THRESHOLD}" \
    "${EXTRA_ARGS[@]}" \
    ${RECOMPUTE_PACK_THRESH} ${EXCLUDE_GOLDEN} ${USE_SYMLINK}