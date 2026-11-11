#!/usr/bin/env bash
# 檔案位置: FoundAD/run_4_split_eval_results.sh (相容 run_3_split_eval_results.sh)
# 用途: 單獨執行 4_split_eval_results.py (支援指定拆分成幾包、每包張數、總數不符嚴格報錯，以及各包獨立重算動態門檻)
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"

# [功能 1] 指定已推論好的「檔案絕對路徑(含檔名)」或「結果目錄絕對路徑」
# 可從命令列第 3 個參數傳入，或直接修改下方預設路徑：
# 範例: "/home/user/FoundAD/wafer_escape_results/W639/TMQB28/20261110_120000/escape_ranking_report.csv"
SRC_INPUT="${3:-./wafer_escape_results/${DATASET_ID}/${LAYER_ID}}"

# [功能 2] 指定切分成幾包 (NUM_PACKS) 與 每包內有幾張圖片 (COUNTS)
# 範例 1 (切成 3 包，張數 700,700,600): bash run_4_split_eval_results.sh "" "" "/abs/path/escape_ranking_report.csv" 3 "700,700,600"
# 範例 2 (切成 4 包，每包固定 500 張) : bash run_4_split_eval_results.sh "" "" "/abs/path/escape_ranking_report.csv" 4 "500"
NUM_PACKS="${4:-3}"          # 選擇將資料集切成幾包 Eval 結果 (預設 3 包)
COUNTS="${5:-}"              # 選擇每包 Eval 的圖片數量：例如填 "500" (每包500張) 或 "700,700,600" (留空 "" 則依比例均分)
                             # ⚠️ 注意：若有填寫 COUNTS，各包數量加總必須「完全等於」可拆分圖片總數！
RATIOS="1:1:1"               # 若 COUNTS 留空 ""，才會改用比例拆分

# [功能 3] 固定亂數種子確保結果 100% 可復現 (預設指定為 random_seed=42，亦可由第 6 個參數傳入)
# 支援從命令列第 6 個參數指定亂數種子、第 7 個參數指定抽樣模式，或直接修改下方預設值：
# 範例 (使用 random 隨機抽樣並指定 seed=123):
#   bash run_4_split_eval_results.sh "" "" "/abs/path/escape_ranking_report.csv" 3 "700,700,600" 123 "random"
RANDOM_SEED="${6:-42}"
SEED="${RANDOM_SEED}"
SPLIT_ORDER="${7:-stratified}"       # 支援: "random"(隨機抽樣，受 RANDOM_SEED 控制)、"stratified"(分層抽樣確保異常/正常比例一致)、"rank"(依分數排名分段)、"filename"(依檔名)

# 拆分後輸出目錄 (留空 "" 則預設輸出在該次時間戳記目錄下的 split_eval_${NUM_PACKS}packs/)
OUT_DIR=""

# 各包獨立重算動態門檻開關
RECOMPUTE_PACK_THRESH="" # 設為 "--recompute_pack_thresh" 讓每包獨立重算動態門檻；設為 "" 則沿用全池統一門檻，確保切分前後瑕疵判定與總數 100% 一致
SIGMA_K=3.5                  # 各包獨立重算門檻時的標準差倍數 k (與 Step 3 一致)
AUTO_PERCENTILE=95.0         # 各包獨立重算門檻時的分位數 (如 95.0 代表 P95)
MIN_THRESHOLD=0.0            # 動態門檻最低安全底線

# 檔案處理策略
EXCLUDE_GOLDEN=""            # 若想在拆分前先排除 Golden 訓練圖 (只拆非 Golden 測試圖)，設為 "--exclude_golden"
USE_SYMLINK=""               # 若想用軟連結取代實體複製圖片與 .npy 以節省空間，設為 "--symlink"
# ==============================================================================

# 自動判斷 SRC_INPUT 是「特定檔案路徑 (含檔名)」還是「資料夾路徑」
TEMP_LINK_CSV=""
TEMP_LINK_TXT=""
cleanup_temp_links() {
    [ -n "${TEMP_LINK_CSV}" ] && [ -L "${TEMP_LINK_CSV}" ] && rm -f "${TEMP_LINK_CSV}"
    [ -n "${TEMP_LINK_TXT}" ] && [ -L "${TEMP_LINK_TXT}" ] && rm -f "${TEMP_LINK_TXT}"
}
trap cleanup_temp_links EXIT

if [ -f "${SRC_INPUT}" ]; then
    # 使用者直接傳入了包含檔案名稱的路徑 (例如 /abs/path/to/xxx.csv)
    SRC_FILE_ABS="$(cd "$(dirname "${SRC_INPUT}")" && pwd)/$(basename "${SRC_INPUT}")"
    SRC_DIR="$(dirname "${SRC_FILE_ABS}")"
    SRC_FILENAME="$(basename "${SRC_FILE_ABS}")"

    # 若指定的檔案名稱不是預設的 escape_ranking_report.csv，自動建立暫時軟連結讓 4_split_eval_results.py 讀取
    if [ "${SRC_FILENAME}" != "escape_ranking_report.csv" ]; then
        TEMP_LINK_CSV="${SRC_DIR}/escape_ranking_report.csv"
        ln -sf "${SRC_FILE_ABS}" "${TEMP_LINK_CSV}"

        # 若有同名的 .txt 標頭報表，也一併連結
        CANDIDATE_TXT="${SRC_FILE_ABS%.*}.txt"
        if [ -f "${CANDIDATE_TXT}" ] && [ ! -f "${SRC_DIR}/escape_ranking_report.txt" ]; then
            TEMP_LINK_TXT="${SRC_DIR}/escape_ranking_report.txt"
            ln -sf "${CANDIDATE_TXT}" "${TEMP_LINK_TXT}"
        fi
    fi
elif [ -d "${SRC_INPUT}" ]; then
    SRC_DIR="$(cd "${SRC_INPUT}" && pwd)"
    SRC_FILENAME="escape_ranking_report.csv (自動搜尋目錄)"
else
    echo "❌ [設定錯誤] 找不到指定的推論結果檔案或來源目錄: ${SRC_INPUT}"   
    echo "   請確認絕對路徑與檔案名稱是否正確，或是否已先執行 run_3_infer_fab_no_gt.sh 完成推論！"
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
echo " [Step 4: 推論結果拆分器啟動]"
echo " [來源路徑] 目錄: ${SRC_DIR} | 指定檔名: ${SRC_FILENAME}" 
echo " [拆分設定] NUM_PACKS=${NUM_PACKS} | ORDER=${SPLIT_ORDER} | COUNTS=${COUNTS:-無(依比例均分)} | RATIOS=${RATIOS:-預設}"
echo " [復現設定] random_seed=${RANDOM_SEED}"
echo " [門檻策略] RECOMPUTE_PACK_THRESH=${RECOMPUTE_PACK_THRESH:-OFF(沿用全池門檻)} (SIGMA_K=${SIGMA_K})"
echo "============================================================"
python 4_split_eval_results.py \
    --src_dir "${SRC_DIR}" \
    --num_packs "${NUM_PACKS}" \
    --split_order "${SPLIT_ORDER}" \
    --seed "${RANDOM_SEED}" \
    --sigma_k "${SIGMA_K}" \
    --auto_percentile "${AUTO_PERCENTILE}" \
    --min_threshold "${MIN_THRESHOLD}" \
    "${EXTRA_ARGS[@]}" \
    ${RECOMPUTE_PACK_THRESH} ${EXCLUDE_GOLDEN} ${USE_SYMLINK}
