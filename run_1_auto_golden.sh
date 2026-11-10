#!/usr/bin/env bash
# 檔案位置: FoundAD/run_1_auto_golden.sh
# 用途: 單獨執行 1_auto_golden.py (Stage 1 挑選 Golden 原圖 + Stage 2 圖塊級二次純化)
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
WAFER_POOL_DIR="${3:-/workspace/dataset/DA_Wafer_use/${DATASET_ID}/${LAYER_ID}/other_status}"

# 輸出與資料集根目錄
DATASETS_DIR="./datasets"
FAB_RESULT_DIR="./wafer_escape_results/${DATASET_ID}/${LAYER_ID}"

# 公司內部的本地 DINOv3 或 DINOv2 .pth 權重路徑 (外部測試可改為 "auto")
COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# Golden 挑選與高解析度切塊純化參數
CATEGORY="wafer"
SHOTS=4                  # Stage 1 挑選的 Golden 原圖張數
TILE_SIZE=512            # Stage 2 子圖塊像素大小 (512 對應模型 1:1 輸入)
TILE_OVERLAP=64          # 相鄰圖塊重疊像素寬度
MAX_TRAIN_TILES=48       # Stage 2 最終保留的代表性純淨訓練圖塊上限
BG_STD_THRESH=1.5        # 過濾純黑/純白背景角落塊的標準差門檻

# 流程控制旗標 (防呆機制：預設 REBUILD_GOLDEN="" 發現已有 Golden Image 時自動略過不重抓)
REBUILD_GOLDEN=""  # 若想強制重跑 Golden 挑選與純化，可設為 "--rebuild_golden"
DISABLE_TILING=""  # 若不切塊想直接用整張原圖，設為 "--disable_tiling"
MOCK_FROM_MVTEC="" # 若在公司外想從 MVTec 建立模擬盲池，可填 "./datasets/mvtec/capsule"

DATA_NAME="wafer_${LAYER_ID}_${SHOTS}shot"
# ==============================================================================

# 強制設定離線環境變數
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="./hub_cache/facebookresearch_dinov3_main"

# Pre-Flight 防呆檢查
if [ -z "${MOCK_FROM_MVTEC}" ] && [ ! -d "${WAFER_POOL_DIR}" ]; then
    echo "❌ [設定錯誤] 找不到晶圓圖片池目錄 WAFER_POOL_DIR: ${WAFER_POOL_DIR}"
    exit 1
fi

if [ "${COMPANY_DINO_CKPT}" != "auto" ] && [ ! -f "${COMPANY_DINO_CKPT}" ]; then
    echo "❌ [設定錯誤] 找不到本地 DINO 權重檔 COMPANY_DINO_CKPT: ${COMPANY_DINO_CKPT}"
    exit 1
fi

echo "============================================================"
echo " [Step 1: Auto-Golden 啟動] DATASET=${DATASET_ID} | LAYER=${LAYER_ID}"
echo " [圖片池路徑] ${WAFER_POOL_DIR}"
echo " [輸出訓練集] ${DATASETS_DIR}/${DATA_NAME}"
echo "============================================================"

MOCK_ARG=()
if [ -n "${MOCK_FROM_MVTEC}" ]; then
    MOCK_ARG=(--mock_from_mvtec "${MOCK_FROM_MVTEC}")
fi

python 1_auto_golden.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CATEGORY}" \
    --shots "${SHOTS}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    --report_dir "${FAB_RESULT_DIR}/golden_reports" \
    "${MOCK_ARG[@]}" \
    ${DISABLE_TILING}
