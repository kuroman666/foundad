#!/usr/bin/env bash
# 檔案位置: FoundAD/run_2a_train_fab_no_gt.sh
# 用途: 單獨執行 2a_train_fab_no_gt.py (負責載入/準備 Golden 資料集與執行 FoundAD 模型訓練)
# 用法: bash run_2a_train_fab_no_gt.sh [DATASET_ID] [LAYER_ID] [WAFER_POOL_DIR]
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
WAFER_POOL_DIR="${3:-/workspace/dataset/DA_Wafer_use/${DATASET_ID}/${LAYER_ID}/other_status}"

DATASETS_DIR="./datasets"
FAB_RESULT_DIR="./wafer_escape_results/${DATASET_ID}/${LAYER_ID}"
COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# 訓練與切塊超參數
CATEGORY="wafer"
SHOTS=4
CKPT_STEP=2000
APP_TRAIN="train_dinov3"
TILE_SIZE=512
TILE_OVERLAP=64
MAX_TRAIN_TILES=48
TRAIN_BATCH_SIZE=8
BG_STD_THRESH=1.5

# 流程控制旗標
SKIP_TRAIN=""            # 若只想重新驗證並對接既有權重，可設為 "--skip_train"
REBUILD_GOLDEN=""        # 若想強制重跑 Golden 挑選與純化，可設為 "--rebuild_golden"
DISABLE_TILING=""        # 若關閉切塊，設為 "--disable_tiling"

DATA_NAME="wafer_${LAYER_ID}_${SHOTS}shot"
DIY_NAME="fab_${LAYER_ID}"
# ==============================================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="./hub_cache/facebookresearch_dinov3_main"

# Pre-Flight 防呆檢查
if [ ! -d "${WAFER_POOL_DIR}" ]; then
    echo "❌ [設定錯誤] 找不到晶圓圖片池目錄 WAFER_POOL_DIR: ${WAFER_POOL_DIR}"
    exit 1
fi

if [ "${COMPANY_DINO_CKPT}" != "auto" ] && [ ! -f "${COMPANY_DINO_CKPT}" ]; then
    echo "❌ [設定錯誤] 找不到本地 DINO 權重檔 COMPANY_DINO_CKPT: ${COMPANY_DINO_CKPT}"
    exit 1
fi

if [ ! -f "foundad/configs/app/${APP_TRAIN}.yaml" ]; then
    if [ -f "foundad/configs/app/train.yaml" ]; then
        echo "⚠️ [自動切換] 找不到 foundad/configs/app/${APP_TRAIN}.yaml，自動改用 train.yaml"
        APP_TRAIN="train"
    else
        echo "❌ [設定錯誤] 在 foundad/configs/app/ 下找不到 ${APP_TRAIN}.yaml！"
        exit 1
    fi
fi

echo "============================================================"
echo " [Step 2a: FoundAD 無GT模型訓練啟動] DATASET=${DATASET_ID} | LAYER=${LAYER_ID}"
echo " [訓練設定] DATA_NAME=${DATA_NAME} | DIY_NAME=${DIY_NAME} | STEP=${CKPT_STEP}"
echo "============================================================"

python 2_train_fab_no_gt.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --out_dir "${FAB_RESULT_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CATEGORY}" \
    --app_train "${APP_TRAIN}" \
    --diy_name "${DIY_NAME}" \
    --ckpt_step "${CKPT_STEP}" \
    --shots "${SHOTS}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_TRAIN} ${REBUILD_GOLDEN} ${DISABLE_TILING}