#!/usr/bin/env bash
# 檔案位置: FoundAD/run_2_train_fab_no_gt_old.sh
# 用途: 單獨執行 2_train_fab_no_gt_old.py (舊版無GT訓練腳本：含自動挑 Golden 圖、切塊與 FoundAD 訓練)
# 用法: bash run_2_train_fab_no_gt_old.sh [DATASET_ID] [LAYER_ID] [WAFER_POOL_DIR]
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

# 流程控制旗標 (注意：2_train_fab_no_gt_old.py 不支援 --rebuild_golden 與 --force_train)
SKIP_TRAIN=""            # 若只想沿用快取並跳過模型訓練步驟，可設為 "--skip_train"
DISABLE_TILING=""        # 若關閉切塊退回整張訓練模式，可設為 "--disable_tiling"

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

# 為了讓舊版 (無內建 .offline_hub_patch) 在內網環境也能攔截 torch.hub.load，若存在則主動掛入 PYTHONPATH
if [ -d "./.offline_hub_patch" ]; then
    export PYTHONPATH="$(pwd)/.offline_hub_patch:${PYTHONPATH}"
fi

echo "============================================================"
echo " [Step 2 (Old): FoundAD 無GT模型訓練啟動] DATASET=${DATASET_ID} | LAYER=${LAYER_ID}"
echo " [訓練設定] DATA_NAME=${DATA_NAME} | DIY_NAME=${DIY_NAME} | STEP=${CKPT_STEP}"
echo "============================================================"

python 2_train_fab_no_gt_old.py \
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
    ${SKIP_TRAIN} ${DISABLE_TILING}