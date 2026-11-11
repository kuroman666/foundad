#!/usr/bin/env bash
# 檔案位置: FoundAD/run_1_auto_golden_old.sh
# 用途: 單獨呼叫 1_auto_golden_old.py 執行 Medoid Centrality 挑選 Golden 原圖
# 用法: bash run_1_auto_golden_old.sh [DATASET_ID] [LAYER_ID] [WAFER_POOL_DIR]
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
WAFER_POOL_DIR="${3:-/workspace/dataset/DA_Wafer_use/${DATASET_ID}/${LAYER_ID}/other_status}"

# 輸出與資料集根目錄
DATASETS_DIR="./datasets"
FAB_RESULT_DIR="./wafer_escape_results/${DATASET_ID}/${LAYER_ID}"
REPORT_DIR="${FAB_RESULT_DIR}/golden_reports"

# 公司內部的本地 DINOv3 或 DINOv2 .pth 權重路徑 (外部測試可改為 "auto" 自動搜尋快取)
COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# Golden 挑選參數 (對應 1_auto_golden_old.py 之 select_golden_images 介面)
CATEGORY="wafer"
SHOTS=4                  # 挑選的 Golden 原圖張數
DATA_NAME="wafer_${LAYER_ID}_${SHOTS}shot"
OUT_DIR="${DATASETS_DIR}/${DATA_NAME}"

# 流程控制旗標 (因為 1_auto_golden_old.py 本身每次都會刪除舊資料夾重跑，此處由 Shell 層提供防呆)
REBUILD_GOLDEN="0"       # 0: 若已存在挑好的 Golden 則自動略過；1: 強制重跑覆蓋
# ==============================================================================

# 強制設定離線環境變數
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

echo "============================================================"
echo " [Auto-Golden (fab_tools) 啟動] DATASET=${DATASET_ID} | LAYER=${LAYER_ID}"
echo " [圖片池路徑] ${WAFER_POOL_DIR}"
echo " [輸出訓練集] ${OUT_DIR}/train/${CATEGORY}"
echo " [報表輸出區] ${REPORT_DIR}"
echo "============================================================"

# Shell 層防呆機制：檢查是否已經有挑好的 Golden 圖片
TARGET_TRAIN_DIR="${OUT_DIR}/train/${CATEGORY}"
if [ "${REBUILD_GOLDEN}" != "1" ] && [ -d "${TARGET_TRAIN_DIR}" ]; then
    EXISTING_COUNT=$(find "${TARGET_TRAIN_DIR}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
    if [ "${EXISTING_COUNT}" -ge "${SHOTS}" ]; then
        echo "⚡ [防呆略過] 偵測到 ${TARGET_TRAIN_DIR} 已存在 ${EXISTING_COUNT} 張 Golden 圖片！"
        echo "   若需強制重挑，請將腳本內 REBUILD_GOLDEN 設為 \"1\"。"
        exit 0
    fi
fi

# 由於 1_auto_golden.py 為純函式模組 (無 main CLI)，透過 python -c 傳參呼叫
CKPT_ARG="${COMPANY_DINO_CKPT}"
if [ "${CKPT_ARG}" = "auto" ]; then
    CKPT_ARG=""
fi

python -c "
import importlib
mod = importlib.import_module('1_auto_golden_old')

mod.select_golden_images(
    pool_dir='${WAFER_POOL_DIR}',
    ckpt_path='${CKPT_ARG}',
    out_dir='${OUT_DIR}',
    category='${CATEGORY}',
    shots=${SHOTS},
    report_dir='${REPORT_DIR}'
)
"