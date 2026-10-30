#!/usr/bin/env bash
# 檔案位置: FoundAD/run_outside.sh (全選覆蓋並存檔)
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto"

# --- Task 1: 有 GT 的公開資料集測試參數 ---
MVTEC_ROOT="${DATASETS_DIR}/mvtec"
GT_DATA_NAME="mvtec_4shot"
GT_CLASSES="[bottle,cable,capsule,grid,tile,transistor]"
GT_DIY_NAME="dbug"
GT_CKPT_STEP=2000
MAX_BATCH_SIZE=16
# SKIP_GT_SAMPLE="--skip_sample"        # 想重抽樣改為 ""
# SKIP_GT_TRAIN="--skip_train"          # 想重訓練 Task 1 改為 ""

# --- Task 2: 無 GT 模擬抓漏 + 高解析度切塊縫合畫紅框測試參數 ---
MOCK_SOURCE_CAT="${DATASETS_DIR}/mvtec/capsule" # 模擬抓漏的來源類別 (可改成其他類別)
SIM_POOL_DIR="${DATASETS_DIR}/sim_fab_capsule_pool"
SHOTS=4
FAB_CKPT_STEP=2000
SIGMA_K=3.2                           # 畫紅框門檻 (越小框越多，越大越嚴格)
APP_TRAIN="train_dinov3"
APP_TEST="test"
SKIP_FAB_TRAIN=""                     # 第一次跑設為 "" (會訓練 2000 epochs)；訓練完後可改 "--skip_train"

# 高解析度切塊與縫合參數 (MVTec 圖片約 900x900~1024x1024，設 512 可在公司外完整模擬切塊與縫合)
TILE_SIZE=512
TILE_OVERLAP=64
MAX_TRAIN_TILES=16
TRAIN_BATCH_SIZE=8
BG_STD_THRESH=1.5
# ===============================================================

echo "============================================================"
echo " [Task 1] 執行有 GT 公開資料集測試 -> ${OUTPUT_ROOT}/1_public_with_gt"
echo "============================================================"
python 1_run_public_with_gt.py \
    --dataset mvtec \
    --test_root "${MVTEC_ROOT}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${GT_DATA_NAME}" \
    --classnames "${GT_CLASSES}" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "${GT_DIY_NAME}" \
    --ckpt_step "${GT_CKPT_STEP}" \
    --shots "${SHOTS}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --save_dir "${OUTPUT_ROOT}/1_public_with_gt" \
    ${SKIP_GT_SAMPLE} ${SKIP_GT_TRAIN}

echo "============================================================"
echo " [Task 2] 執行無 GT 高解析度切塊縫合盲池測試 -> ${OUTPUT_ROOT}/2_fab_no_gt"
echo "============================================================"
python 2_run_fab_no_gt.py \
    --mock_from_mvtec "${MOCK_SOURCE_CAT}" \
    --pool_dir "${SIM_POOL_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --ckpt "${DINO_CKPT_PATH}" \
    --out_dir "${OUTPUT_ROOT}/2_fab_no_gt" \
    --data_name "mvtec_sim_${SHOTS}shot" \
    --category "wafer" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "sim_escape" \
    --ckpt_step "${FAB_CKPT_STEP}" \
    --shots "${SHOTS}" \
    --sigma_k "${SIGMA_K}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_FAB_TRAIN}