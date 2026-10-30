#!/usr/bin/env bash
# 檔案位置: FoundAD/run_inside.sh (全選覆蓋並存檔)
set -e

# ===================== [進公司只需改這區參數] =====================
# 1. 工程師初篩為正常品、無 Ground Truth 的 5000x5000 晶圓圖片資料夾路徑
WAFER_POOL_DIR="/path/to/company/engineer_passed_wafers"

# 2. 公司內部 H100 上的本地 DINOv3 或 DINOv2 .pth 權重路徑
COMPANY_DINO_CKPT="/path/to/company/dinov3_or_dinov2.pth"

# 3. 輸出結果資料夾 (裡面會自動產生 5000x5000 紅框標註圖與 escape_ranking_report.txt)
FAB_RESULT_DIR="./wafer_escape_results"

# 4. 模型與畫框超參數
SHOTS=4                  # 自動挑選 4 張最純淨的 Golden Wafer
CKPT_STEP=2000           # FoundAD 訓練步數
SIGMA_K=3.2              # 紅色瑕疵框敏感度 (若框太多調大至 3.8；若微弱瑕疵沒框到調小至 2.6)
APP_TRAIN="train"        # 對應 foundad/configs/app/ 下的訓練設定檔
APP_TEST="test"          # 對應 foundad/configs/app/ 下的測試設定檔
SKIP_TRAIN=""            # 第一次跑設為 ""；若已訓練完只想重調 SIGMA_K 畫框，可設為 "--skip_train"

# 5. 5000x5000 高解析度防失真切塊 (Tiling) 與無縫縫合參數
TILE_SIZE=512            # 切塊大小 (512 對應模型原生輸入解析度，達成 1:1 零壓縮失真；若想加速可設 1024)
TILE_OVERLAP=64          # 相鄰圖塊重疊像素 (用於 2D 餘弦平滑拼接，消除邊界接縫)
MAX_TRAIN_TILES=16       # 從 4 張 Golden Wafer 分層抽出的高解析度代表性訓練圖塊數
TRAIN_BATCH_SIZE=8       # 訓練 Batch Size 上限 (防止高解析度 OOM)
BG_STD_THRESH=1.5        # 自動跳過晶圓圓周外純黑/純白背景角落塊的標準差門檻
# =================================================================

# 強制設定離線環境變數，防止任何套件在公司內網嘗試連外網而卡死
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 讓 foundad.py 自動讀取公司內的本地 DINO .pth 權重與本地 repo 資料夾
export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="./hub_cache/facebookresearch_dinov3_main"  # 若用 dinov2 可留空

echo "============================================================"
echo " [Fab Escape 5000x5000 高解析度抓漏啟動] 晶圓池: ${WAFER_POOL_DIR}"
echo " [切塊設定] TILE_SIZE=${TILE_SIZE} | OVERLAP=${TILE_OVERLAP} | MAX_TRAIN_TILES=${MAX_TRAIN_TILES}"
echo "============================================================"

python 2_run_fab_no_gt.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --out_dir "${FAB_RESULT_DIR}" \
    --data_name "mvtec_wafer_${SHOTS}shot" \
    --category "wafer" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "fab_h100" \
    --ckpt_step "${CKPT_STEP}" \
    --shots "${SHOTS}" \
    --sigma_k "${SIGMA_K}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_TRAIN}

echo "============================================================"
echo " [全部完成！5000x5000 無失真紅框圖與排名報表已整合至 ${FAB_RESULT_DIR}/]"
echo "  ├── ${FAB_RESULT_DIR}/escape_ranking_report.txt (異常排名表)"
echo "  ├── ${FAB_RESULT_DIR}/bbox_images/              (全解析度紅框標註圖)"
echo "  └── ${FAB_RESULT_DIR}/heatmaps/                 (全解析度無縫拼接熱力圖)"
echo "============================================================"