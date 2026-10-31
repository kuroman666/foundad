#!/usr/bin/env bash
# 檔案位置: FoundAD/run_inside.sh (全選覆蓋並存檔)
set -e

# ===================== [進公司只需改這區參數] =====================
# 1. 工程師初篩為正常品、無 Ground Truth 的 5000x5000 圖片資料夾路徑 (保留原始檔名不更動)
WAFER_POOL_DIR="/path/to/company/engineer_passed_wafers"

# 2. 公司內部 H100 上的本地 DINOv3 或 DINOv2 .pth 權重路徑
COMPANY_DINO_CKPT="/path/to/company/dinov3_or_dinov2.pth"

# 3. 輸出結果根資料夾 (每次執行會自動在底下建立 <YYYYMMDD_HHMMSS> 時間資料夾，絕不覆蓋)
FAB_RESULT_DIR="./wafer_escape_results"

# 4. 模型與異常判定門檻超參數
SHOTS=4                  # 自動挑選 4 張最純淨的 Golden Image
CKPT_STEP=2000           # FoundAD 訓練步數
THRESHOLD=0.25           # 異常分數門檻：只有最高分 >= THRESHOLD 的圖片才會畫紅框並放入 anomaly_images/ 資料夾
CORE_RATIO=0.55          # 紅框核心收斂比例 (0.45~0.65，越高紅框越緊貼最高分缺陷中心)
APP_TRAIN="train"        # 對應 foundad/configs/app/ 下的訓練設定檔
APP_TEST="test"          # 對應 foundad/configs/app/ 下的測試設定檔
SKIP_TRAIN=""            # 第一次跑設為 ""；若已訓練完只想重調 THRESHOLD 畫框，可設為 "--skip_train"

# 5. 5000x5000 高解析度防失真切塊 (Tiling) 與無縫縫合參數
TILE_SIZE=512            # 切塊大小 (512 對應模型原生輸入解析度，達成 1:1 零壓縮失真；若想加速可設 1024)
TILE_OVERLAP=64          # 相鄰圖塊重疊像素 (用於 2D 餘弦平滑拼接，消除邊界接縫)
MAX_TRAIN_TILES=16       # 從 4 張 Golden 大圖分層抽出的高解析度代表性訓練圖塊數
TRAIN_BATCH_SIZE=8       # 訓練 Batch Size 上限 (防止高解析度 OOM)
BG_STD_THRESH=1.5        # 自動跳過純黑/純白背景角落塊的標準差門檻
# =================================================================

# 強制設定離線環境變數，防止任何套件在公司內網嘗試連外網而卡死
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 讓 foundad.py 自動讀取公司內的本地 DINO .pth 權重與本地 repo 資料夾
export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="./hub_cache/facebookresearch_dinov3_main"  # 若用 dinov2 可留空

echo "============================================================"
echo " [Fab Escape 5000x5000 高解析度抓漏啟動] 圖片池: ${WAFER_POOL_DIR}"
echo " [門檻設定] THRESHOLD=${THRESHOLD} | CORE_RATIO=${CORE_RATIO} | TILE_SIZE=${TILE_SIZE}"
echo "============================================================"

python 2_run_fab_no_gt.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --out_dir "${FAB_RESULT_DIR}" \
    --data_name "mvtec_fab_${SHOTS}shot" \
    --category "auto" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "fab_h100" \
    --ckpt_step "${CKPT_STEP}" \
    --shots "${SHOTS}" \
    --threshold "${THRESHOLD}" \
    --core_ratio "${CORE_RATIO}" \
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