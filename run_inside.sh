#!/usr/bin/env bash
# 檔案位置: FoundAD/run_inside.sh
set -e

# ===================== [進公司只需改這區參數] =====================
# 1. 工程師初篩為正常品、無 Ground Truth 的晶圓圖片資料夾路徑
WAFER_POOL_DIR="/path/to/company/engineer_passed_wafers"

# 2. 公司內部 H100 上的本地 DINOv3 或 DINOv2 .pth 權重路徑
COMPANY_DINO_CKPT="/path/to/company/dinov3_or_dinov2.pth"

# 3. 輸出結果資料夾 (裡面會自動產生紅框標註圖與 escape_ranking_report.txt)
FAB_RESULT_DIR="./wafer_escape_results"

# 4. 模型與畫框超參數
SHOTS=4                  # 自動挑選 4 張最純淨的 Golden Image
CKPT_STEP=2000           # FoundAD 訓練步數 (需與 configs/app/train.yaml 的 epochs 對應)
SIGMA_K=3.2              # 紅色瑕疵框敏感度 (若框太多調大至 3.8；若微弱瑕疵沒框到調小至 2.6)
APP_TRAIN="train"        # 對應 foundad/configs/app/ 下的訓練設定檔
APP_TEST="test"          # 對應 foundad/configs/app/ 下的測試設定檔
SKIP_TRAIN=""            # 第一次跑設為 ""；若已訓練完只想重調 SIGMA_K 畫框，可設為 "--skip_train"
# =================================================================

# 強制設定離線環境變數，防止任何套件在公司內網嘗試連外網而卡死
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 讓 foundad.py 自動讀取公司內的本地 DINO .pth 權重與本地 repo 資料夾
export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="./hub_cache/facebookresearch_dinov3_main"  # 若用 dinov2 可留空

echo "============================================================"
echo " [Fab Escape 抓漏任務啟動] 晶圓圖片池: ${WAFER_POOL_DIR}"
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
    ${SKIP_TRAIN}

echo "============================================================"
echo " [全部完成！所有文字與圖片結果已整合至 ${FAB_RESULT_DIR}/ 資料夾]"
echo "  ├── ${FAB_RESULT_DIR}/1_public_with_gt/ (有 GT 分數 .txt/.csv 與分割圖)"
echo "  └── ${FAB_RESULT_DIR}/2_fab_no_gt/      (Golden名單、排名表 .txt/.csv、紅框瑕疵圖)"
echo "============================================================"
