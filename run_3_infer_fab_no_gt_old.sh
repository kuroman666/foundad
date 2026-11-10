#!/usr/bin/env bash
# ==============================================================================
# 檔案位置: FoundAD/run_3_infer_fab_no_gt_old.sh
# 用途: 執行 3_infer_fab_no_gt_old.py (FoundAD 無 GT 盲池高解析度純推論與防誤報篩選流程)
# 使用方式:
#   1. 直接使用預設參數執行:
#      bash run_3_infer_fab_no_gt_old.sh
#   2. 透過環境變數覆蓋參數，或於後方直接附加 CLI 旗標 (如 --force_infer / --raw_score_mode):
#      GPU_ID=0 THRESH_MODE=mean_std SIGMA_K=3.0 bash run_3_infer_fab_no_gt_old.sh --force_infer
# ==============================================================================

set -euo pipefail

# 自動切換至腳本所在的專案根目錄 (FoundAD/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ------------------------------------------------------------------------------
# 0. 硬體與執行環境設定
# ------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

# ------------------------------------------------------------------------------
# 1. 資料路徑與模型權重設定 (對齊 Step 1 / Step 2 的輸出設定)
# ------------------------------------------------------------------------------
POOL_DIR="${POOL_DIR:-./datasets/mock_unlabeled_pool}"       # 無 GT 測試盲池圖片資料夾
MOCK_FROM_MVTEC="${MOCK_FROM_MVTEC:-}"                       # 若需自動從 MVTec 建立模擬盲池可填路徑 (如 ./datasets/mvtec/transistor)，留空則直接讀 POOL_DIR
OUT_DIR="${OUT_DIR:-./output/3_infer_fab_no_gt_old}"         # 推論結果輸出主目錄 (程式內會自動建立時間戳記子目錄)
DATASETS_DIR="${DATASETS_DIR:-./datasets}"                   # 資料集與切塊快取根目錄
DATA_NAME="${DATA_NAME:-mvtec_sim_4shot}"                    # 對應 logs/<DATA_NAME> 與 datasets/<DATA_NAME> 的實驗名稱
CATEGORY="${CATEGORY:-auto}"                                 # 產品/晶圓類別 (預設 auto 自動依路徑推斷)
APP_TEST="${APP_TEST:-test}"                                 # FoundAD 推論設定檔模式
DIY_NAME="${DIY_NAME:-sim_escape}"                           # 訓練實驗自訂後綴名稱
CKPT_STEP="${CKPT_STEP:}"                               # 欲載入的訓練權重步數 (若不存在會自動抓取最大 step)

# ------------------------------------------------------------------------------
# 2. 高解析度切塊 (Tiling) 與批次推論參數
# ------------------------------------------------------------------------------
TILE_SIZE="${TILE_SIZE:-512}"                                # 子圖塊像素大小 (建議 512 或 1024)
TILE_OVERLAP="${TILE_OVERLAP:-64}"                           # 相鄰子圖塊重疊寬度 (Raised-Cosine 無縫平滑拼接)
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"                   # 子圖塊推論 Mini-Batch 大小
BG_STD_THRESH="${BG_STD_THRESH:-1.5}"                        # 過濾純背景角落塊的標準差門檻
DISABLE_TILING="${DISABLE_TILING:-0}"                        # 設為 1 則附加 --disable_tiling (關閉切塊)

# ------------------------------------------------------------------------------
# 3. 動態門檻對齊參數 (Thresholding)
#    可選模式: hybrid | mean_std | percentile | top_k | fixed
# ------------------------------------------------------------------------------
THRESH_MODE="${THRESH_MODE:-hybrid}"                         # 異常門檻決定模式
THRESHOLD="${THRESHOLD:-0.25}"                               # fixed 模式固定門檻 / hybrid 模式保底下限之一
MIN_THRESHOLD="${MIN_THRESHOLD:-0.27}"                       # 動態門檻絕對最低安全底線 (適用於 hybrid / percentile / top_k)
AUTO_PERCENTILE="${AUTO_PERCENTILE:-95.0}"                   # 盲池分位數 (如 95.0 代表 P95)
TOP_K="${TOP_K:-0}"                                          # 若 > 0 則限制最多只輸出前 K 高分可疑圖片
SIGMA_K="${SIGMA_K:-3.5}"                                    # mean_std 或 hybrid 模式之標準差倍數 k

# ------------------------------------------------------------------------------
# 4. 四合一防誤報濾波與紅框收斂參數
# ------------------------------------------------------------------------------
RAW_SCORE_MODE="${RAW_SCORE_MODE:-0}"                        # 設為 1 則附加 --raw_score_mode (關閉四合一濾波，供消融實驗對照)
CORE_RATIO="${CORE_RATIO:-0.55}"                             # 異常熱區峰值核心收斂比例 (0.45~0.65)
BORDER_RATIO="${BORDER_RATIO:-0.02}"                         # 矩形外框 Padding 偽影抑制比例 (預設 2%)
WAFER_EDGE_MODE="${WAFER_EDGE_MODE:-auto}"                   # 晶圓圓周邊緣抑制: auto | on | off
WAFER_EDGE_MARGIN="${WAFER_EDGE_MARGIN:-0.04}"               # 晶圓外圈倒角抑制寬度佔半徑比例 (預設 4%)
SPECKLE_FILTER_RATIO="${SPECKLE_FILTER_RATIO:-0.008}"        # 灰階形態學開運算濾除微塵尖峰之核大小比例 (設 0 關閉)
MIN_AREA_RATIO="${MIN_AREA_RATIO:-0.00012}"                  # 有效紅框最小面積佔全圖比例

# ------------------------------------------------------------------------------
# 5. 快取控制開關
# ------------------------------------------------------------------------------
FORCE_INFER="${FORCE_INFER:-0}"                              # 設為 1 則附加 --force_infer (強制重切圖塊並重跑 GPU 推論)
SKIP_TRAIN="${SKIP_TRAIN:-1}"                                # 相容保留旗標 (設為 1 則附加 --skip_train)

# ------------------------------------------------------------------------------
# 組裝執行參數陣列
# ------------------------------------------------------------------------------
CMD_ARGS=(
    "3_infer_fab_no_gt_old.py"
    "--pool_dir" "${POOL_DIR}"
    "--out_dir" "${OUT_DIR}"
    "--datasets_dir" "${DATASETS_DIR}"
    "--data_name" "${DATA_NAME}"
    "--category" "${CATEGORY}"
    "--app_test" "${APP_TEST}"
    "--diy_name" "${DIY_NAME}"
    "--ckpt_step" "${CKPT_STEP}"
    "--thresh_mode" "${THRESH_MODE}"
    "--threshold" "${THRESHOLD}"
    "--min_threshold" "${MIN_THRESHOLD}"
    "--auto_percentile" "${AUTO_PERCENTILE}"
    "--top_k" "${TOP_K}"
    "--sigma_k" "${SIGMA_K}"
    "--core_ratio" "${CORE_RATIO}"
    "--border_ratio" "${BORDER_RATIO}"
    "--wafer_edge_mode" "${WAFER_EDGE_MODE}"
    "--wafer_edge_margin" "${WAFER_EDGE_MARGIN}"
    "--speckle_filter_ratio" "${SPECKLE_FILTER_RATIO}"
    "--min_area_ratio" "${MIN_AREA_RATIO}"
    "--tile_size" "${TILE_SIZE}"
    "--tile_overlap" "${TILE_OVERLAP}"
    "--infer_batch_size" "${INFER_BATCH_SIZE}"
    "--bg_std_thresh" "${BG_STD_THRESH}"
)

if [[ -n "${MOCK_FROM_MVTEC}" ]]; then
    CMD_ARGS+=("--mock_from_mvtec" "${MOCK_FROM_MVTEC}")
fi

if [[ "${RAW_SCORE_MODE}" == "1" ]]; then
    CMD_ARGS+=("--raw_score_mode")
fi

if [[ "${DISABLE_TILING}" == "1" ]]; then
    CMD_ARGS+=("--disable_tiling")
fi

if [[ "${FORCE_INFER}" == "1" ]]; then
    CMD_ARGS+=("--force_infer")
fi

if [[ "${SKIP_TRAIN}" == "1" ]]; then
    CMD_ARGS+=("--skip_train")
fi

# 印出啟動資訊摘要
echo "==========================================================================="
echo "🚀 啟動 FoundAD 無 GT 高解析度純推論 (3_infer_fab_no_gt_old.py)"
echo "---------------------------------------------------------------------------"
echo "  🖥️  GPU 裝置 (CUDA)    : ${CUDA_VISIBLE_DEVICES}"
echo "  📂 測試盲池 (pool_dir): ${POOL_DIR}"
echo "  📦 實驗名稱 (data_name): ${DATA_NAME} (diy_name=${DIY_NAME}, step=${CKPT_STEP})"
echo "  🧩 切塊設定 (Tiling)  : Tile=${TILE_SIZE}, Overlap=${TILE_OVERLAP}, Batch=${INFER_BATCH_SIZE}"
echo "  🎯 門檻模式 (Thresh)  : Mode=${THRESH_MODE}, SigmaK=${SIGMA_K}, MinThresh=${MIN_THRESHOLD}"
echo "  🛡️  濾波設定 (Filter)  : RawMode=${RAW_SCORE_MODE}, WaferEdge=${WAFER_EDGE_MODE} (${WAFER_EDGE_MARGIN})"
echo "  💾 輸出目錄 (out_dir) : ${OUT_DIR}"
echo "==========================================================================="

# 執行 Python 主程式，並支援將命令列額外傳入的參數 ("$@") 附加在最後方覆蓋設定
exec "${PYTHON_BIN}" "${CMD_ARGS[@]}" "$@"