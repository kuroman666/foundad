#!/usr/bin/env bash
# 檔案位置: FoundAD/run_2b_infer_fab_no_gt.sh
# 用途: 單獨執行 2b_infer_fab_no_gt.py (測試池切塊推論 + RAM 無縫縫合 + 四合一防誤報畫紅框)
set -e

# ===================== [參數設定區：支援命令列傳入或預設值] =====================
DATASET_ID="${1:-W639}"
LAYER_ID="${2:-TMQB28}"
WAFER_POOL_DIR="${3:-/workspace/dataset/DA_Wafer_use/${DATASET_ID}/${LAYER_ID}/other_status}"

DATASETS_DIR="./datasets"
FAB_RESULT_DIR="./wafer_escape_results/${DATASET_ID}/${LAYER_ID}"
COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# 1. 基礎模型與推論參數
CATEGORY="wafer"
SHOTS=4
CKPT_STEP=2000
APP_TEST="test"
INFER_BATCH_SIZE=16

# 2. 動態異常門檻對齊超參數 (重調門檻只需改這區並直接重跑此腳本！)
THRESH_MODE="hybrid"     # 支援: "hybrid"、"mean_std"、"percentile"、"top_k"、"fixed"
SIGMA_K=3.5              # mean_std 模式下的標準差倍數 k；或 hybrid 模式下的 MAD 倍數
THRESHOLD=0.0            # fixed 模式的絕對門檻；或 hybrid 模式的基礎下限
MIN_THRESHOLD=0.0        # 動態門檻的最低安全底線
AUTO_PERCENTILE=95.0     # 自動對齊盲池分位數 (如 95.0 代表 P95)
TOP_K=0                  # 若設 > 0 (如 10)，限制最多只輸出前 K 張最可疑晶圓
CORE_RATIO=0.55          # 紅框核心收斂比例 (0.45~0.65)
RAW_SCORE_MODE=""        # 預設 "" (啟用四合一防誤報濾波)；若想對照未濾波原生分數設為 "--raw_score_mode"
MICRO_DEFECT_MODE="--micro_defect_mode"  # 設為 "--micro_defect_mode" 啟動微瑕疵高靈敏模式；設為 "" 則為一般模式

# 3. 晶圓圓周邊緣抑制與單點微塵濾除參數
WAFER_EDGE_MODE="on"     # 圓形晶圓建議設 "on" (或 "auto")
WAFER_EDGE_MARGIN=0.04   # 晶圓最外圈 4% 半徑平滑衰減
BORDER_RATIO=0.02        # 影像四週矩形邊框抑制比例 (2%)
SPECKLE_FILTER_RATIO=0.008 # 灰階形態學開運算微塵濾波比例
MIN_AREA_RATIO=0.00012   # 有效缺陷最小面積比例

# 4. 高解析度切塊參數
TILE_SIZE=512
TILE_OVERLAP=64
MAX_TRAIN_TILES=48
BG_STD_THRESH=1.5
FORCE_INFER=""           # 若想強制重新切塊測試池，設為 "--force_infer"
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

if [ ! -f "foundad/configs/app/${APP_TEST}.yaml" ]; then
    echo "❌ [設定錯誤] 在 foundad/configs/app/ 下找不到測試設定檔 ${APP_TEST}.yaml！"
    exit 1
fi

echo "============================================================"
echo " [Step 2b: FoundAD 無GT高解析度推論啟動] DATASET=${DATASET_ID} | LAYER=${LAYER_ID}"
echo " [防誤報設定] THRESH_MODE=${THRESH_MODE} (SIGMA_K=${SIGMA_K}, P${AUTO_PERCENTILE}, Min=${MIN_THRESHOLD}) | WAFER_EDGE=${WAFER_EDGE_MODE} (${WAFER_EDGE_MARGIN}) | RAW_MODE=${RAW_SCORE_MODE:-OFF}"
echo "============================================================"

python 3_infer_fab_no_gt.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --out_dir "${FAB_RESULT_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CATEGORY}" \
    --app_test "${APP_TEST}" \
    --diy_name "${DIY_NAME}" \
    --ckpt_step "${CKPT_STEP}" \
    --shots "${SHOTS}" \
    --thresh_mode "${THRESH_MODE}" \
    --threshold "${THRESHOLD}" \
    --min_threshold "${MIN_THRESHOLD}" \
    --auto_percentile "${AUTO_PERCENTILE}" \
    --top_k "${TOP_K}" \
    --sigma_k "${SIGMA_K}" \
    --core_ratio "${CORE_RATIO}" \
    --border_ratio "${BORDER_RATIO}" \
    --wafer_edge_mode "${WAFER_EDGE_MODE}" \
    --wafer_edge_margin "${WAFER_EDGE_MARGIN}" \
    --speckle_filter_ratio "${SPECKLE_FILTER_RATIO}" \
    --min_area_ratio "${MIN_AREA_RATIO}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --infer_batch_size "${INFER_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${FORCE_INFER} ${RAW_SCORE_MODE} ${MICRO_DEFECT_MODE} ${DISABLE_TILING}

# 終端機列印最新報表統計總結
python - "${FAB_RESULT_DIR}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base = sys.argv[1]
csv_files = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
if not csv_files:
    sys.exit(0)

latest_csv = csv_files[-1]
latest_txt = latest_csv.replace(".csv", ".txt")
eff_thresh = 0.25
overfit_line = ""
score_mode_str = "FILTERED_ROBUST"
if os.path.exists(latest_txt):
    with open(latest_txt, "r", encoding="utf-8") as tf:
        txt_head = tf.read(1500)
        m = re.search(r"Effective Threshold:\s*([0-9.]+)", txt_head)
        if m:
            eff_thresh = float(m.group(1))
        m_ov = re.search(r"Overfit Diagnostic\s*:\s*(.+)", txt_head)
        if m_ov:
            overfit_line = m_ov.group(1).strip()
        m_sm = re.search(r"Score Mode:\s*([A-Z_]+)", txt_head)
        if m_sm:
            score_mode_str = m_sm.group(1).strip()

records = []
with open(latest_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

total = len(records)
if total == 0:
    sys.exit(0)

non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
pass_non_golden = [r for r in non_golden if "PASS" in r["Status"]]
anom_all = [r for r in records if "ANOMALY" in r["Status"]]

target_recs = non_golden if non_golden else records
non_g_scores = [float(r["Max_Score"]) for r in target_recs]
max_s = float(np.max(non_g_scores))
p99_s = float(np.percentile(non_g_scores, 99))
p95_s = float(np.percentile(non_g_scores, 95))
median_s = float(np.median(non_g_scores))
mean_s = float(np.mean(non_g_scores))
std_s = float(np.std(non_g_scores))
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

print("\n" + "=" * 80)
print(f" 🎯 [公司內部無 GT 晶圓池 — 非 Golden 正常圖分數分佈與 Threshold 參考總結 ({score_mode_str})]")
print("-" * 80)
print(f"  📂 報表位置                   : {latest_csv}")
print(f"  🖼️ 測試池圖片總數              : {total} 張 (含 {total - len(non_golden)} 張 Golden 圖, 抓出 {len(anom_all)} 張異常)")
print(f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}")
if overfit_line:
    print(f"  🩺 模型過擬合自檢 (Overfit)    : {overfit_line}")
print("-" * 80)
print(f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}")
print(f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}")
if "Raw_Max_Score" in target_recs[0] and "Filtered_Max_Score" in target_recs[0]:
    raw_arr = [float(r["Raw_Max_Score"]) for r in target_recs]
    flt_arr = [float(r["Filtered_Max_Score"]) for r in target_recs]
    print(f"  🔬 濾波前後正常圖對照 (Ablation): [未濾波 Raw] Max={float(np.max(raw_arr)):.4f}, Mean={float(np.mean(raw_arr)):.4f}  vs  [四合一濾波] Max={float(np.max(flt_arr)):.4f}, Mean={float(np.mean(flt_arr)):.4f}")
if pass_non_golden and len(pass_non_golden) < len(non_golden):
    pg_scores = [float(r["Max_Score"]) for r in pass_non_golden]
    pg_max = float(np.max(pg_scores))
    pg_p99 = float(np.percentile(pg_scores, 99))
    pg_p95 = float(np.percentile(pg_scores, 95))
    pg_med = float(np.median(pg_scores))
    pg_mean = float(np.mean(pg_scores))
    pg_std = float(np.std(pg_scores))
    print("-" * 80)
    print(f"  📊 排除異常後 PASS 圖分數分佈  : Max={pg_max:.4f} | P99={pg_p99:.4f} | P95={pg_p95:.4f} | Median={pg_med:.4f} | Mean={pg_mean:.4f} | Std={pg_std:.4f}")
    print(f"  📐 排除異常後 PASS 門檻參考值  : mean+1*std={pg_mean+pg_std:.4f} | mean+2*std={pg_mean+2*pg_std:.4f} | mean+3*std={pg_mean+3*pg_std:.4f}")
print("=" * 80 + "\n")
EOF