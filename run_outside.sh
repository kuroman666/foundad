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

# --- Task 2: 無 GT 模擬抓漏 + 動態門檻對齊與防誤報畫紅框測試參數 ---
MOCK_SOURCE_CAT="${DATASETS_DIR}/mvtec/capsule" # 模擬抓漏的來源類別 (可改成其他類別)
SIM_POOL_DIR="${DATASETS_DIR}/sim_fab_capsule_pool"
SHOTS=4
FAB_CKPT_STEP=2000
THRESH_MODE="hybrid"                  # 門檻模式: "hybrid"(P95+MAD穩健統計+保底)、"mean_std"(直接用 mean+SIGMA_K*std 當門檻)、"percentile"、"top_k"、"fixed"
SIGMA_K=3.5                           # 當 THRESH_MODE="mean_std" 時為 mean + k*std 的倍數 k (如 1.0, 2.0, 3.0)；當為 "hybrid" 時為 MAD 標準差倍數
THRESHOLD=0.25                        # 當 THRESH_MODE="fixed" 時為絕對門檻；當為 "hybrid" 時為基礎門檻下限
MIN_THRESHOLD=0.27                    # 動態門檻最低安全底線 (適用於 hybrid / percentile / top_k)
AUTO_PERCENTILE=95.0                  # 自動對齊分位數 (P95)
TOP_K=0                               # 若設 > 0 (如 8)，最多只抓前 K 高分
CORE_RATIO=0.55                       # 異常熱區峰值核心收斂比例 (0.45~0.65)
WAFER_EDGE_MODE="auto"                # "auto" 自動偵測圓形物件 (如 bottle/cable/wafer) 並抑制外圈圓周反光誤報
WAFER_EDGE_MARGIN=0.04                # 圓周外圈倒角抑制寬度比例 (4%)
BORDER_RATIO=0.02                     # 四週矩形邊框 Padding 偽影抑制比例 (2%)
SPECKLE_FILTER_RATIO=0.008            # 灰階形態學開運算濾除單點微塵尖峰 (消除 tile/grid 單一 Patch 針狀極值誤報)
MIN_AREA_RATIO=0.00012                # 有效缺陷最小面積比例
APP_TRAIN="train_dinov3"
APP_TEST="test"
SKIP_FAB_TRAIN=""                     # 第一次跑設為 "" (會訓練 2000 epochs)；訓練完後可改 "--skip_train" 秒速重測不同門檻

# 高解析度切塊與縫合參數 (MVTec 約 1000x1000 可設 1024 不切塊或設 512 測試切塊；公司 5000x5000 設 512)
TILE_SIZE=1024
TILE_OVERLAP=64
MAX_TRAIN_TILES=48                    # 涵蓋更多局部結構與跨 Golden 圖公差，不增加訓練時間但大幅減少結構誤報
TRAIN_BATCH_SIZE=8
BG_STD_THRESH=1.5
# ===============================================================

echo "============================================================"
echo " [Task 1] 執行有 GT 公開資料集測試 -> ${OUTPUT_ROOT}/1_public_with_gt/<當下時間>"
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
echo " [Task 2] 執行無 GT 盲池測試 (Mode=${THRESH_MODE}, SIGMA_K=${SIGMA_K}) -> ${OUTPUT_ROOT}/2_fab_no_gt/<當下時間>"
echo "============================================================"
python 2_run_fab_no_gt.py \
    --mock_from_mvtec "${MOCK_SOURCE_CAT}" \
    --pool_dir "${SIM_POOL_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --ckpt "${DINO_CKPT_PATH}" \
    --out_dir "${OUTPUT_ROOT}/2_fab_no_gt" \
    --data_name "mvtec_sim_${SHOTS}shot" \
    --category "auto" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "sim_escape" \
    --ckpt_step "${FAB_CKPT_STEP}" \
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
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_FAB_TRAIN}

python - "${OUTPUT_ROOT}/2_fab_no_gt" "${MOCK_SOURCE_CAT}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base = sys.argv[1]
cat_name = os.path.basename(os.path.normpath(sys.argv[2]))

csv_files = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
if not csv_files:
    sys.exit(0)

latest_csv = csv_files[-1]
latest_txt = latest_csv.replace(".csv", ".txt")
eff_thresh = 0.25
if os.path.exists(latest_txt):
    with open(latest_txt, "r", encoding="utf-8") as tf:
        txt_head = tf.read(500)
        m = re.search(r"Effective Threshold:\s*([0-9.]+)", txt_head)
        if m:
            eff_thresh = float(m.group(1))

records = []
with open(latest_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

total = len(records)
if total == 0:
    sys.exit(0)

non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
non_golden_normal = [
    r for r in non_golden if not r["Original_Filename"].startswith("escape_")
]
escape_defects = [
    r for r in records if r["Original_Filename"].startswith("escape_")
]

target_normal = non_golden_normal if non_golden_normal else non_golden
non_g_scores = [float(r["Max_Score"]) for r in target_normal] if target_normal else [float(r["Max_Score"]) for r in records]

max_s = float(np.max(non_g_scores))
p99_s = float(np.percentile(non_g_scores, 99))
p95_s = float(np.percentile(non_g_scores, 95))
median_s = float(np.median(non_g_scores))
mean_s = float(np.mean(non_g_scores))
std_s = float(np.std(non_g_scores))
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

fp_normal = [r for r in target_normal if "ANOMALY" in r["Status"]]
caught_escape = [r for r in escape_defects if "ANOMALY" in r["Status"]]

print("\n" + "=" * 80)
print(f" 🎯 [{cat_name.upper()} 模擬無 GT 盲池 — 非 Golden 正常圖分數分佈與 Threshold 參考]")
print("-" * 80)
print(f"  📂 報表位置                   : {latest_csv}")
print(f"  🖼️ 盲池圖片結構               : 總計 {total} 張 (Golden: {total - len(non_golden)} 張 | 非Golden正常圖: {len(target_normal)} 張 | 隱藏Escape瑕疵: {len(escape_defects)} 張)")
print(f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}")
print(f"  🎯 抓漏與誤報表現             : 成功抓出瑕疵 {len(caught_escape)}/{len(escape_defects)} 張 | 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張")
print("-" * 80)
print(f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}")
print(f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}")
if escape_defects:
    esc_scores = [float(r["Max_Score"]) for r in escape_defects]
    print(f"  🔍 隱藏 Escape 瑕疵圖分數分佈  : Min={float(np.min(esc_scores)):.4f} | Median={float(np.median(esc_scores)):.4f} | Max={float(np.max(esc_scores)):.4f}")
print("=" * 80 + "\n")
EOF
