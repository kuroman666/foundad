#!/usr/bin/env bash
# 檔案位置: FoundAD/run_pure_normal_test.sh (全選覆蓋並存檔)
# 用途: 100% 純正常資料集誤報率 (False Positive Rate, FPR) 專項驗證腳本
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto"                 # 公司外設 "auto"；進公司改為本地 DINO 權重路徑

# --- 1. 選擇測試來源 (二選一) ---
# [選項 A] 使用 MVTec 指定類別的 test/good (100% 純正常測試集)
CATEGORY="grid"                    # 可改為 bottle, cable, capsule, grid, tile, transistor
MVTEC_GOOD_DIR="${DATASETS_DIR}/mvtec/${CATEGORY}/test/good"
PURE_POOL_DIR="${DATASETS_DIR}/sim_pure_normal_${CATEGORY}_pool"

# [選項 B] 若在公司內部測試真實 100% 正常晶圓資料夾，請填寫下方路徑 (留空則預設使用上方選項 A)
CUSTOM_PURE_NORMAL_DIR=""

# --- 2. 模型與異常判定門檻超參數 ---
SHOTS=4
FAB_CKPT_STEP=2000
THRESHOLD=0.25                        # 判定異常門檻：最高分 >= THRESHOLD 視為誤報 (False Positive) 並畫紅框輸出
CORE_RATIO=0.55                       # 異常熱區峰值核心收斂比例 (0.45~0.65)
SIGMA_K=3.2                           # 若將 THRESHOLD 設為 -1，則啟用 Golden μ + kσ 自動門檻
APP_TRAIN="train_dinov3"              # 公司外用 "train_dinov3"；進公司若用預設設定可改為 "train"
APP_TEST="test"

# 若先前已跑過 run_outside.sh 訓練完畢，可改為 "--skip_train" 直接秒速測試不同 THRESHOLD 的誤報張數！
# 第一次獨立執行、或更換 CATEGORY 時，請維持 "" 讓模型重新訓練 2000 steps
SKIP_FAB_TRAIN=""

# 對齊 run_outside.sh 的名稱，方便直接共用已訓練好的權重
DATA_NAME="mvtec_sim_${SHOTS}shot"
DIY_NAME="sim_escape"

# --- 3. 高解析度切塊與縫合參數 ---
TILE_SIZE=1024                        # MVTec 約 1000x1000 可設 1024 或 512；公司 5000x5000 建議設 512
TILE_OVERLAP=64
MAX_TRAIN_TILES=16
TRAIN_BATCH_SIZE=8
BG_STD_THRESH=1.5
# ===============================================================

if [ -n "${CUSTOM_PURE_NORMAL_DIR}" ]; then
    TARGET_POOL_DIR="${CUSTOM_PURE_NORMAL_DIR}"
    echo "============================================================"
    echo " [純正常資料集驗證] 使用自訂 100% 正常圖片池: ${TARGET_POOL_DIR}"
    echo "============================================================"
else
    if [ ! -d "${MVTEC_GOOD_DIR}" ]; then
        echo "[錯誤] 找不到 MVTec 正常測試集目錄: ${MVTEC_GOOD_DIR}"
        exit 1
    fi
    echo "============================================================"
    echo " [Step 0] 正在從 ${MVTEC_GOOD_DIR} 建立 100% 純正常盲池 (0 張瑕疵)..."
    echo "============================================================"
    rm -rf "${PURE_POOL_DIR}"
    mkdir -p "${PURE_POOL_DIR}"

    count=0
    for img in "${MVTEC_GOOD_DIR}"/*; do
        if [ -f "${img}" ]; then
            fname=$(basename "${img}")
            cp "${img}" "${PURE_POOL_DIR}/clean_${fname}"
            chmod 644 "${PURE_POOL_DIR}/clean_${fname}"
            count=$((count + 1))
        fi
    done
    echo " [✓] 100% 純正常盲池建立完成：共複製 ${count} 張正常圖片 -> ${PURE_POOL_DIR}"
    TARGET_POOL_DIR="${PURE_POOL_DIR}"
fi

SAVE_BASE_DIR="${OUTPUT_ROOT}/3_pure_normal_${CATEGORY}"

echo "============================================================"
echo " [啟動無 GT 純正常池誤報測試] THRESHOLD=${THRESHOLD} | CATEGORY=${CATEGORY}"
echo " -> 輸出根目錄: ${SAVE_BASE_DIR}/<當下時間>"
echo "============================================================"

# 注意：此處故意「不傳入 --mock_from_mvtec」，確保不會混入任何異常瑕疵圖片
# 同時明確指定 --category "${CATEGORY}"，避免自動抓資料夾名稱時誤抓成 "good"
python 2_run_fab_no_gt.py \
    --pool_dir "${TARGET_POOL_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --ckpt "${DINO_CKPT_PATH}" \
    --out_dir "${SAVE_BASE_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CATEGORY}" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "${DIY_NAME}" \
    --ckpt_step "${FAB_CKPT_STEP}" \
    --shots "${SHOTS}" \
    --threshold "${THRESHOLD}" \
    --core_ratio "${CORE_RATIO}" \
    --sigma_k "${SIGMA_K}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_FAB_TRAIN}

# 自動抓取最新時間戳記資料夾中的 CSV 報表，直接統計誤報張數與 FPR 並推薦最佳門檻
python - "${SAVE_BASE_DIR}" "${THRESHOLD}" << 'EOF'
import os
import sys
import csv
import glob
import numpy as np

save_base = sys.argv[1]
thresh = float(sys.argv[2])

csv_files = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
if not csv_files:
    sys.exit(0)

latest_csv = csv_files[-1]
run_dir = os.path.dirname(latest_csv)

records = []
with open(latest_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

total = len(records)
if total == 0:
    sys.exit(0)

scores = [float(r["Max_Score"]) for r in records]
fp_All = [r for r in records if "ANOMALY" in r["Status"]]
pass_All = [r for r in records if "PASS" in r["Status"]]
non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
fp_non_golden = [r for r in non_golden if "ANOMALY" in r["Status"]]

fpr_total = len(fp_All) / total * 100.0
fpr_non_golden = (len(fp_non_golden) / len(non_golden) * 100.0) if non_golden else 0.0

max_s = float(np.max(scores))
p99_s = float(np.percentile(scores, 99))
p95_s = float(np.percentile(scores, 95))
median_s = float(np.median(scores))
mean_s = float(np.mean(scores))

print("\n" + "=" * 75)
print(" 🎯 [100% 純正常資料集 — 誤報率 (False Positive Rate) 驗算總結]")
print("-" * 75)
print(f"  📂 報表位置                   : {latest_csv}")
print(f"  🖼️ 測試池純正常圖片總數        : {total} 張 (含 {total - len(non_golden)} 張自動挑選之 Golden 圖)")
print(f"  🎚️ 當前設定門檻 (Threshold)    : {thresh:.4f}")
print("-" * 75)
print(f"  ✅ 正確放行 (PASS_SKIPPED)     : {len(pass_All)} 張")
print(f"  🚨 誤報張數 (False Positives)  : {len(fp_All)} 張 (在 anomaly_images/ 中被畫紅框的正常圖)")
print(f"  📈 整體誤報率 (Overall FPR)    : {fpr_total:.2f}% ({len(fp_All)}/{total})")
print(f"  📈 扣除 Golden 後誤報率 (FPR)  : {fpr_non_golden:.2f}% ({len(fp_non_golden)}/{len(non_golden)})")
print("-" * 75)
print("  📊 純正常圖片 Max_Score 分數分佈 (可用於校準門檻):")
print(f"      ├─ 全池最高分 (Max)        : {max_s:.4f}  <-- 若 Threshold 設為 > {max_s:.4f} 可達成 0 誤報 (0% FPR)")
print(f"      ├─ 99% 分位數 (P99)        : {p99_s:.4f}  <-- 若設為此值，誤報率約 1%")
print(f"      ├─ 95% 分位數 (P95)        : {p95_s:.4f}  <-- 若設為此值，誤報率約 5%")
print(f"      ├─ 中位數 (Median)         : {median_s:.4f}")
print(f"      └─ 平均值 (Mean)           : {mean_s:.4f}")
if fp_All:
    print("-" * 75)
    print("  ⚠️ 被誤報的正常圖片清單 (Top 10 False Positives):")
    for r in fp_All[:10]:
        print(f"      - Rank {int(r['Rank']):02d} | {r['Original_Filename']:<22} | Score: {float(r['Max_Score']):.4f} | BBoxes: {r['Num_BBoxes']}")
print("=" * 75 + "\n")
EOF
