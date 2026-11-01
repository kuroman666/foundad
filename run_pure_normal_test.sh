#!/usr/bin/env bash
# 檔案位置: FoundAD/run_pure_normal_test.sh (全選覆蓋並存檔)
# 用途: 100% 純正常資料集誤報率 (False Positive Rate, FPR) 專項驗證腳本 (含圓周抑制、微塵開運算濾波與動態門檻對齊)
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto"                 # 公司外設 "auto"；進公司改為本地 DINO 權重路徑

# --- 1. 選擇測試來源 (支援單一類別或一次跑完 6 大類別) ---
# 可填入單一類別如 ("tile")，或一次驗證全部類別 ("bottle" "cable" "capsule" "grid" "tile" "transistor")
CATEGORIES=("bottle" "cable" "capsule" "grid" "tile" "transistor")

# 若在公司內部測試真實 100% 正常晶圓資料夾，請填寫下方路徑 (留空則預設跑上方 MVTec CATEGORIES)
CUSTOM_PURE_NORMAL_DIR=""

# --- 2. 模型、動態門檻與防誤報超參數 ---
SHOTS=4
FAB_CKPT_STEP=2000
THRESH_MODE="hybrid"                  # 門檻模式: "hybrid"(P95+MAD穩健上界+保底)、"mean_std"(直接用 mean+SIGMA_K*std 當門檻)、"percentile"、"top_k"、"fixed"
SIGMA_K=3.5                           # 當 THRESH_MODE="mean_std" 時為 mean + k*std 的倍數 k (如 1.0, 2.0, 3.0)；當為 "hybrid" 時為 MAD 標準差倍數
THRESHOLD=0.25                        # 當 THRESH_MODE="fixed" 時為絕對門檻；當為 "hybrid" 時為基礎下限
MIN_THRESHOLD=0.27                    # 動態門檻最低安全底線 (適用於 hybrid / percentile / top_k)
AUTO_PERCENTILE=95.0                  # 自動對齊分位數 (P95)
TOP_K=0                               # 若設 > 0 則限制最多輸出前 K 張
CORE_RATIO=0.55                       # 異常熱區峰值核心收斂比例 (0.45~0.65)

WAFER_EDGE_MODE="auto"                # "auto" 自動偵測圓形物件 (bottle/cable/wafer) 抑制外圈倒角反光；進公司跑晶圓可設 "on"
WAFER_EDGE_MARGIN=0.04                # 圓周外圈倒角抑制寬度佔半徑比例 (4%)
BORDER_RATIO=0.02                     # 影像四週矩形邊框抑制比例 (2%)
SPECKLE_FILTER_RATIO=0.008            # 灰階形態學開運算濾除單點微塵尖峰 (解決 tile/grid 單點極值誤報)
MIN_AREA_RATIO=0.00012                # 有效缺陷最小面積比例

APP_TRAIN="train_dinov3"              # 公司外用 "train_dinov3"；進公司若用預設設定可改為 "train"
APP_TEST="test"

# 若該類別剛訓練過且權重還在 ./logs/mvtec_sim_4shot/ 下，可設為 "--skip_train" 秒速重測推論！
# 注意：若一次迴圈跑多個不同類別，每個類別結構不同，請維持 SKIP_FAB_TRAIN="" 讓每個類別各自訓練
SKIP_FAB_TRAIN=""

DATA_NAME="mvtec_sim_${SHOTS}shot"
DIY_NAME="sim_escape"

# --- 3. 高解析度切塊與縫合參數 ---
TILE_SIZE=1024                        # MVTec 約 1000x1000 設 1024；公司 5000x5000 設 512
TILE_OVERLAP=64
MAX_TRAIN_TILES=48
TRAIN_BATCH_SIZE=8
BG_STD_THRESH=1.5
# ===============================================================

run_single_category() {
    local CAT="$1"
    local TARGET_POOL_DIR=""

    if [ -n "${CUSTOM_PURE_NORMAL_DIR}" ]; then
        TARGET_POOL_DIR="${CUSTOM_PURE_NORMAL_DIR}"
        echo "============================================================"
        echo " [純正常資料集驗證] 使用自訂 100% 正常圖片池: ${TARGET_POOL_DIR}"
        echo "============================================================"
    else
        local MVTEC_GOOD_DIR="${DATASETS_DIR}/mvtec/${CAT}/test/good"
        local PURE_POOL_DIR="${DATASETS_DIR}/sim_pure_normal_${CAT}_pool"
        if [ ! -d "${MVTEC_GOOD_DIR}" ]; then
            echo "[錯誤] 找不到 MVTec 正常測試集目錄: ${MVTEC_GOOD_DIR}"
            exit 1
        fi
        echo "============================================================"
        echo " [Step 0] 正在從 ${MVTEC_GOOD_DIR} 建立 100% 純正常盲池 (0 張瑕疵)..."
        echo "============================================================"
        rm -rf "${PURE_POOL_DIR}"
        mkdir -p "${PURE_POOL_DIR}"

        local count=0
        for img in "${MVTEC_GOOD_DIR}"/*; do
            if [ -f "${img}" ]; then
                local fname=$(basename "${img}")
                cp "${img}" "${PURE_POOL_DIR}/clean_${fname}"
                chmod 644 "${PURE_POOL_DIR}/clean_${fname}"
                count=$((count + 1))
            fi
        done
        echo " [✓] 100% 純正常盲池建立完成 (${CAT})：共複製 ${count} 張正常圖片 -> ${PURE_POOL_DIR}"
        TARGET_POOL_DIR="${PURE_POOL_DIR}"
    fi

    local SAVE_BASE_DIR="${OUTPUT_ROOT}/3_pure_normal_${CAT}"

    echo "============================================================"
    echo " [啟動無 GT 純正常池誤報測試] CATEGORY=${CAT} | THRESH_MODE=${THRESH_MODE} (SIGMA_K=${SIGMA_K})"
    echo " -> 輸出根目錄: ${SAVE_BASE_DIR}/<當下時間>"
    echo "============================================================"

    python 2_run_fab_no_gt.py \
        --pool_dir "${TARGET_POOL_DIR}" \
        --datasets_dir "${DATASETS_DIR}" \
        --ckpt "${DINO_CKPT_PATH}" \
        --out_dir "${SAVE_BASE_DIR}" \
        --data_name "${DATA_NAME}" \
        --category "${CAT}" \
        --app_train "${APP_TRAIN}" \
        --app_test "${APP_TEST}" \
        --diy_name "${DIY_NAME}" \
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

    python - "${SAVE_BASE_DIR}" "${CAT}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base = sys.argv[1]
cat_name = sys.argv[2]

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

scores = [float(r["Max_Score"]) for r in records]
fp_All = [r for r in records if "ANOMALY" in r["Status"]]
pass_All = [r for r in records if "PASS" in r["Status"]]
non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
fp_non_golden = [r for r in non_golden if "ANOMALY" in r["Status"]]

fpr_total = len(fp_All) / total * 100.0
fpr_non_golden = (len(fp_non_golden) / len(non_golden) * 100.0) if non_golden else 0.0

non_g_scores = [float(r["Max_Score"]) for r in non_golden] if non_golden else scores
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
print(f" 🎯 [{cat_name.upper()} 100% 純正常資料集 — 誤報率 (FPR) 驗算總結]")
print("-" * 80)
print(f"  📂 報表位置                   : {latest_csv}")
print(f"  🖼️ 測試池純正常圖片總數        : {total} 張 (含 {total - len(non_golden)} 張 Golden 圖)")
print(f"  🎚️ 動態對齊後有效門檻          : {eff_thresh:.4f}")
print("-" * 80)
print(f"  ✅ 正確放行 (PASS)             : {len(pass_All)} 張")
print(f"  🚨 誤報張數 (False Positives)  : {len(fp_All)} 張")
print(f"  📈 整體誤報率 (Overall FPR)    : {fpr_total:.2f}% ({len(fp_All)}/{total})")
print(f"  📈 扣除 Golden 後誤報率 (FPR)  : {fpr_non_golden:.2f}% ({len(fp_non_golden)}/{len(non_golden)})")
print("-" * 80)
print(f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}")
print(f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}")
print("=" * 80 + "\n")
EOF
}

if [ -n "${CUSTOM_PURE_NORMAL_DIR}" ]; then
    run_single_category "wafer"
else
    for cat in "${CATEGORIES[@]}"; do
        run_single_category "${cat}"
    done
fi
