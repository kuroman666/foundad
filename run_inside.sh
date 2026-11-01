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

# 4. 模型與動態異常門檻對齊超參數
SHOTS=4                  # 自動挑選 4 張最純淨的 Golden Image
CKPT_STEP=2000           # FoundAD 訓練步數
THRESH_MODE="hybrid"     # 門檻模式: "hybrid"(P95+MAD穩健統計+保底)、"mean_std"(直接用 mean+SIGMA_K*std 當門檻)、"percentile"(分位數)、"top_k"(前K張)、"fixed"(固定門檻)
SIGMA_K=3.5              # 當 THRESH_MODE="mean_std" 時為 mean + k*std 的倍數 k (如 1.0, 2.0, 3.0)；當為 "hybrid" 時為 MAD 標準差倍數
THRESHOLD=0.25           # 當 THRESH_MODE="fixed" 時為絕對門檻；當為 "hybrid" 時作為基礎下限
MIN_THRESHOLD=0.28       # 動態門檻的絕對最低安全底線 (適用於 hybrid / percentile / top_k)
AUTO_PERCENTILE=95.0     # 自動對齊盲池分位數 (例如 95.0 代表 P95；若想更嚴格防誤報可設 98.0)
TOP_K=0                  # 若設 > 0 (例如 10)，則限制每次最多只輸出最可疑的前 K 張晶圓供工程師複檢 (設 0 代表不限制)
CORE_RATIO=0.55          # 紅框核心收斂比例 (0.45~0.65，越高紅框越緊貼最高分缺陷中心)
APP_TRAIN="train"        # 對應 foundad/configs/app/ 下的訓練設定檔
APP_TEST="test"          # 對應 foundad/configs/app/ 下的測試設定檔
SKIP_TRAIN=""            # 第一次跑設為 ""；若已訓練完只想秒速重調門檻或邊緣參數，可設為 "--skip_train"

# 5. 晶圓圓周邊緣抑制 (Wafer Bevel Exclusion) 與單點微塵濾除參數
WAFER_EDGE_MODE="on"     # 進公司跑圓形晶圓強烈建議設 "on" (或 "auto")，自動抑制晶圓最外圈圓弧倒角反光與 Notch 邊緣誤報
WAFER_EDGE_MARGIN=0.04   # 晶圓圓周外圈倒角抑制寬度佔半徑比例 (0.04 代表外縮 4% 半徑平滑衰減)
BORDER_RATIO=0.02        # 影像四週矩形邊框 Padding 偽影抑制比例
SPECKLE_FILTER_RATIO=0.008 # 灰階形態學開運算濾除單點微塵尖峰之比例 (消除單一 Patch 針狀噪點誤報)
MIN_AREA_RATIO=0.00012   # 有效缺陷最小面積比例 (低於此面積之微塵不畫框、不輸出)

# 6. 5000x5000 高解析度防失真切塊 (Tiling) 與無縫縫合參數
TILE_SIZE=512            # 切塊大小 (512 對應模型原生輸入解析度，達成 1:1 零壓縮失真；若想加速可設 1024)
TILE_OVERLAP=64          # 相鄰圖塊重疊像素 (用於 2D 餘弦平滑拼接，消除邊界接縫)
MAX_TRAIN_TILES=48       # 調大至 48 (或 64)，涵蓋中心、中環、外環與跨 Golden 晶圓公差，訓練時間不變但大幅降低結構誤報！
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
echo " [防誤報設定] THRESH_MODE=${THRESH_MODE} (SIGMA_K=${SIGMA_K}, P${AUTO_PERCENTILE}, Min=${MIN_THRESHOLD}) | WAFER_EDGE=${WAFER_EDGE_MODE} (${WAFER_EDGE_MARGIN})"
echo "============================================================"

python 2_run_fab_no_gt.py \
    --pool_dir "${WAFER_POOL_DIR}" \
    --ckpt "${COMPANY_DINO_CKPT}" \
    --out_dir "${FAB_RESULT_DIR}" \
    --data_name "mvtec_fab_${SHOTS}shot" \
    --category "wafer" \
    --app_train "${APP_TRAIN}" \
    --app_test "${APP_TEST}" \
    --diy_name "fab_h100" \
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
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${SKIP_TRAIN}

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
non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
pass_non_golden = [r for r in non_golden if "PASS" in r["Status"]]
anom_all = [r for r in records if "ANOMALY" in r["Status"]]

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
print(" 🎯 [公司內部無 GT 晶圓池 — 非 Golden 正常圖分數分佈與 Threshold 參考總結]")
print("-" * 80)
print(f"  📂 報表位置                   : {latest_csv}")
print(f"  🖼️ 測試池圖片總數              : {total} 張 (含 {total - len(non_golden)} 張 Golden 圖, 抓出 {len(anom_all)} 張異常)")
print(f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}")
print("-" * 80)
print(f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}")
print(f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}")
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
