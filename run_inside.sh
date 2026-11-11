#!/usr/bin/env bash
# 檔案位置: FoundAD/run_inside.sh (全選覆蓋並存檔)
# 用途: 公司內部實機 (Fab Inside) 完全無 GT 盲池四階段一鍵測試腳本
# 流程: 1_auto_golden.py -> 2_train_fab_no_gt.py -> 3_infer_fab_no_gt.py -> 4_split_eval_results.py
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./wafer_escape_results"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto" # 自動從 ~/.cache/torch/hub 或 ./hub_cache 離線搜尋 DINOv3/v2 權重

# 若執行時有帶參數 (如 bash run_inside.sh /path/to/lot1 /path/to/lot2)，優先使用命令列參數；
# 否則使用下方預設的公司內部無標籤盲池資料夾清單 (可放入單一或多個 Lot 資料夾)
if [ $# -ge 1 ]; then
  TARGET_POOL_DIRS=("$@")
else
  TARGET_POOL_DIRS=(
    "${DATASETS_DIR}/fab_wafer_pool"
  )
fi

# 選填：若想跨批號指定既有 Golden 基準或已訓練好的權重，可填寫路徑 (留空則各池自動獨立挑 Golden 與訓練)
CUSTOM_GOLDEN_SRC=""   # 例如: "--golden_src ./datasets/fab_sim_wafer_4shot"
CUSTOM_TRAINED_CKPT="" # 例如: "--trained_ckpt ./logs/fab_sim_wafer_4shot/dinov3fab_wafer/pretrained.pth.tar"

REBUILD_GOLDEN="" # 若想強制重挑 Golden，設為 "--rebuild_golden"
FORCE_TRAIN=""    # 若想強制重跑模型訓練，設為 "--force_train"

# --- 四階段執行控制開關 (內建自動防呆：有 Golden 不重挑、有權重不重訓) ---
SKIP_GOLDEN=0
SKIP_FAB_TRAIN=""
FORCE_INFER=""
RUN_SPLIT_EVAL=1

# --- Stage 1 & 2 候選池與切塊參數 (針對公司內部大量高解析度晶圓圖優化) ---
SHOTS=4
CANDIDATE_POOL_SIZE=16 # Stage 1 初篩進入 Stage 2 切塊對決的候選晶圓數
MAX_COARSE_SCAN=250    # 面對上千張大盲池時，Stage 1 快速抽樣掃描上限

TILE_SIZE=512
TILE_OVERLAP=64
MIN_TILE_TRIGGER=1280  # 原圖長或寬 > 1280px 自動啟動高解析度切塊；<= 1280px 保持整張 1:1 對齊
MAX_TRAIN_TILES=48
TRAIN_BATCH_SIZE=8
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""      # 若想強制關閉切塊，設為 "--disable_tiling"

# --- 模型、動態門檻與四合一防誤報超參數 ---
FAB_CKPT_STEP=2000
APP_TRAIN="train_dinov3"
APP_TEST="test"

THRESH_MODE="hybrid"   # 可選: hybrid, mean_std, percentile, top_k, fixed
SIGMA_K=3.5
THRESHOLD=0
MIN_THRESHOLD=0
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE=""      # 若想關閉濾波看原始分數，設為 "--raw_score_mode"

WAFER_EDGE_MODE="auto" # 可選: auto, on, off (自動偵測圓形晶圓並抑制邊緣倒角偽影)
WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012
MICRO_DEFECT_MODE="--micro_defect_mode" # 一鍵啟動晶圓微瑕疵高靈敏模式 (白頂帽增強 + 防稀釋尖峰)
MICRO_BOOST_WEIGHT=0.35
MICRO_PEAK_PIXELS=24

# --- Step 4 評估結果拆分參數 ---
NUM_PACKS=3
SPLIT_COUNTS=""        # 可填固定張數如 "500" 或 "700,700,600"；留空則依 SPLIT_RATIOS 比例均分
SPLIT_RATIOS="1:1:1"
SPLIT_ORDER="stratified"
RECOMPUTE_PACK_THRESH="--recompute_pack_thresh"
EXCLUDE_GOLDEN=""      # 若拆分時想排除 4 張 Golden 圖，設為 "--exclude_golden"
USE_SYMLINK="--symlink"
# ===============================================================

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/inside_multi_pool_summary_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_inside_summary.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_inside_summary.csv"

echo "==========================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 🏭 [FoundAD 公司內部無 GT 盲池批次訓練與推論總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Pool_Key,Category,Total_Imgs,Golden_Imgs,Unseen_Test_Imgs,Flagged_Anomalies,Anomaly_Rate_Pct,Eff_Threshold,Core_Normal_Med,Core_Normal_Mean,Core_Normal_Std,Train_Dir,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

run_single_inside_pool() {
  local POOL_DIR="$1"
  if [ ! -d "${POOL_DIR}" ]; then
    echo "❌ [路徑錯誤] 找不到公司內部測試盲池資料夾: ${POOL_DIR}"
    exit 1
  fi

  local RAW_CAT_NAME
  RAW_CAT_NAME=$(basename "$(cd "${POOL_DIR}" && pwd)")
  local CAT_NAME="${RAW_CAT_NAME// /_}"
  local PARENT_DIR_NAME
  PARENT_DIR_NAME=$(basename "$(dirname "$(cd "${POOL_DIR}" && pwd)")")
  local DS_PREFIX="${PARENT_DIR_NAME//-/_}"
  DS_PREFIX="${DS_PREFIX// /_}"

  # 自動產生唯一識別標籤，防止不同 Lot / 站點的權重與輸出互相覆蓋
  local UNIQUE_TAG="${CAT_NAME}"
  if [ "${DS_PREFIX}" != "datasets" ] && [ "${DS_PREFIX}" != "." ] && [ -n "${DS_PREFIX}" ]; then
    UNIQUE_TAG="${DS_PREFIX}_${CAT_NAME}"
  fi

  local DATA_NAME="fab_inside_${UNIQUE_TAG}_${SHOTS}shot"
  local DIY_NAME="fab_inside_${UNIQUE_TAG}"
  local SAVE_BASE_DIR="${OUTPUT_ROOT}/inside_no_gt_${UNIQUE_TAG}"

  echo ""
  echo "==========================================================================="
  echo " 🚀 [公司內部無 GT 實機測試啟動] 盲池路徑: ${POOL_DIR}"
  echo "    CATEGORY : ${CAT_NAME} | UNIQUE_TAG: ${UNIQUE_TAG}"
  echo "    DATA_NAME: ${DATA_NAME} | DIY_NAME: ${DIY_NAME}"
  echo "    獨立輸出目錄: ${SAVE_BASE_DIR}"
  echo "==========================================================================="

  # Step 1: 執行 1_auto_golden.py (從無 GT 盲池自動挑選最純淨 Golden 並切塊純化)
  local EXISTING_GOLDEN_COUNT=0
  if [ -d "${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME}" ]; then
    EXISTING_GOLDEN_COUNT=$(find "${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \) | wc -l)
  fi

  if [ -n "${CUSTOM_GOLDEN_SRC}" ]; then
    echo " [⚡ Step 1/4 略過] 已指定外部 Golden 來源 (${CUSTOM_GOLDEN_SRC})，將於推論階段直接載入！"
  elif [ "${SKIP_GOLDEN}" -eq 0 ] && [ -z "${REBUILD_GOLDEN}" ] && [ "${EXISTING_GOLDEN_COUNT}" -gt 0 ]; then
    echo " [⚡ Step 1/4 防呆略過] ${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME} 已存在 ${EXISTING_GOLDEN_COUNT} 張 Golden 訓練樣本，直接沿用不重挑！"
  elif [ "${SKIP_GOLDEN}" -eq 0 ] || [ -n "${REBUILD_GOLDEN}" ]; then
    echo "---------------------------------------------------------------------------"
    echo " [Step 1/4] 執行 1_auto_golden.py -> 自動挑選並純化 ${CAT_NAME} Golden 訓練集"
    echo "---------------------------------------------------------------------------"
    python 1_auto_golden.py \
      --pool_dir "${POOL_DIR}" \
      --ckpt "${DINO_CKPT_PATH}" \
      --datasets_dir "${DATASETS_DIR}" \
      --data_name "${DATA_NAME}" \
      --category "${CAT_NAME}" \
      --shots "${SHOTS}" \
      --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
      --max_coarse_scan "${MAX_COARSE_SCAN}" \
      --tile_size "${TILE_SIZE}" \
      --tile_overlap "${TILE_OVERLAP}" \
      --min_tile_trigger "${MIN_TILE_TRIGGER}" \
      --max_train_tiles "${MAX_TRAIN_TILES}" \
      --bg_std_thresh "${BG_STD_THRESH}" \
      --report_dir "${SAVE_BASE_DIR}/golden_reports_${CAT_NAME}" \
      ${REBUILD_GOLDEN} ${DISABLE_TILING}
  fi

  # Step 2: 執行 2_train_fab_no_gt.py (若未指定外部已訓練權重，則執行訓練或自動沿用既有權重)
  if [ -n "${CUSTOM_TRAINED_CKPT}" ]; then
    echo " [⚡ Step 2/4 略過] 已指定外部訓練權重 (${CUSTOM_TRAINED_CKPT})，直接進入 Step 3 推論！"
  else
    echo "---------------------------------------------------------------------------"
    echo " [Step 2/4] 執行 2_train_fab_no_gt.py -> 模型訓練與權重對接檢查"
    echo "---------------------------------------------------------------------------"
    python 2_train_fab_no_gt.py \
      --pool_dir "${POOL_DIR}" \
      --ckpt "${DINO_CKPT_PATH}" \
      --out_dir "${SAVE_BASE_DIR}" \
      --datasets_dir "${DATASETS_DIR}" \
      --data_name "${DATA_NAME}" \
      --category "${CAT_NAME}" \
      --app_train "${APP_TRAIN}" \
      --diy_name "${DIY_NAME}" \
      --ckpt_step "${FAB_CKPT_STEP}" \
      --shots "${SHOTS}" \
      --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
      --max_coarse_scan "${MAX_COARSE_SCAN}" \
      --tile_size "${TILE_SIZE}" \
      --tile_overlap "${TILE_OVERLAP}" \
      --min_tile_trigger "${MIN_TILE_TRIGGER}" \
      --max_train_tiles "${MAX_TRAIN_TILES}" \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      --bg_std_thresh "${BG_STD_THRESH}" \
      ${SKIP_FAB_TRAIN} ${FORCE_TRAIN} ${REBUILD_GOLDEN} ${DISABLE_TILING}
  fi

  # Step 3: 執行 3_infer_fab_no_gt.py (高解析度切塊推論、RAM 即時拼接銷毀碎片與紅框定位)
  echo "---------------------------------------------------------------------------"
  echo " [Step 3/4] 執行 3_infer_fab_no_gt.py -> 高解析度切塊推論與微瑕疵紅框定位"
  echo "---------------------------------------------------------------------------"
  python 3_infer_fab_no_gt.py \
    --pool_dir "${POOL_DIR}" \
    --ckpt "${DINO_CKPT_PATH}" \
    --out_dir "${SAVE_BASE_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CAT_NAME}" \
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
    --micro_boost_weight "${MICRO_BOOST_WEIGHT}" \
    --micro_peak_pixels "${MICRO_PEAK_PIXELS}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --min_tile_trigger "${MIN_TILE_TRIGGER}" \
    --max_train_tiles "${MAX_TRAIN_TILES}" \
    --infer_batch_size "${INFER_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${CUSTOM_GOLDEN_SRC} ${CUSTOM_TRAINED_CKPT} \
    ${FORCE_INFER} ${RAW_SCORE_MODE} ${MICRO_DEFECT_MODE} ${DISABLE_TILING}

  # Step 4: 執行 4_split_eval_results.py (評估結果拆分為多包獨立 Eval 報表)
  if [ "${RUN_SPLIT_EVAL}" -eq 1 ]; then
    local SPLIT_SCRIPT=""
    if [ -f "4_split_eval_results.py" ]; then
      SPLIT_SCRIPT="4_split_eval_results.py"
    elif [ -f "3_split_eval_results.py" ]; then
      SPLIT_SCRIPT="3_split_eval_results.py"
    elif [ -f "fab_tools/3_split_eval_results.py" ]; then
      SPLIT_SCRIPT="fab_tools/3_split_eval_results.py"
    fi

    if [ -n "${SPLIT_SCRIPT}" ]; then
      echo "---------------------------------------------------------------------------"
      echo " [Step 4/4] 執行 ${SPLIT_SCRIPT} -> 拆分為 ${NUM_PACKS} 份獨立 Eval 報表"
      echo "---------------------------------------------------------------------------"
      local SPLIT_ARGS=()
      if [ -n "${SPLIT_COUNTS}" ]; then
        SPLIT_ARGS+=(--counts "${SPLIT_COUNTS}")
      elif [ -n "${SPLIT_RATIOS}" ]; then
        if [ "${SPLIT_RATIOS}" = "1:1:1" ] && [ "${NUM_PACKS}" -ne 3 ]; then
          local AUTO_RATIOS
          AUTO_RATIOS=$(printf "1:%.0s" $(seq 1 "${NUM_PACKS}") | sed 's/:$//')
          SPLIT_ARGS+=(--ratios "${AUTO_RATIOS}")
        else
          SPLIT_ARGS+=(--ratios "${SPLIT_RATIOS}")
        fi
      fi

      python "${SPLIT_SCRIPT}" \
        --src_dir "${SAVE_BASE_DIR}" \
        --num_packs "${NUM_PACKS}" \
        --split_order "${SPLIT_ORDER}" \
        --sigma_k "${SIGMA_K}" \
        --auto_percentile "${AUTO_PERCENTILE}" \
        --min_threshold "${MIN_THRESHOLD}" \
        --threshold "${THRESHOLD}" \
        --top_k "${TOP_K}" \
        "${SPLIT_ARGS[@]}" \
        ${RECOMPUTE_PACK_THRESH} ${EXCLUDE_GOLDEN} ${USE_SYMLINK}
    fi
  fi

  # 終端機列印該盲池無 GT 統計總結，並同步 Append 寫入多盲池總表
  python - "${SAVE_BASE_DIR}" "${CAT_NAME}" "${UNIQUE_TAG}" "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_CSV}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base, cat_name, unique_tag, batch_txt, batch_csv = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)

all_csvs = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
csv_files = [p for p in all_csvs if not os.path.basename(os.path.dirname(p)).startswith("eval_pack_")]
if not csv_files:
    sys.exit(0)

latest_csv = csv_files[-1]
infer_dir = os.path.dirname(latest_csv)
train_dirs = sorted(glob.glob(os.path.join(save_base, "train_*")))
latest_train_dir = train_dirs[-1] if train_dirs else save_base

latest_txt = latest_csv.replace(".csv", ".txt")
eff_thresh = 0.25
overfit_line = ""
score_mode_str = "MICRO_DEFECT_SENSITIVE"
if os.path.exists(latest_txt):
    with open(latest_txt, "r", encoding="utf-8") as tf:
        txt_head = tf.read(1800)
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

golden_recs = [r for r in records if str(r["Status"]).startswith("GOLDEN")]
non_golden = [r for r in records if not str(r["Status"]).startswith("GOLDEN")]
flagged_anomalies = [r for r in non_golden if "ANOMALY" in str(r["Status"])]
passed_normals = [r for r in non_golden if not "ANOMALY" in str(r["Status"])]

# 真無 GT 環境：優先統計未超門檻的正常樣本分佈；若全數超門檻則取非 Golden 左半 50% 核心
target_normal = passed_normals if passed_normals else (non_golden if non_golden else records)
norm_scores = [float(r["Max_Score"]) for r in target_normal]

max_s = float(np.max(norm_scores))
p99_s = float(np.percentile(norm_scores, 99))
p95_s = float(np.percentile(norm_scores, 95))
median_s = float(np.median(norm_scores))
mean_s = float(np.mean(norm_scores))
std_s = float(np.std(norm_scores))
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

anom_rate_pct = (len(flagged_anomalies) / len(non_golden) * 100.0) if non_golden else 0.0

summary_block = [
    "=" * 90,
    f" 🏭 [{unique_tag.upper()} 公司內部無 GT 盲池測試總結 — 分數分佈與門檻參考 ({score_mode_str})]",
    "-" * 90,
    f" 📂 訓練紀錄資料夾         : {os.path.abspath(latest_train_dir)}",
    f" 📂 推論紀錄資料夾         : {os.path.abspath(infer_dir)}",
    f" 📄 最新排名報表位置       : {os.path.abspath(latest_csv)}",
    f" 🖼️ 盲池圖片結構           : 總計 {total} 張 (Golden訓練基準: {len(golden_recs)} 張 | 待測盲池圖片: {len(non_golden)} 張)",
    f" 🎚️ 本次動態對齊後有效門檻 : {eff_thresh:.4f}",
    f" 🚨 無GT異常檢出統計       : 抓出異常並畫框 {len(flagged_anomalies)}/{len(non_golden)} 張 (檢出率 Anomaly Rate={anom_rate_pct:.2f}%) | 正常略過 {len(passed_normals)} 張",
]
if overfit_line:
    summary_block.append(f" 🩺 模型過擬合自檢 (Overfit) : {overfit_line}")
summary_block.extend([
    "-" * 90,
    f" 📊 盲池正常核心分數分佈   : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}",
    f" 📐 盲池正常核心門檻參考值 : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}",
])
if flagged_anomalies:
    anom_scores = [float(r["Max_Score"]) for r in flagged_anomalies]
    top_names = [f"{r['Original_Filename']}({float(r['Max_Score']):.3f})" for r in flagged_anomalies[:5]]
    summary_block.append(
        f" 🔍 檢出異常圖分數分佈     : Min={float(np.min(anom_scores)):.4f} | Median={float(np.median(anom_scores)):.4f} | Max={float(np.max(anom_scores)):.4f}"
    )
    summary_block.append(
        f" 🏆 Top-5 最高分異常樣本   : {', '.join(top_names)}"
    )
summary_block.append("=" * 90 + "\n")

full_text = "\n".join(summary_block)
print("\n" + full_text)

with open(batch_txt, "a", encoding="utf-8") as bf:
    bf.write(full_text + "\n")

with open(batch_csv, "a", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow([
        unique_tag,
        cat_name,
        total,
        len(golden_recs),
        len(non_golden),
        len(flagged_anomalies),
        f"{anom_rate_pct:.2f}",
        f"{eff_thresh:.4f}",
        f"{median_s:.4f}",
        f"{mean_s:.4f}",
        f"{std_s:.4f}",
        os.path.abspath(latest_train_dir),
        os.path.abspath(infer_dir),
        os.path.abspath(latest_csv),
    ])
EOF
}

for pool_path in "${TARGET_POOL_DIRS[@]}"; do
  run_single_inside_pool "${pool_path}"
done

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================="
echo " ✅ 公司內部全部 ${#TARGET_POOL_DIRS[@]} 個無 GT 測試盲池執行完畢！所有紀錄與紅框圖皆已獨立保存："
echo " 📄 本次完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo " 📄 最新完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo " 📊 最新完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo "=========================================================================================================="