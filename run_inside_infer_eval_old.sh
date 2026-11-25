#!/usr/bin/env bash
# 檔案位置: FoundAD/run_inside_infer_eval_old.sh
# 用途: 公司內部實機 (Fab Inside) 舊版僅執行「推論 + 結果拆分評估」二階段一鍵腳本
# 用法範例:
#   1. 預設抓對應權重與預設門檻(0.18): bash run_inside_infer_eval_old.sh ./datasets/fab_wafer_pool
#   2. 命令列直接指定權重檔與固定門檻: bash run_inside_infer_eval_old.sh ./datasets/new_pool ./logs/wafer_A/dinov3/train-step2000.pth.tar 0.32
# 流程: 3_infer_fab_no_gt_old.py -> 4_split_eval_results.py (不重挑 Golden、不重新訓練模型)
set -e

# ===== DINOv3 Offline Settings =====

COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="$(pwd)/hub_cache/facebookresearch_dinov3_main"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 檢查離線模型資源是否存在
if [ ! -f "${FAB_LOCAL_DINO_CKPT}" ]; then
    echo "ERROR: DINOv3 pretrained checkpoint not found"
    exit 1
fi

if [ ! -f "${FAB_LOCAL_DINO_REPO}/hubconf.py" ]; then
    echo "ERROR: Local DINOv3 repository not found"
    exit 1
fi

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./wafer_escape_results"
DATASETS_DIR="./datasets"

# 1. 選填：直接指定訓練好的權重檔 (.pth.tar) 或權重資料夾路徑 (留空 "" 則自動抓預設對應的權重)
CUSTOM_CKPT_PATH=""

# 2. 選填：直接指定固定門檻數值，例如 "0.32" (留空 "" 且命令列也未傳入數字時，自動抓預設門檻 0.18)
CUSTOM_THRESHOLD=""
DEFAULT_THRESHOLD="0.0"

# 選填：舊版手動指定 DATA_NAME 與 DIY_NAME (若已填 CUSTOM_CKPT_PATH 則會自動解析，無須重複填寫)
CUSTOM_DATA_NAME=""
CUSTOM_DIY_NAME=""

# 複製所有命令列參數至陣列，自動提取數字門檻與 .pth.tar 權重路徑
CLI_ARGS=()
CLI_THRESHOLD=""
for arg in "$@"; do
  if [[ "${arg}" =~ ^[0-9]+(\.[0-9]+)?$ || "${arg}" =~ ^\.[0-9]+$ ]]; then
    CLI_THRESHOLD="${arg}"
  elif [[ "${arg}" == *.pth.tar ]]; then
    CUSTOM_CKPT_PATH="${arg}"
  else
    CLI_ARGS+=("${arg}")
  fi
done

# 優先順序：命令列傳入的數字 > 腳本內設定的 CUSTOM_THRESHOLD > 環境變數 THRESHOLD > 預設值 DEFAULT_THRESHOLD (0.25)
THRESHOLD="${CLI_THRESHOLD:-${CUSTOM_THRESHOLD:-${THRESHOLD:-${DEFAULT_THRESHOLD}}}}"

# 剩餘的參數即為盲池資料夾清單；若未傳入任何資料夾，則使用預設路徑
if [ ${#CLI_ARGS[@]} -ge 1 ]; then
  TARGET_POOL_DIRS=("${CLI_ARGS[@]}")
else
  TARGET_POOL_DIRS=(
    "${DATASETS_DIR}/fab_wafer_pool"
  )
fi

# --- 二階段執行控制開關 ---
FORCE_INFER=""         # 若想忽略測試池切塊 PNG 快取、強制重新切塊推論，設為 "--force_infer"
RUN_SPLIT_EVAL=1       # 1: 推論完自動執行 4_split_eval_results.py；0: 僅執行推論

# --- 切塊與推論批次參數 (須與訓練時設定一致以命中快取，對應 3_infer_fab_no_gt_old.py) ---
SHOTS=4                # 用於對齊預設 DATA_NAME 命名 (${SHOTS}shot)
TILE_SIZE=512
TILE_OVERLAP=64
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""      # 若想強制關閉切塊，設為 "--disable_tiling"

# --- 模型、固定/動態門檻與四合一防誤報超參數 (對應 3_infer_fab_no_gt_old.py) ---
# 若外部未指定 FAB_CKPT_STEP 或 CKPT_STEP，預設傳入 -1 以自動抓取最大 step
FAB_CKPT_STEP="${FAB_CKPT_STEP:-${CKPT_STEP:--1}}"
APP_TEST="test"

THRESH_MODE="auto"   # 已固定使用 fixed 模式；可選: hybrid, mean_std, percentile, top_k, fixed
SIGMA_K=3.5
MIN_THRESHOLD="${MIN_THRESHOLD:-0}"     # 動態門檻的絕對最低安全底線 (fixed 模式下不影響)
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE=""      # 若想關閉四合一濾波看原始單點最大值，設為 "--raw_score_mode"

WAFER_EDGE_MODE="auto" # 可選: auto, on, off (自動偵測圓形晶圓並抑制邊緣倒角偽影)
WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012

# --- Step 4 評估結果拆分參數 (對應 4_split_eval_results.py) ---
NUM_PACKS=3
SPLIT_COUNTS=""        # 可填固定張數如 "500" 或 "700,700,600"；留空則依 SPLIT_RATIOS 比例均分
SPLIT_RATIOS="1:1:1"
SPLIT_ORDER="stratified"
RECOMPUTE_PACK_THRESH=""
EXCLUDE_GOLDEN="--exclude_golden"      # 若拆分時想排除 4 張 Golden 圖，設為 "--exclude_golden"
USE_SYMLINK="--symlink"
# ===============================================================

# ---------------------------------------------------------------------------
# [環境與模組相容自動檢查]
# 確保 3_infer_fab_no_gt_old.py 頂部的 importlib.import_module("1_auto_golden") 能正常載入
# ---------------------------------------------------------------------------
ensure_old_modules_ready() {
  if [ ! -f "1_auto_golden.py" ]; then
    if [ ! -f "1_auto_golden_old.py" ]; then
      echo "❌ [檔案遺失] 找不到 1_auto_golden_old.py 或 1_auto_golden.py！"
      exit 1
    fi
    mkdir -p fab_tools
    if [ ! -f "fab_tools/auto_golden.py" ]; then
      cp -f "1_auto_golden_old.py" "fab_tools/auto_golden.py"
    fi
    python - << 'EOF'
import shutil

shutil.copyfile("1_auto_golden_old.py", "1_auto_golden.py")
shim_code = """

# --- Auto-generated compatibility shim for 3_infer_fab_no_gt_old.py ---
try:
    _train_old = __import__("2_train_fab_no_gt_old")
except ImportError:
    _train_old = None
format_duration = getattr(_train_old, "format_duration", None)
log_and_print = getattr(_train_old, "log_and_print", None)
safe_rmtree = getattr(_train_old, "safe_rmtree", None)
safe_copy = getattr(_train_old, "safe_copy", None)
compute_pool_signature = getattr(_train_old, "compute_pool_signature", None)
load_image_rgb_uint8 = getattr(_train_old, "load_image_rgb_uint8", None)
get_axis_positions = getattr(_train_old, "get_axis_positions", None)
is_background_tile = getattr(_train_old, "is_background_tile", None)
prepare_tiled_golden_trainset = getattr(_train_old, "prepare_tiled_golden_trainset", None)
load_or_prepare_golden_dataset = None
"""
with open("1_auto_golden.py", "a", encoding="utf-8") as f:
    f.write(shim_code)
print(" [🔧 自動橋接] 已從 1_auto_golden_old.py 生成相容的 1_auto_golden.py 模組")
EOF
  fi
}

ensure_old_modules_ready

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/inside_infer_eval_old_summary_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_inside_infer_eval_old_summary.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_inside_infer_eval_old_summary.csv"

echo "==========================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 🏭 [FoundAD (Old Pipeline) 公司內部無 GT 盲池 — 純推論與拆分評估總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Pool_Key,Category,Total_Imgs,Golden_Imgs,Unseen_Test_Imgs,Flagged_Anomalies,Anomaly_Rate_Pct,Eff_Threshold,Core_Normal_Med,Core_Normal_Mean,Core_Normal_Std,Train_Dir,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

run_single_inside_infer_eval_old() {
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
  
  # 自動從 Dataset 路徑取得批號
  # 預期路徑：.../W642/TMQB28/chipping
  local BATCH_ID
  BATCH_ID=$(basename "$(dirname "$(dirname "$(cd "${POOL_DIR}" && pwd)")")")
  
  # 組合輸出名稱：批號_Layer_資料類型
  local UNIQUE_TAG="${BATCH_ID}_${DS_PREFIX}_${CAT_NAME}"

  local DATA_NAME="fab_inside_old_${UNIQUE_TAG}_${SHOTS}shot"
  local DIY_NAME="fab_inside_old_${UNIQUE_TAG}"
  local SAVE_BASE_DIR="${OUTPUT_ROOT}/inside_no_gt_old_${UNIQUE_TAG}"
  local ACTIVE_CKPT_STEP="${FAB_CKPT_STEP}"

  # 1. 若有指定 CUSTOM_CKPT_PATH (.pth.tar 檔或資料夾)，自動反推 DATA_NAME、DIY_NAME 與 STEP
  if [ -n "${CUSTOM_CKPT_PATH}" ]; then
    if [ ! -e "${CUSTOM_CKPT_PATH}" ]; then
      echo "❌ [權重路徑錯誤] 找不到指定的權重檔案或資料夾: ${CUSTOM_CKPT_PATH}"
      exit 1
    fi
    local RESOLVED_CKPT_INFO
    RESOLVED_CKPT_INFO=$(python - "${CUSTOM_CKPT_PATH}" "${ACTIVE_CKPT_STEP}" << 'EOF'
import os
import re
import sys

p = os.path.abspath(sys.argv[1])
req_step = sys.argv[2]

if os.path.isfile(p):
    ckpt_dir = os.path.dirname(p)
    fname = os.path.basename(p)
    m = re.search(r"train-step(\d+)\.pth\.tar", fname)
    if m:
        req_step = m.group(1)
else:
    ckpt_dir = p

# 尋找 logs/ 下第一層資料夾名稱作為 data_name，其下層資料夾作為 diy_name
parts = ckpt_dir.replace("\\", "/").split("/")
data_name = ckpt_dir
diy_name = ""
if "logs" in parts:
    idx = parts.index("logs")
    if idx + 1 < len(parts):
        data_name = parts[idx + 1]
    if idx + 2 < len(parts):
        diy_name = parts[idx + 2]

print(f"{data_name}|{diy_name}|{req_step}")
EOF
)
    IFS='|' read -r DATA_NAME DIY_NAME ACTIVE_CKPT_STEP <<< "${RESOLVED_CKPT_INFO}"
    echo " [🎯 指定權重載入] 路徑: ${CUSTOM_CKPT_PATH}"
    echo "    -> 自動解析 DATA_NAME=${DATA_NAME} | DIY_NAME=${DIY_NAME} | STEP=${ACTIVE_CKPT_STEP}"
  # 2. 若未填 CUSTOM_CKPT_PATH，但有填 CUSTOM_DATA_NAME，則沿用舊邏輯
  elif [ -n "${CUSTOM_DATA_NAME}" ]; then
    DATA_NAME="${CUSTOM_DATA_NAME}"
  # 3. 預設自動對齊當前盲池對應的權重
  elif [ ! -d "./logs/${DATA_NAME}" ] && [ -d "./logs/fab_inside_${UNIQUE_TAG}_${SHOTS}shot" ]; then
    DATA_NAME="fab_inside_${UNIQUE_TAG}_${SHOTS}shot"
    DIY_NAME="fab_inside_${UNIQUE_TAG}"
    echo " [🔄 權重自動對接] 偵測到既有訓練權重位於 ./logs/${DATA_NAME}，自動對接該權重與 Golden 基準！"
  fi

  if [ -n "${CUSTOM_DIY_NAME}" ]; then
    DIY_NAME="${CUSTOM_DIY_NAME}"
  fi

  echo ""
  echo "==========================================================================="
  echo " 🚀 [公司內部無 GT 純推論與評估啟動 (Old 版)] 盲池路徑: ${POOL_DIR}"
  echo "    CATEGORY : ${CAT_NAME} | UNIQUE_TAG: ${UNIQUE_TAG}"
  echo "    DATA_NAME: ${DATA_NAME} | DIY_NAME: ${DIY_NAME} | CKPT_STEP: ${ACTIVE_CKPT_STEP}"
  echo "    門檻設定 : THRESH_MODE=${THRESH_MODE} | THRESHOLD=${THRESHOLD}"
  echo "    獨立輸出目錄: ${SAVE_BASE_DIR}"
  echo "==========================================================================="

  # 防呆檢查：確認 ./logs/${DATA_NAME} 或直接傳入之路徑下已有訓練好的 train-step*.pth.tar 權重
  local CKPT_FOUND=""
  if [ -n "${CUSTOM_CKPT_PATH}" ] && [ -f "${CUSTOM_CKPT_PATH}" ]; then
    CKPT_FOUND="${CUSTOM_CKPT_PATH}"
  elif [ -d "./logs/${DATA_NAME}" ]; then
    CKPT_FOUND=$(find "./logs/${DATA_NAME}" -name "train-step*.pth.tar" 2>/dev/null | head -n 1)
  elif [ -d "${DATA_NAME}" ]; then
    CKPT_FOUND=$(find "${DATA_NAME}" -name "train-step*.pth.tar" 2>/dev/null | head -n 1)
  fi

  if [ -z "${CKPT_FOUND}" ]; then
    echo "❌ [缺少訓練權重錯誤] 在 ./logs/${DATA_NAME}/ 下找不到任何 train-step*.pth.tar 訓練權重檔！"
    echo "   ├── 請先執行 bash run_inside_old.sh 完成該盲池的初次訓練，"
    echo "   └── 或於本腳本設定 CUSTOM_CKPT_PATH 指定既有模型權重檔 (.pth.tar) 路徑。"
    exit 1
  fi

  # Step 1/2: 執行 3_infer_fab_no_gt_old.py (高解析度切塊推論、RAM 即時拼接銷毀碎片與四合一防誤報紅框定位)
  echo "---------------------------------------------------------------------------"
  echo " [Step 1/2] 執行 3_infer_fab_no_gt_old.py -> 高解析度切塊推論與四合一濾波紅框定位"
  echo "---------------------------------------------------------------------------"
  python 3_infer_fab_no_gt_old.py \
    --pool_dir "${POOL_DIR}" \
    --out_dir "${SAVE_BASE_DIR}" \
    --datasets_dir "${DATASETS_DIR}" \
    --data_name "${DATA_NAME}" \
    --category "${CAT_NAME}" \
    --app_test "${APP_TEST}" \
    --diy_name "${DIY_NAME}" \
    --ckpt_step "${ACTIVE_CKPT_STEP}" \
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
    --infer_batch_size "${INFER_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    ${FORCE_INFER} ${RAW_SCORE_MODE} ${DISABLE_TILING}

  # Step 2/2: 執行 4_split_eval_results.py (評估結果拆分為多包獨立 Eval 報表)
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
      echo " [Step 2/2] 執行 ${SPLIT_SCRIPT} -> 拆分為 ${NUM_PACKS} 份獨立 Eval 報表"
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

# 尋找訓練階段產生的時間戳記目錄 (內含 selected_*_golden_images.txt 但不含 escape_ranking_report.csv)
all_sub_dirs = sorted([d for d in glob.glob(os.path.join(save_base, "*")) if os.path.isdir(d)])
train_dirs = [
    d for d in all_sub_dirs
    if glob.glob(os.path.join(d, "selected_*_golden_images.txt"))
    and not os.path.exists(os.path.join(d, "escape_ranking_report.csv"))
]
latest_train_dir = train_dirs[-1] if train_dirs else save_base

latest_txt = latest_csv.replace(".csv", ".txt")
eff_thresh = 0.25
overfit_line = ""
score_mode_str = "FILTERED_ROBUST"
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
passed_normals = [r for r in non_golden if "ANOMALY" not in str(r["Status"])]

target_normal = passed_normals if passed_normals else (non_golden if non_golden else records)
norm_scores = [float(r["Max_Score"]) for r in target_normal]

max_s = float(np.max(norm_scores))
p99_s = float(np.percentile(norm_scores, 99))
p95_s = float(np.percentile(norm_scores, 95))
median_s = float(np.median(norm_scores))
mean_s = float(np.mean(norm_scores))
std_s = float(np.std(norm_scores))
_all_eval_scores = np.array([float(r["Max_Score"]) for r in (non_golden if non_golden else records)], dtype=np.float64)
_mean_all = float(np.mean(_all_eval_scores))
_std_all = float(np.std(_all_eval_scores))
_th_3sigma = _mean_all + 3.0 * _std_all
_med_all = float(np.median(_all_eval_scores))
_mad_all = float(np.median(np.abs(_all_eval_scores - _med_all)))
_th_modz = _med_all + (3.5 / 0.6745) * _mad_all
if _th_3sigma <= _th_modz:
    _low_th, _low_name = _th_3sigma, "3-Sigma"
    _high_th, _high_name = _th_modz, "Mod-Z(3.5)"
else:
    _low_th, _low_name = _th_modz, "Mod-Z(3.5)"
    _high_th, _high_name = _th_3sigma, "3-Sigma"
_n_all = int(_all_eval_scores.size)
_low_cnt = int(np.sum(_all_eval_scores >= _low_th))
_high_cnt = int(np.sum(_all_eval_scores >= _high_th))
_high_pct = (_high_cnt / max(1, _n_all)) * 100.0
_high_p = float(np.mean(_all_eval_scores < _high_th) * 100.0)
_high_cut_str = (
    f"高於最高分({float(np.max(_all_eval_scores)):.4f})，切出 0 張"
    if _high_cnt == 0
    else f"切在 Rank {_high_cnt}/{_n_all} 之後 (共 {_high_cnt} 張, {_high_pct:.2f}%, 約 P{_high_p:.2f})"
)
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

anom_rate_pct = (len(flagged_anomalies) / len(non_golden) * 100.0) if non_golden else 0.0

summary_block = [
    "=" * 90,
    f" 🏭 [{unique_tag.upper()} 公司內部無 GT 盲池純推論總結 (Old 版) — 分數分佈與門檻參考 ({score_mode_str})]",
    "-" * 90,
    f" 📂 對應訓練紀錄資料夾     : {os.path.abspath(latest_train_dir)}",
    f" 📂 本次推論紀錄資料夾     : {os.path.abspath(infer_dir)}",
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
    f" 📐 盲池正常核心門檻參考值 :  3-Sigma(Shewhart 1931)={_th_3sigma:.4f} | Mod-Z 3.5(Iglewicz 1993)={_th_modz:.4f}",
    f"  🎯 兩者較小值參考         : [{_low_name}] = {_low_th:.4f} (切出 {_low_cnt}/{_n_all} 張, {_low_cnt/max(1,_n_all)*100.0:.2f}%)",
    f"  📍 兩者較高值切點         : [{_high_name}] = {_high_th:.4f} -> {_high_cut_str}",
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
  run_single_inside_infer_eval_old "${pool_path}"
done

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================="
echo " ✅ 公司內部全部 ${#TARGET_POOL_DIRS[@]} 個無 GT 測試盲池【純推論 + 結果拆分 (Old 版)】執行完畢！"
echo " 📄 本次完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo " 📄 最新完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo " 📊 最新完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo "=========================================================================================================="
