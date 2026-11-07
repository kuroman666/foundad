#!/usr/bin/env bash
# 檔案位置: FoundAD/run_outside.sh (全選覆蓋並存檔)
# 用途: 廠外公開資料集 (MVTec AD / MVTec AD 2) 四階段無 GT 模擬抓漏與拆分評估一鍵測試腳本
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto"                 # 廠外自動從 ~/.cache/torch/hub 或 ./hub_cache 搜尋權重

if [ $# -ge 1 ]; then
    TARGET_CAT_DIRS=("$@")
else
    TARGET_CAT_DIRS=(
        "${DATASETS_DIR}/mvtec/capsule"
        "${DATASETS_DIR}/mvtec-2/can"
        "${DATASETS_DIR}/mvtec/bottle"
        "${DATASETS_DIR}/mvtec/cable"
        "${DATASETS_DIR}/mvtec/grid"
        "${DATASETS_DIR}/mvtec/tile"
        "${DATASETS_DIR}/mvtec/transistor"
    )
fi

POOL_MODE="escape"
DEFECT_LIMIT=8
REBUILD_POOL=""                       # 若想強制重建模擬盲池與重挑 Golden，設為 "--rebuild_golden"
FORCE_TRAIN=""                        # 若想強制重跑模型訓練，設為 "--force_train"

# --- 四階段執行控制開關 (已有自動防呆：有 Golden 不重挑、有權重不重訓) ---
SKIP_GOLDEN=0
SKIP_FAB_TRAIN=""
FORCE_INFER=""
RUN_SPLIT_EVAL=1

# --- 模型、動態門檻與四合一防誤報超參數 ---
SHOTS=4
FAB_CKPT_STEP=2000
APP_TRAIN="train_dinov3"
APP_TEST="test"

THRESH_MODE="hybrid"
SIGMA_K=5
THRESHOLD=0
MIN_THRESHOLD=0
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE=""

WAFER_EDGE_MODE="auto"
WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012
MICRO_DEFECT_MODE="--micro_defect_mode"  # 一鍵啟動微瑕疵高靈敏模式

# 並在 Step 3/4 執行 python 3_infer_fab_no_gt.py 的最後一行加上 ${MICRO_DEFECT_MODE}：
        ${FORCE_INFER} ${RAW_SCORE_MODE} ${MICRO_DEFECT_MODE} ${DISABLE_TILING}

# --- 高解析度切塊與縫合參數 ---
TILE_SIZE=512
TILE_OVERLAP=64
MAX_TRAIN_TILES=48
TRAIN_BATCH_SIZE=8
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""

# --- Step 4 評估結果拆分參數 ---
NUM_PACKS=3
SPLIT_COUNTS=""
SPLIT_RATIOS="1:1:1"
SPLIT_ORDER="stratified"
EXCLUDE_GOLDEN=""
USE_SYMLINK="--symlink"
# ===============================================================

prepare_public_sim_pool() {
    local SRC_CAT_DIR="$1"
    local DST_POOL_DIR="$2"
    local MODE="$3"
    local LIMIT="$4"
    local FORCE_REBUILD="$5"

    python - "${SRC_CAT_DIR}" "${DST_POOL_DIR}" "${MODE}" "${LIMIT}" "${FORCE_REBUILD}" << 'EOF'
import os
import sys
import stat
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
src_cat_dir, dst_pool_dir, mode, defect_limit, force_rebuild = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    sys.argv[5],
)

if os.path.isdir(dst_pool_dir) and not force_rebuild:
    existing = [f for f in os.listdir(dst_pool_dir) if f.lower().endswith(IMG_EXTS)]
    if existing:
        print(f"[⚡ 模擬無GT盲池快取命中] 直接沿用既有 {len(existing)} 張盲池圖片 -> {os.path.abspath(dst_pool_dir)}")
        sys.exit(0)

test_dir = None
for cand in ["test", "test_public", "val", "validation"]:
    cand_path = os.path.join(src_cat_dir, cand)
    if os.path.isdir(cand_path) and os.path.isdir(os.path.join(cand_path, "good")):
        test_dir = cand_path
        break

if test_dir is None:
    raise FileNotFoundError(
        f"在 {src_cat_dir} 下找不到包含 good/ 的測試資料夾 (已檢查 test, test_public, validation)！"
    )

if os.path.exists(dst_pool_dir):
    shutil.rmtree(dst_pool_dir)
os.makedirs(dst_pool_dir, exist_ok=True)

def copy_unlocked(src, dst):
    shutil.copyfile(src, dst)
    try:
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass

good_dir = os.path.join(test_dir, "good")
good_count = 0
for root, _, files in os.walk(good_dir):
    for f in sorted(files):
        if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
            copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"clean_{f}"))
            good_count += 1

if good_count < 8:
    for backup_sub in ["validation/good", "train/good"]:
        b_dir = os.path.join(src_cat_dir, backup_sub)
        if os.path.isdir(b_dir):
            for f in sorted(os.listdir(b_dir)):
                if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
                    dst_f = os.path.join(dst_pool_dir, f"clean_val_{f}")
                    if not os.path.exists(dst_f):
                        copy_unlocked(os.path.join(b_dir, f), dst_f)
                        good_count += 1
                        if good_count >= 25:
                            break
        if good_count >= 25:
            break

defect_count = 0
if mode == "escape" and defect_limit > 0:
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good":
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for root, _, files in os.walk(sub_dir):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS) and not f.startswith(".") and defect_count < defect_limit:
                    copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"escape_{sub}_{f}"))
                    defect_count += 1

print(
    f"[✓ 模擬無GT盲池建立完成] 來源: {test_dir} | 模式: {mode} | "
    f"共 {good_count + defect_count} 張圖 (正常: {good_count} 張, 隱藏瑕疵: {defect_count} 張) -> {os.path.abspath(dst_pool_dir)}"
)
EOF
}

run_single_public_category() {
    local CAT_DIR="$1"
    if [ ! -d "${CAT_DIR}" ]; then
        echo "❌ [路徑錯誤] 找不到類別資料夾: ${CAT_DIR}"
        exit 1
    fi

    local CAT_NAME
    CAT_NAME=$(basename "${CAT_DIR}")
    local PARENT_DATASET
    PARENT_DATASET=$(basename "$(dirname "${CAT_DIR}")")
    local DS_PREFIX="${PARENT_DATASET//-/_}"

    local SIM_POOL_DIR=""
    local DATA_NAME=""
    local DIY_NAME=""
    local SAVE_BASE_DIR=""

    if [ "${POOL_MODE}" = "pure_normal" ]; then
        SIM_POOL_DIR="${DATASETS_DIR}/sim_pure_normal_${CAT_NAME}_pool"
        DATA_NAME="${DS_PREFIX}_pure_${CAT_NAME}_${SHOTS}shot"
        DIY_NAME="pure_fpr_${CAT_NAME}"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/3_pure_normal_${CAT_NAME}"
        # 向下相容舊版命名 (pure_fpr)
        if [ ! -d "./logs/${DATA_NAME}" ] && [ -d "./logs/mvtec_pure_${CAT_NAME}_${SHOTS}shot/dinov3pure_fpr" ]; then
            DATA_NAME="mvtec_pure_${CAT_NAME}_${SHOTS}shot"
            DIY_NAME="pure_fpr"
        fi
    else
        SIM_POOL_DIR="${DATASETS_DIR}/sim_fab_${CAT_NAME}_pool"
        DATA_NAME="${DS_PREFIX}_sim_${CAT_NAME}_${SHOTS}shot"
        DIY_NAME="sim_escape_${CAT_NAME}"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/2_fab_no_gt_${CAT_NAME}"
        # 向下相容舊版命名：若舊版 mvtec_sim_4shot 已存有該類別的 Golden 圖塊或權重，優先沿用舊版命名！
        if [ ! -d "${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME}" ] && [ -d "${DATASETS_DIR}/mvtec_sim_${SHOTS}shot/train/${CAT_NAME}" ]; then
            echo " [🔄 命名相容防護] 偵測到舊版 Golden 目錄 mvtec_sim_${SHOTS}shot/train/${CAT_NAME}，自動沿用避免重挑！"
            DATA_NAME="mvtec_sim_${SHOTS}shot"
            if [ -d "./logs/mvtec_sim_${SHOTS}shot/dinov3sim_escape" ]; then
                DIY_NAME="sim_escape"
            fi
        fi
    fi

    echo ""
    echo "==========================================================================="
    echo " 🚀 [廠外公開資料集無 GT 測試啟動] 資料集: ${PARENT_DATASET} | 類別: ${CAT_NAME}"
    echo "    DATA_NAME: ${DATA_NAME} | DIY_NAME: ${DIY_NAME}"
    echo "    模式: ${POOL_MODE} | 門檻: ${THRESH_MODE} (SIGMA_K=${SIGMA_K}) | 切塊: ${TILE_SIZE}"
    echo "==========================================================================="

    # Step 0: 建立或沿用模擬盲池
    prepare_public_sim_pool "${CAT_DIR}" "${SIM_POOL_DIR}" "${POOL_MODE}" "${DEFECT_LIMIT}" "${REBUILD_POOL}"

    # Step 1: 執行 1_auto_golden.py (內建防呆：只要 train/<CAT_NAME> 已有圖片就自動略過不重挑)
    local EXISTING_GOLDEN_COUNT=0
    if [ -d "${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME}" ]; then
        EXISTING_GOLDEN_COUNT=$(find "${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \) | wc -l)
    fi

    if [ "${SKIP_GOLDEN}" -eq 0 ] && [ -z "${REBUILD_POOL}" ] && [ "${EXISTING_GOLDEN_COUNT}" -gt 0 ]; then
        echo " [⚡ Step 1/4 防呆略過] ${DATASETS_DIR}/${DATA_NAME}/train/${CAT_NAME} 已存在 ${EXISTING_GOLDEN_COUNT} 張 Golden 圖塊，直接沿用不重挑！"
    elif [ "${SKIP_GOLDEN}" -eq 0 ] || [ -n "${REBUILD_POOL}" ]; then
        echo "---------------------------------------------------------------------------"
        echo " [Step 1/4] 執行 1_auto_golden.py -> 自動挑選並純化 ${CAT_NAME} Golden 訓練集"
        echo "---------------------------------------------------------------------------"
        python 1_auto_golden.py \
            --pool_dir "${SIM_POOL_DIR}" \
            --ckpt "${DINO_CKPT_PATH}" \
            --datasets_dir "${DATASETS_DIR}" \
            --data_name "${DATA_NAME}" \
            --category "${CAT_NAME}" \
            --shots "${SHOTS}" \
            --tile_size "${TILE_SIZE}" \
            --tile_overlap "${TILE_OVERLAP}" \
            --max_train_tiles "${MAX_TRAIN_TILES}" \
            --bg_std_thresh "${BG_STD_THRESH}" \
            --report_dir "${SAVE_BASE_DIR}/golden_reports" \
            ${REBUILD_POOL} ${DISABLE_TILING}
    fi

    # Step 2: 執行 2_train_fab_no_gt.py (內建防呆：若 ./logs/<DATA_NAME> 下已有權重檔，自動略過訓練)
    echo "---------------------------------------------------------------------------"
    echo " [Step 2/4] 執行 2_train_fab_no_gt.py -> 模型訓練與權重對接檢查"
    echo "---------------------------------------------------------------------------"
    python 2_train_fab_no_gt.py \
        --pool_dir "${SIM_POOL_DIR}" \
        --ckpt "${DINO_CKPT_PATH}" \
        --out_dir "${SAVE_BASE_DIR}" \
        --datasets_dir "${DATASETS_DIR}" \
        --data_name "${DATA_NAME}" \
        --category "${CAT_NAME}" \
        --app_train "${APP_TRAIN}" \
        --diy_name "${DIY_NAME}" \
        --ckpt_step "${FAB_CKPT_STEP}" \
        --shots "${SHOTS}" \
        --tile_size "${TILE_SIZE}" \
        --tile_overlap "${TILE_OVERLAP}" \
        --max_train_tiles "${MAX_TRAIN_TILES}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --bg_std_thresh "${BG_STD_THRESH}" \
        ${SKIP_FAB_TRAIN} ${FORCE_TRAIN} ${REBUILD_POOL} ${DISABLE_TILING}

    # Step 3: 執行 3_infer_fab_no_gt.py
    echo "---------------------------------------------------------------------------"
    echo " [Step 3/4] 執行 3_infer_fab_no_gt.py -> 高解析度切塊推論與防誤報紅框定位"
    echo "---------------------------------------------------------------------------"
    python 3_infer_fab_no_gt.py \
        --pool_dir "${SIM_POOL_DIR}" \
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
        --tile_size "${TILE_SIZE}" \
        --tile_overlap "${TILE_OVERLAP}" \
        --max_train_tiles "${MAX_TRAIN_TILES}" \
        --infer_batch_size "${INFER_BATCH_SIZE}" \
        --bg_std_thresh "${BG_STD_THRESH}" \
        ${FORCE_INFER} ${RAW_SCORE_MODE} ${MICRO_DEFECT_MODE} ${DISABLE_TILING}

    # Step 4: 執行評估結果拆分 (自動相容 4_split_eval_results.py 或 3_split_eval_results.py)
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
                SPLIT_ARGS+=(--ratios "${SPLIT_RATIOS}")
            fi

            python "${SPLIT_SCRIPT}" \
                --src_dir "${SAVE_BASE_DIR}" \
                --num_packs "${NUM_PACKS}" \
                --split_order "${SPLIT_ORDER}" \
                "${SPLIT_ARGS[@]}" \
                ${EXCLUDE_GOLDEN} ${USE_SYMLINK}
        fi
    fi

    # 終端機列印該類別抓漏率與誤報率統計總結
    python - "${SAVE_BASE_DIR}" "${CAT_NAME}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base = sys.argv[1]
cat_name = sys.argv[2]

all_csvs = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
csv_files = [p for p in all_csvs if not os.path.basename(os.path.dirname(p)).startswith("eval_pack_")]
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
non_golden_normal = [
    r for r in non_golden if not os.path.basename(r["Original_Filename"]).startswith("escape_")
]
escape_defects = [
    r for r in records if os.path.basename(r["Original_Filename"]).startswith("escape_")
]

target_normal = non_golden_normal if non_golden_normal else (non_golden if non_golden else records)
non_g_scores = [float(r["Max_Score"]) for r in target_normal]

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
fpr_pct = (len(fp_normal) / len(target_normal) * 100.0) if target_normal else 0.0
recall_pct = (len(caught_escape) / len(escape_defects) * 100.0) if escape_defects else 0.0

print("\n" + "=" * 85)
print(f" 🎯 [{cat_name.upper()} 廠外公開資料集測試總結 — 分數分佈與 Threshold 參考 ({score_mode_str})]")
print("-" * 85)
print(f"  📂 最新報表位置               : {latest_csv}")
print(f"  🖼️ 盲池圖片結構               : 總計 {total} 張 (Golden: {total - len(non_golden)} 張 | 非Golden正常圖: {len(target_normal)} 張 | 隱藏Escape瑕疵: {len(escape_defects)} 張)")
print(f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}")
if escape_defects:
    print(f"  🎯 抓漏與誤報表現             : 成功抓出瑕疵 {len(caught_escape)}/{len(escape_defects)} 張 (Recall={recall_pct:.1f}%) | 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (FPR={fpr_pct:.1f}%)")
else:
    print(f"  🎯 純正常池誤報表現 (FPR)     : 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (扣除Golden後 FPR={fpr_pct:.2f}%)")
if overfit_line:
    print(f"  🩺 模型過擬合自檢 (Overfit)    : {overfit_line}")
print("-" * 85)
print(f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}")
print(f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}")
if "Raw_Max_Score" in target_normal[0] and "Filtered_Max_Score" in target_normal[0]:
    raw_arr = [float(r["Raw_Max_Score"]) for r in target_normal]
    flt_arr = [float(r["Filtered_Max_Score"]) for r in target_normal]
    print(f"  🔬 濾波前後正常圖對照 (Ablation): [未濾波 Raw] Max={float(np.max(raw_arr)):.4f}, Mean={float(np.mean(raw_arr)):.4f}  vs  [四合一濾波] Max={float(np.max(flt_arr)):.4f}, Mean={float(np.mean(flt_arr)):.4f}")
if escape_defects:
    esc_scores = [float(r["Max_Score"]) for r in escape_defects]
    print(f"  🔍 隱藏 Escape 瑕疵圖分數分佈  : Min={float(np.min(esc_scores)):.4f} | Median={float(np.median(esc_scores)):.4f} | Max={float(np.max(esc_scores)):.4f}")
print("=" * 85 + "\n")
EOF
}

for cat_path in "${TARGET_CAT_DIRS[@]}"; do
    run_single_public_category "${cat_path}"
done