#!/usr/bin/env bash
# 檔案位置: FoundAD/run_outside_new_datasets.sh
# 用途: FoundAD 廠外新資料集 (YDFID-2 / Mura / mini_AVMap2) 及舊資料集 (MVTec / MVTec-2) 四階段無 GT 模擬抓漏與拆分評估腳本
# 相容腳本: 1_auto_golden_old.py / 2_train_fab_no_gt_old.py / 3_infer_fab_no_gt_old.py / 4_split_eval_results.py
#
# 使用範例:
#   1) 跑預設清單 (預設 YDFID-2/CL1):        bash run_outside_new_datasets.sh
#   2) 一次跑完 YDFID-2 全部 19 個類別:      bash run_outside_new_datasets.sh ./data/YDFID-2
#   3) 跑 Mura 與 AVMap 資料集:              bash run_outside_new_datasets.sh ./data/mini_AVMap2 ./data/Mura2_wo_bg_w_pb
#   4) 跑舊資料集 (完全相容):                bash run_outside_new_datasets.sh ./data/mvtec/capsule ./data/mvtec-2/can
# 背景執行全部 YDFID-2（19 個類別）： nohup ./run_outside_new_datasets.sh ./data/YDFID-2 > foundad_YDFID2.log 2>&1 &
#                                 tail -f foundad_YDFID2.log
# 執行 Mura / mini_AVMap2： nohup ./run_outside_new_datasets.sh ./data/mini_AVMap2 ./data/Mura2_wo_bg_w_pb ./data/Mura2_bg_mask > foundad_mura.log 2>&1 &
#                          tail -f foundad_mura.log
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./data"
DINO_CKPT_PATH="auto"                 # 自動從 ~/.cache/torch/hub 或 ./hub_cache 搜尋 DINOv3 / DINOv2 權重
ENABLE_MEM_MONITOR=1                  # 1=自動背景每秒記錄 RAM & GPU VRAM 至 log；0=關閉

if [ $# -ge 1 ]; then
    TARGET_CAT_DIRS=("$@")
else
    TARGET_CAT_DIRS=(
        # --- 新資料集: YDFID-2 (填入父目錄可自動展開全部 19 類，或指定單一類別) ---
        "${DATASETS_DIR}/YDFID-2/CL1"
        # "${DATASETS_DIR}/YDFID-2"

        # --- 新資料集: Mura / mini_AVMap2 ---
        # "${DATASETS_DIR}/mini_AVMap2"
        # "${DATASETS_DIR}/Mura2_wo_bg_w_pb"
        # "${DATASETS_DIR}/Mura2_bg_mask"

        # --- 舊資料集範例 (完全相容) ---
        # "${DATASETS_DIR}/mvtec/capsule"
        # "${DATASETS_DIR}/mvtec-2/can"
    )
fi

POOL_MODE="escape"
DEFECT_LIMIT=8
REBUILD_POOL=""                       # 若想強制重建模擬盲池與重挑 Golden，設為 "--rebuild_golden"
FORCE_TRAIN=""                        # 若想強制重跑模型訓練，設為 "--force_train"

# --- 四階段執行控制開關 (已有自動防呆：有 Golden 不重挑、有權重不重訓) ---
SKIP_GOLDEN=0
SKIP_FAB_TRAIN=""                     # 若想強制跳過訓練，可設為 "--skip_train"
FORCE_INFER=""                        # 若想強制重跑推論，可設為 "--force_infer"
RUN_SPLIT_EVAL=1

# --- 模型、動態門檻與四合一防誤報超參數 ---
SHOTS=4
FAB_CKPT_STEP=2000
APP_TRAIN="train_dinov3"
APP_TEST="test"

THRESH_MODE="hybrid"
SIGMA_K=3.5
THRESHOLD=0.25
MIN_THRESHOLD=0.27
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE=""                     # 若想使用未濾波原生分數，設為 "--raw_score_mode"

WAFER_EDGE_MODE="auto"
WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012

# --- 高解析度切塊與縫合參數 ---
TILE_SIZE=512
TILE_OVERLAP=64
MAX_TRAIN_TILES=48
TRAIN_BATCH_SIZE=8
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""                     # 若想關閉切塊，設為 "--disable_tiling"

# --- Step 4 評估結果拆分參數 ---
NUM_PACKS=3
SPLIT_COUNTS=""
SPLIT_RATIOS="1:1:1"
SPLIT_ORDER="stratified"
RECOMPUTE_PACK_THRESH=""
EXCLUDE_GOLDEN="--exclude_golden"
USE_SYMLINK="--symlink"
# ===============================================================

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/multi_dataset_batch_summary_foundad_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_multi_dataset_summary_foundad.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_multi_dataset_summary_foundad.csv"
MEM_MONITOR_LOG="${OUTPUT_ROOT}/mem_monitor_foundad_${BATCH_TIMESTAMP}.log"
MEM_MONITOR_LATEST="${OUTPUT_ROOT}/latest_mem_monitor_foundad.log"
MEM_MONITOR_PID=""

start_mem_monitor() {
    if [ "${ENABLE_MEM_MONITOR}" -ne 1 ]; then
        return
    fi
    echo "=== FoundAD System RAM & GPU VRAM Monitor Started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "${MEM_MONITOR_LOG}"
    (
        while true; do
            TS=$(date '+%Y-%m-%d %H:%M:%S')
            RAM_INFO=$(free -m | awk '/^Mem:/ {printf "RAM_Used=%d/%d MiB (Avail=%d MiB)", $3, $2, $7} /^Swap:/ {printf " | Swap=%d/%d MiB", $3, $2}')
            if command -v nvidia-smi >/dev/null 2>&1; then
                GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 | awk -F", " '{printf "GPU_Util=%s%% | VRAM=%s/%s MiB (%.1f%%)", $1, $2, $3, ($2/$3)*100}')
            else
                GPU_INFO="GPU=N/A"
            fi
            echo "[${TS}] ${RAM_INFO} | ${GPU_INFO}" >> "${MEM_MONITOR_LOG}"
            sleep 1
        done
    ) &
    MEM_MONITOR_PID=$!
    echo "📊 [系統資源監控已背景啟動] PID=${MEM_MONITOR_PID} -> 即時紀錄檔: ${MEM_MONITOR_LOG}"
}

cleanup_mem_monitor() {
    if [ -n "${MEM_MONITOR_PID}" ] && kill -0 "${MEM_MONITOR_PID}" 2>/dev/null; then
        kill "${MEM_MONITOR_PID}" 2>/dev/null || true
        wait "${MEM_MONITOR_PID}" 2>/dev/null || true
    fi
    if [ -f "${MEM_MONITOR_LOG}" ]; then
        cp -f "${MEM_MONITOR_LOG}" "${MEM_MONITOR_LATEST}"
        python - "${MEM_MONITOR_LOG}" << 'PYEOF'
import re, sys
log_p = sys.argv[1]
ram_used, ram_avail, vram_used = [], [], []
vram_tot = 0
with open(log_p, "r", encoding="utf-8") as f:
    for line in f:
        m_r = re.search(r"RAM_Used=(\d+)/(\d+)\s*MiB\s*\(Avail=(\d+)\s*MiB\)", line)
        if m_r:
            ram_used.append(int(m_r.group(1)))
            ram_avail.append(int(m_r.group(3)))
        m_v = re.search(r"VRAM=(\d+)/(\d+)\s*MiB", line)
        if m_v:
            vram_used.append(int(m_v.group(1)))
            vram_tot = int(m_v.group(2))
if ram_used or vram_used:
    print("\n📈 [整趟執行資源峰值統計 (RAM & GPU VRAM Summary)]")
    if ram_used:
        print(f"   ├─ 系統 RAM 最高已用 : {max(ram_used)} MiB ({max(ram_used)/1024:.2f} GiB) | 最低剩餘可用 (Min Avail): {min(ram_avail)} MiB ({min(ram_avail)/1024:.2f} GiB)")
    if vram_used:
        print(f"   └─ GPU VRAM 最高峰值 : {max(vram_used)} / {vram_tot} MiB ({max(vram_used)/1024:.2f} GiB, {max(vram_used)/max(1,vram_tot)*100:.1f}%)")
PYEOF
    fi
}

trap cleanup_mem_monitor EXIT INT TERM
start_mem_monitor

echo "==========================================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 📊 [FoundAD (Few-Shot) 多資料集批次訓練與推論總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Dataset_Key,Category,Total_Imgs,Golden_Imgs,Normal_Imgs,Escape_Imgs,Eff_Threshold,I_AUROC,I_AP,F1_Max,Recall_Pct,FPR_Pct,Precision_Pct,F1_Cur_Pct,Normal_Mean,Normal_Std,Train_Dir,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

prepare_public_sim_pool() {
    local SRC_CAT_DIR="$1"
    local DST_POOL_DIR="$2"
    local MODE="$3"
    local LIMIT="$4"
    local FORCE_REBUILD="$5"

    python - "${SRC_CAT_DIR}" "${DST_POOL_DIR}" "${MODE}" "${LIMIT}" "${FORCE_REBUILD}" << 'PYEOF'
import os
import sys
import stat
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
GOOD_FOLDER_NAMES = {"good", "defect-free", "defect_free", "ok", "normal"}
IGNORE_DIRS = {"masks", "masks2", "mask", "ground_truth", "groundtruth", "gt", "with_gt", "__macosx"}

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

def find_good_subdir(parent_dir):
    if not os.path.isdir(parent_dir):
        return None
    for d in sorted(os.listdir(parent_dir)):
        if d.lower() in GOOD_FOLDER_NAMES and os.path.isdir(os.path.join(parent_dir, d)):
            return os.path.join(parent_dir, d)
    return None

test_dir = None
good_dir = None
for cand in ["test", "test_public", "val", "validation"]:
    cand_path = os.path.join(src_cat_dir, cand)
    if not os.path.isdir(cand_path):
        continue
    g = find_good_subdir(cand_path)
    if g is not None:
        test_dir = cand_path
        good_dir = g
        break
    sub_entries = [
        os.path.join(cand_path, d)
        for d in sorted(os.listdir(cand_path))
        if os.path.isdir(os.path.join(cand_path, d)) and d.lower() not in IGNORE_DIRS
    ]
    for sub_p in sub_entries:
        g2 = find_good_subdir(sub_p)
        if g2 is not None:
            test_dir = sub_p
            good_dir = g2
            break
    if test_dir is not None:
        break

if test_dir is None or good_dir is None:
    raise FileNotFoundError(
        f"在 {src_cat_dir} 下找不到包含 good/ 或 defect-free/ 的測試資料夾！"
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

good_count = 0
for root, dirs, files in os.walk(good_dir):
    dirs[:] = [d for d in sorted(dirs) if d.lower() not in IGNORE_DIRS]
    for f in sorted(files):
        if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
            copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"clean_{f}"))
            good_count += 1

if good_count < 8:
    rel_sub = os.path.relpath(test_dir, src_cat_dir)
    sub_cat_name = os.path.basename(test_dir) if os.path.dirname(rel_sub) else ""
    backup_parents = ["validation", "train"]
    if sub_cat_name:
        backup_parents = [f"validation/{sub_cat_name}", f"train/{sub_cat_name}"] + backup_parents
    for bp in backup_parents:
        b_dir = find_good_subdir(os.path.join(src_cat_dir, bp))
        if b_dir and os.path.isdir(b_dir):
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
        if sub.lower() in GOOD_FOLDER_NAMES or sub.lower() in IGNORE_DIRS:
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for root, dirs, files in os.walk(sub_dir):
            dirs[:] = [d for d in sorted(dirs) if d.lower() not in IGNORE_DIRS]
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS) and not f.startswith(".") and defect_count < defect_limit:
                    copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"escape_{sub}_{f}"))
                    defect_count += 1

print(
    f"[✓ 模擬無GT盲池建立完成] 來源: {test_dir} | 模式: {mode} | "
    f"共 {good_count + defect_count} 張圖 (正常: {good_count} 張, 隱藏瑕疵: {defect_count} 張) -> {os.path.abspath(dst_pool_dir)}"
)
PYEOF
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
    local UNIQUE_TAG="${DS_PREFIX}_${CAT_NAME}"

    local SIM_POOL_DIR=""
    local DATA_NAME=""
    local DIY_NAME=""
    local SAVE_BASE_DIR=""

    if [ "${POOL_MODE}" = "pure_normal" ]; then
        SIM_POOL_DIR="${DATASETS_DIR}/sim_pure_normal_${UNIQUE_TAG}_pool"
        DATA_NAME="${DS_PREFIX}_pure_${CAT_NAME}_${SHOTS}shot"
        DIY_NAME="pure_fpr_${UNIQUE_TAG}"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/3_pure_normal_${UNIQUE_TAG}"
    else
        SIM_POOL_DIR="${DATASETS_DIR}/sim_fab_${UNIQUE_TAG}_pool"
        DATA_NAME="${DS_PREFIX}_sim_${CAT_NAME}_${SHOTS}shot"
        DIY_NAME="sim_escape_${UNIQUE_TAG}"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/2_fab_no_gt_${UNIQUE_TAG}"
    fi

    echo ""
    echo "==========================================================================="
    echo " 🚀 [廠外公開資料集無 GT 測試啟動 (FoundAD)] 資料集: ${PARENT_DATASET} | 類別: ${CAT_NAME}"
    echo "    DATA_NAME: ${DATA_NAME} | DIY_NAME: ${DIY_NAME}"
    echo "    獨立輸出目錄: ${SAVE_BASE_DIR}"
    echo "==========================================================================="

    # Step 0: 建立或沿用模擬盲池
    prepare_public_sim_pool "${CAT_DIR}" "${SIM_POOL_DIR}" "${POOL_MODE}" "${DEFECT_LIMIT}" "${REBUILD_POOL}"

    # 自動偵測 blind pool 解析度：若圖片尺寸 >= 1536px，自動放大 tile_size 為 1024 防止圖塊過多
    local EFF_TILE_SIZE="${TILE_SIZE}"
    local EFF_TILE_OVERLAP="${TILE_OVERLAP}"
    local RES_DETECT
    RES_DETECT=$(python - "${SIM_POOL_DIR}" "${TILE_SIZE}" "${TILE_OVERLAP}" << 'PYEOF'
import os, sys
from PIL import Image
pool_dir, ts, ov = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imgs = [os.path.join(pool_dir, f) for f in sorted(os.listdir(pool_dir)) if f.lower().endswith(exts)]
max_side = 0
for p in imgs[:10]:
    try:
        with Image.open(p) as im:
            max_side = max(max_side, max(im.size))
    except Exception:
        pass
if max_side >= 1536 and ts < 1024:
    print(f"1024 128 {max_side}")
else:
    print(f"{ts} {ov} {max_side}")
PYEOF
)
    EFF_TILE_SIZE=$(echo "${RES_DETECT}" | awk '{print $1}')
    EFF_TILE_OVERLAP=$(echo "${RES_DETECT}" | awk '{print $2}')
    local DETECTED_MAX_RES=$(echo "${RES_DETECT}" | awk '{print $3}')
    if [ "${EFF_TILE_SIZE}" != "${TILE_SIZE}" ]; then
        echo " [🛡️ 大圖防爆自動調節] 偵測到高解析度影像 (max=${DETECTED_MAX_RES}px)，自動將切塊尺寸調整為 ${EFF_TILE_SIZE}x${EFF_TILE_SIZE} (Overlap=${EFF_TILE_OVERLAP})"
    fi

    # Step 1: 執行 1_auto_golden_old.py
    local SHOT_DIR="${DATASETS_DIR}/${DATA_NAME}"
    local EXISTING_GOLDEN_COUNT=0
    if [ -d "${SHOT_DIR}/train/${CAT_NAME}" ]; then
        EXISTING_GOLDEN_COUNT=$(find "${SHOT_DIR}/train/${CAT_NAME}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \) | wc -l)
    fi

    if [ "${SKIP_GOLDEN}" -eq 0 ] && [ -z "${REBUILD_POOL}" ] && [ "${EXISTING_GOLDEN_COUNT}" -gt 0 ]; then
        echo " [⚡ Step 1/4 防呆略過] ${SHOT_DIR}/train/${CAT_NAME} 已存在 ${EXISTING_GOLDEN_COUNT} 張 Golden 圖塊，直接沿用不重挑！"
    elif [ "${SKIP_GOLDEN}" -eq 0 ] || [ -n "${REBUILD_POOL}" ]; then
        echo "---------------------------------------------------------------------------"
        echo " [Step 1/4] 執行 1_auto_golden_old.py -> 自動挑選 ${CAT_NAME} Golden 訓練集"
        echo "---------------------------------------------------------------------------"
        python - "${SIM_POOL_DIR}" "${DINO_CKPT_PATH}" "${SHOT_DIR}" "${CAT_NAME}" "${SHOTS}" "${SAVE_BASE_DIR}/golden_reports_${CAT_NAME}" << 'PYEOF'
import sys
import importlib

pool_dir, ckpt, out_dir, category, shots, report_dir = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    int(sys.argv[5]),
    sys.argv[6],
)
ag_old = importlib.import_module("1_auto_golden_old")
ag_old.select_golden_images(
    pool_dir=pool_dir,
    ckpt_path=ckpt,
    out_dir=out_dir,
    category=category,
    shots=shots,
    report_dir=report_dir,
)
PYEOF
    fi

    # Step 2: 執行 2_train_fab_no_gt_old.py
    local EFF_SKIP_TRAIN="${SKIP_FAB_TRAIN}"
    if [ -z "${FORCE_TRAIN}" ] && [ -z "${REBUILD_POOL}" ] && [ -z "${EFF_SKIP_TRAIN}" ]; then
        local EXISTING_CKPTS=0
        if [ -d "./logs/${DATA_NAME}" ]; then
            EXISTING_CKPTS=$(find "./logs/${DATA_NAME}" -type f -name "train-step*.pth.tar" 2>/dev/null | wc -l)
        fi
        if [ "${EXISTING_CKPTS}" -gt 0 ]; then
            echo " [⚡ Step 2/4 防呆略過] ./logs/${DATA_NAME} 已存在訓練權重檔，自動啟用 --skip_train！"
            EFF_SKIP_TRAIN="--skip_train"
        fi
    fi

    echo "---------------------------------------------------------------------------"
    echo " [Step 2/4] 執行 2_train_fab_no_gt_old.py -> 模型訓練與權重對接檢查"
    echo "---------------------------------------------------------------------------"
    python 2_train_fab_no_gt_old.py \
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
        --tile_size "${EFF_TILE_SIZE}" \
        --tile_overlap "${EFF_TILE_OVERLAP}" \
        --max_train_tiles "${MAX_TRAIN_TILES}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --bg_std_thresh "${BG_STD_THRESH}" \
        ${EFF_SKIP_TRAIN} ${DISABLE_TILING}

    # Step 3: 執行 3_infer_fab_no_gt_old.py
    echo "---------------------------------------------------------------------------"
    echo " [Step 3/4] 執行 3_infer_fab_no_gt_old.py -> 高解析度切塊推論與防誤報紅框定位"
    echo "---------------------------------------------------------------------------"
    python 3_infer_fab_no_gt_old.py \
        --pool_dir "${SIM_POOL_DIR}" \
        --out_dir "${SAVE_BASE_DIR}" \
        --datasets_dir "${DATASETS_DIR}" \
        --data_name "${DATA_NAME}" \
        --category "${CAT_NAME}" \
        --app_test "${APP_TEST}" \
        --diy_name "${DIY_NAME}" \
        --ckpt_step "${FAB_CKPT_STEP}" \
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
        --tile_size "${EFF_TILE_SIZE}" \
        --tile_overlap "${EFF_TILE_OVERLAP}" \
        --infer_batch_size "${INFER_BATCH_SIZE}" \
        --bg_std_thresh "${BG_STD_THRESH}" \
        ${FORCE_INFER} ${RAW_SCORE_MODE} ${DISABLE_TILING}

    # Step 4: 執行評估結果拆分 (4_split_eval_results.py)
    if [ "${RUN_SPLIT_EVAL}" -eq 1 ]; then
        local SPLIT_SCRIPT=""
        if [ -f "4_split_eval_results.py" ]; then
            SPLIT_SCRIPT="4_split_eval_results.py"
        elif [ -f "3_split_eval_results.py" ]; then
            SPLIT_SCRIPT="3_split_eval_results.py"
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

    # 終端機列印該類別統計總結（含 Recall / FPR / I-AUROC / I-AP / F1-max），並同步 Append 寫入多資料集總表
    python - "${SAVE_BASE_DIR}" "${CAT_NAME}" "${UNIQUE_TAG}" "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_CSV}" << 'PYEOF'
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
train_dirs = sorted(glob.glob(os.path.join(save_base, "*")))
train_dirs = [
    d for d in train_dirs
    if os.path.isdir(d) and d != infer_dir and not os.path.basename(d).startswith(("golden_reports", "split_eval_"))
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

non_golden = [r for r in records if not r["Status"].startswith("GOLDEN")]
eval_records = non_golden if non_golden else records
non_golden_normal = [
    r for r in eval_records if not os.path.basename(r["Original_Filename"]).startswith("escape_")
]
escape_defects = [
    r for r in eval_records if os.path.basename(r["Original_Filename"]).startswith("escape_")
]

target_normal = non_golden_normal if non_golden_normal else eval_records
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

auroc_val = ap_val = f1_max_val = best_th_val = 0.0
prec_pct = f1_cur_pct = 0.0

if escape_defects and non_golden_normal:
    y_true = np.array(
        [1 if os.path.basename(r["Original_Filename"]).startswith("escape_") else 0 for r in eval_records],
        dtype=np.int32,
    )
    y_score = np.array([float(r["Max_Score"]) for r in eval_records], dtype=np.float64)
    n_pos, n_neg = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))

    order = np.argsort(-y_score, kind="mergesort")
    yt_s, ys_s = y_true[order], y_score[order]
    ranks = np.zeros(len(ys_s), dtype=np.float64)
    i, n = 0, len(ys_s)
    while i < n:
        j = i + 1
        while j < n and ys_s[j] == ys_s[i]:
            j += 1
        ranks[i:j] = 0.5 * ((n - j + 1) + (n - i))
        i = j
    auroc_val = (float(np.sum(ranks[yt_s == 1])) - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))

    tp_c, fp_c = np.cumsum(yt_s == 1), np.cumsum(yt_s == 0)
    t_idx = np.r_[np.where(np.diff(ys_s))[0], n - 1]
    tps, fps = tp_c[t_idx], fp_c[t_idx]
    precs = tps / np.maximum(1, tps + fps)
    recs = tps / max(1, n_pos)
    ap_val = float(np.sum(np.diff(np.r_[0.0, recs]) * precs))
    f1s = (2.0 * precs * recs) / np.maximum(1e-12, precs + recs)
    b_i = int(np.argmax(f1s))
    f1_max_val = float(f1s[b_i])
    best_th_val = float(ys_s[t_idx[b_i]])

    tp_cur = len(caught_escape)
    fp_cur = len(fp_normal)
    prec_pct = (tp_cur / max(1, tp_cur + fp_cur)) * 100.0
    f1_cur_pct = (2.0 * prec_pct * recall_pct) / max(1e-12, prec_pct + recall_pct)

summary_block = [
    "=" * 95,
    f" 🎯 [{unique_tag.upper()} (FoundAD) 廠外公開資料集測試總結 — 分數分佈與量化評估指標 ({score_mode_str})]",
    "-" * 95,
    f"  📂 訓練紀錄資料夾             : {os.path.abspath(latest_train_dir)}",
    f"  📂 推論紀錄資料夾             : {os.path.abspath(infer_dir)}",
    f"  📄 最新排名報表位置           : {os.path.abspath(latest_csv)}",
    f"  🖼️ 盲池圖片結構               : 總計 {total} 張 (Golden: {total - len(non_golden)} 張 | 非Golden正常圖: {len(target_normal)} 張 | 隱藏Escape瑕疵: {len(escape_defects)} 張)",
    f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}",
]
if escape_defects:
    summary_block.extend([
        f"  🏆 標準量化評估指標 (Metrics) : I-AUROC={auroc_val:.4f} | I-AP={ap_val:.4f} | F1-max={f1_max_val:.4f} (Best_Th={best_th_val:.4f})",
        f"  🎯 抓漏與誤報表現 (Current)   : Recall={recall_pct:.2f}% ({len(caught_escape)}/{len(escape_defects)}) | FPR={fpr_pct:.2f}% ({len(fp_normal)}/{len(target_normal)}) | Precision={prec_pct:.2f}% | F1={f1_cur_pct:.2f}%",
    ])
else:
    summary_block.append(
        f"  🎯 純正常池誤報表現 (FPR)     : 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (扣除Golden後 FPR={fpr_pct:.2f}%)"
    )
if overfit_line:
    summary_block.append(f"  🩺 模型過擬合自檢 (Overfit)    : {overfit_line}")
summary_block.extend([
    "-" * 95,
    f"  📊 非 Golden 正常圖分數分佈    : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}",
    f"  📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}",
])
if escape_defects:
    esc_scores = [float(r["Max_Score"]) for r in escape_defects]
    summary_block.append(
        f"  🔍 隱藏 Escape 瑕疵圖分數分佈  : Min={float(np.min(esc_scores)):.4f} | Median={float(np.median(esc_scores)):.4f} | Max={float(np.max(esc_scores)):.4f}"
    )
summary_block.append("=" * 95 + "\n")

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
        total - len(non_golden),
        len(target_normal),
        len(escape_defects),
        f"{eff_thresh:.4f}",
        f"{auroc_val:.4f}",
        f"{ap_val:.4f}",
        f"{f1_max_val:.4f}",
        f"{recall_pct:.2f}",
        f"{fpr_pct:.2f}",
        f"{prec_pct:.2f}",
        f"{f1_cur_pct:.2f}",
        f"{mean_s:.4f}",
        f"{std_s:.4f}",
        os.path.abspath(latest_train_dir),
        os.path.abspath(infer_dir),
        os.path.abspath(latest_csv),
    ])
PYEOF
}

# 自動展開目標清單：若傳入整個資料集根目錄 (例如 ./data/YDFID-2 或 ./data/mvtec)，自動展開底下所有類別
EXPANDED_CAT_DIRS=()
for entry_path in "${TARGET_CAT_DIRS[@]}"; do
    if [ ! -d "${entry_path}" ]; then
        echo "❌ [路徑錯誤] 找不到資料夾: ${entry_path}"
        exit 1
    fi
    if [ -d "${entry_path}/test" ] || [ -d "${entry_path}/test_public" ] || [ -d "${entry_path}/validation" ]; then
        EXPANDED_CAT_DIRS+=("${entry_path}")
    else
        for sub_cat in $(find "${entry_path}" -mindepth 1 -maxdepth 1 -type d | sort); do
            if [ -d "${sub_cat}/test" ] || [ -d "${sub_cat}/test_public" ] || [ -d "${sub_cat}/validation" ]; then
                EXPANDED_CAT_DIRS+=("${sub_cat}")
            fi
        done
    fi
done

for cat_path in "${EXPANDED_CAT_DIRS[@]}"; do
    run_single_public_category "${cat_path}"
done

# 產生「全類別並列彙整總表（含平均指標）」附加到總表最下方，方便一鍵截圖
python - "${BATCH_SUMMARY_CSV}" "${BATCH_SUMMARY_TXT}" << 'PYEOF'
import os
import sys
import csv
import numpy as np

csv_p, txt_p = sys.argv[1], sys.argv[2]
if not os.path.isfile(csv_p):
    sys.exit(0)

with open(csv_p, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    sys.exit(0)

table_lines = [
    "=" * 124,
    " 📋 [FoundAD 廠外模擬無 GT 測試 — 全類別彙整總表 (All Categories Summary Table)]",
    "-" * 124,
    f" {'Dataset_Key':<22} | {'Total':>5} | {'Gold':>4} | {'Norm':>4} | {'Esc':>3} | {'Thresh':>6} | {'I-AUROC':>7} | {'I-AP':>6} | {'F1-max':>6} | {'Recall(%)':>9} | {'FPR(%)':>7} | {'Normal Mean±Std'}",
    "-" * 124,
]

esc_rows = []
all_fprs = []
all_threshs = []
all_means = []
all_stds = []

for r in rows:
    esc_cnt = int(r["Escape_Imgs"])
    fpr_v = float(r["FPR_Pct"])
    th_v = float(r["Eff_Threshold"])
    m_v = float(r["Normal_Mean"])
    s_v = float(r["Normal_Std"])
    all_fprs.append(fpr_v)
    all_threshs.append(th_v)
    all_means.append(m_v)
    all_stds.append(s_v)

    if esc_cnt > 0:
        esc_rows.append(r)
        auroc_s = f"{float(r['I_AUROC']):.4f}"
        ap_s = f"{float(r['I_AP']):.4f}"
        f1m_s = f"{float(r['F1_Max']):.4f}"
        rec_s = f"{float(r['Recall_Pct']):6.2f}%"
    else:
        auroc_s = "  N/A  "
        ap_s = " N/A  "
        f1m_s = " N/A  "
        rec_s = "   N/A   "

    table_lines.append(
        f" {r['Dataset_Key']:<22} | {int(r['Total_Imgs']):>5} | {int(r['Golden_Imgs']):>4} | "
        f"{int(r['Normal_Imgs']):>4} | {esc_cnt:>3} | {th_v:>6.4f} | {auroc_s:>7} | {ap_s:>6} | "
        f"{f1m_s:>6} | {rec_s:>9} | {fpr_v:>6.2f}% | {m_v:.4f}±{s_v:.4f}"
    )

table_lines.append("-" * 124)
if esc_rows:
    avg_auroc = f"{np.mean([float(r['I_AUROC']) for r in esc_rows]):.4f}"
    avg_ap = f"{np.mean([float(r['I_AP']) for r in esc_rows]):.4f}"
    avg_f1m = f"{np.mean([float(r['F1_Max']) for r in esc_rows]):.4f}"
    avg_rec = f"{np.mean([float(r['Recall_Pct']) for r in esc_rows]):6.2f}%"
else:
    avg_auroc, avg_ap, avg_f1m, avg_rec = "  N/A  ", " N/A  ", " N/A  ", "   N/A   "

table_lines.append(
    f" {'AVERAGE (平均)':<22} | {'---':>5} | {'---':>4} | {'---':>4} | {'---':>3} | "
    f"{np.mean(all_threshs):>6.4f} | {avg_auroc:>7} | {avg_ap:>6} | {avg_f1m:>6} | "
    f"{avg_rec:>9} | {np.mean(all_fprs):>6.2f}% | {np.mean(all_means):.4f}±{np.mean(all_stds):.4f}"
)
table_lines.append("=" * 124 + "\n")

summary_table_str = "\n".join(table_lines)
print("\n" + summary_table_str)
with open(txt_p, "a", encoding="utf-8") as bf:
    bf.write(summary_table_str + "\n")
PYEOF

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================================="
echo " ✅ 全部 ${#EXPANDED_CAT_DIRS[@]} 個類別 (FoundAD) 執行完畢！所有紀錄皆已獨立保存："
echo "    📄 本次多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo "    📄 最新多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo "    📊 最新多資料集完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo "    📈 系統 RAM / VRAM 資源監控日誌  : ${MEM_MONITOR_LOG}"
echo "=========================================================================================================================="