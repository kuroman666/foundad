
#!/usr/bin/env bash
# ============================================================
# FoundAD Old Inference — No-Split
#
# 用途：
#   1. 使用既有 FoundAD 權重進行 Inference
#   2. 輸出完整 Anomaly Map、Bounding Box、Ranking
#   3. 產生每個 Dataset 及跨 Dataset 統計
#   4. 不執行 4_split_eval_results.py
#
# 請在 FoundAD 專案根目錄執行
# ============================================================

set -e

# ============================================================
# 1. DINOv3 Offline Settings
# ============================================================

COMPANY_DINO_CKPT="/workspace/model_weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

export FAB_LOCAL_DINO_CKPT="${COMPANY_DINO_CKPT}"
export FAB_LOCAL_DINO_REPO="$(pwd)/hub_cache/facebookresearch_dinov3_main"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# ============================================================
# 2. 使用者設定區
# ============================================================

OUTPUT_ROOT="./wafer_escape_results"
DATASETS_DIR="./datasets"

# FoundAD 訓練完成的權重
CUSTOM_CKPT_PATH=""

# 指定權重時，DATA_NAME 會從權重路徑自動取得
CUSTOM_DATA_NAME=""

# 訓練時的 DIY_NAME，不需要加 dinov3 前綴
CUSTOM_DIY_NAME=""

# 固定 Threshold，請自行設定
CUSTOM_THRESHOLD=""

# ------------------------------------------------------------
# 模型與推論設定
# ------------------------------------------------------------

SHOTS=4

FAB_CKPT_STEP="${FAB_CKPT_STEP:-${CKPT_STEP:--1}}"

APP_TEST="test"

THRESH_MODE="fixed"

SIGMA_K=3.5
MIN_THRESHOLD=0
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55

RAW_SCORE_MODE=""

# ------------------------------------------------------------
# Wafer Post-processing
# ------------------------------------------------------------

WAFER_EDGE_MODE="auto"
WAFER_EDGE_MARGIN=0.04

BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012

# ------------------------------------------------------------
# Tiling / Inference
# ------------------------------------------------------------

TILE_SIZE=512
TILE_OVERLAP=64
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5

# 如需重新切塊，設定 "--force_infer"
FORCE_INFER=""

# 如需關閉切塊，設定 "--disable_tiling"
DISABLE_TILING=""


# ============================================================
# 3. 解析命令列參數
# ============================================================

CLI_POOL_DIRS=()
CLI_THRESHOLD=""

for arg in "$@"; do

    if [[ "${arg}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then

        CLI_THRESHOLD="${arg}"

    elif [[ "${arg}" == *.pth.tar ]]; then

        CUSTOM_CKPT_PATH="${arg}"

    else

        CLI_POOL_DIRS+=("${arg}")

    fi

done

# 命令列 Threshold 優先於腳本設定
THRESHOLD="${CLI_THRESHOLD:-${CUSTOM_THRESHOLD:-${THRESHOLD:-}}}"

if [[ ! "${THRESHOLD}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then

    echo 'ERROR: 請設定 CUSTOM_THRESHOLD="0.28"'
    exit 1

fi

# 支援一次指定多個 Dataset
if [ ${#CLI_POOL_DIRS[@]} -gt 0 ]; then

    TARGET_POOL_DIRS=("${CLI_POOL_DIRS[@]}")

else

    TARGET_POOL_DIRS=(
        "${DATASETS_DIR}/fab_wafer_pool"
    )

fi


# ============================================================
# 4. 離線環境檢查
# ============================================================

if [ ! -f "${FAB_LOCAL_DINO_CKPT}" ]; then

    echo "ERROR: 找不到 DINOv3 預訓練權重"
    echo "${FAB_LOCAL_DINO_CKPT}"
    exit 1

fi

if [ ! -f "${FAB_LOCAL_DINO_REPO}/hubconf.py" ]; then

    echo "ERROR: 找不到離線 DINOv3 Repository"
    echo "${FAB_LOCAL_DINO_REPO}"
    exit 1

fi

if [ ! -f "3_infer_fab_no_gt_old.py" ]; then

    echo "ERROR: 請先切換到 FoundAD 專案根目錄"
    exit 1

fi


# ============================================================
# 5. Old 模組相容處理
# ============================================================

if [ ! -f "1_auto_golden.py" ] && \
   [ -f "1_auto_golden_old.py" ]; then

    mkdir -p fab_tools

    if [ ! -f "fab_tools/auto_golden.py" ]; then
        cp "1_auto_golden_old.py" "fab_tools/auto_golden.py"
    fi

    python - <<'PY'
import shutil

shutil.copyfile(
    "1_auto_golden_old.py",
    "1_auto_golden.py"
)

with open("1_auto_golden.py", "a", encoding="utf-8") as f:
    f.write('''
# Compatibility shim for Old Inference
try:
    _old = __import__("2_train_fab_no_gt_old")
except ImportError:
    _old = None

for _n in (
    "format_duration",
    "log_and_print",
    "safe_rmtree",
    "safe_copy",
    "compute_pool_signature",
    "load_image_rgb_uint8",
    "get_axis_positions",
    "is_background_tile",
    "prepare_tiled_golden_trainset",
):
    globals()[_n] = getattr(_old, _n, None)

load_or_prepare_golden_dataset = None
''')
PY

fi


# ============================================================
# 6. Summary 輸出設定
# ============================================================

BATCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_ROOT}"

BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/inside_infer_old_summary_${BATCH_TIMESTAMP}.txt"

BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_inside_infer_old_summary.txt"

BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_inside_infer_old_summary.csv"

{
    echo "========================================================"
    echo "FoundAD Old — 純推論 + 完整資料集統計（No-Split）"
    echo "啟動時間：${BATCH_TIMESTAMP}"
    echo "========================================================"
} > "${BATCH_SUMMARY_TXT}"

echo "Pool_Key,Category,Total_Imgs,Golden_Imgs,Unseen_Test_Imgs,Flagged_Anomalies,Anomaly_Rate_Pct,Threshold_Mode,Eff_Threshold,Core_Normal_Med,Core_Normal_Mean,Core_Normal_Std,Data_Name,DIY_Name,Checkpoint_Path,Model_Dir,Infer_Dir,Report_CSV" \
    > "${BATCH_SUMMARY_CSV}"


# ============================================================
# 7. 單一 Dataset 執行函式
# ============================================================

run_one_pool() {

    local POOL_DIR="$1"

    if [ ! -d "${POOL_DIR}" ]; then
        echo "ERROR: 測試資料夾不存在：${POOL_DIR}"
        exit 1
    fi

    local FULL_POOL_DIR
    local CAT_NAME
    local DS_PREFIX
    local BATCH_ID
    local UNIQUE_TAG

    FULL_POOL_DIR="$(cd "${POOL_DIR}" && pwd)"

    CAT_NAME="$(basename "${FULL_POOL_DIR}")"
    CAT_NAME="${CAT_NAME// /_}"

    DS_PREFIX="$(basename "$(dirname "${FULL_POOL_DIR}")")"
    DS_PREFIX="${DS_PREFIX//-/_}"
    DS_PREFIX="${DS_PREFIX// /_}"

    # 預期資料夾結構：
    # .../W642/TMQB28/chipping
    #
    # 自動取得批號
    BATCH_ID="$(basename "$(dirname "$(dirname "${FULL_POOL_DIR}")")")"

    # 輸出名稱：
    # W642_TMQB28_chipping
    UNIQUE_TAG="${BATCH_ID}_${DS_PREFIX}_${CAT_NAME}"

    local DATA_NAME="fab_inside_old_${UNIQUE_TAG}_${SHOTS}shot"
    local DIY_NAME="fab_inside_old_${UNIQUE_TAG}"

    local ACTIVE_CKPT_STEP="${FAB_CKPT_STEP}"

    local SAVE_BASE_DIR="${OUTPUT_ROOT}/inside_no_gt_old_${UNIQUE_TAG}"

    local CKPT_FOUND=""

    # --------------------------------------------------------
    # 解析 FoundAD 權重
    # --------------------------------------------------------

    if [ -n "${CUSTOM_CKPT_PATH}" ]; then

        if [ ! -f "${CUSTOM_CKPT_PATH}" ]; then
            echo "ERROR: 權重不存在：${CUSTOM_CKPT_PATH}"
            exit 1
        fi

        CKPT_FOUND="$(realpath "${CUSTOM_CKPT_PATH}")"

        local INFO

        INFO="$(python - "${CKPT_FOUND}" "${ACTIVE_CKPT_STEP}" <<'PY'
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
parts = p.parts

if "logs" not in parts:
    raise SystemExit(
        "ERROR: 權重應位於 logs/<DATA_NAME>/dinov3<DIY_NAME>/ 下"
    )

idx = len(parts) - 1 - parts[::-1].index("logs")

if idx + 2 >= len(parts):
    raise SystemExit("ERROR: 權重資料夾結構不完整")

data_name = parts[idx + 1]
model_dir = parts[idx + 2]

match = re.fullmatch(
    r"train-step(\d+)\.pth\.tar",
    p.name
)

step = match.group(1) if match else sys.argv[2]

print(f"{data_name}|{model_dir}|{step}")
PY
)"

        local MODEL_DIR_NAME

        IFS='|' read -r DATA_NAME MODEL_DIR_NAME ACTIVE_CKPT_STEP <<< "${INFO}"

        # 移除 dinov3 前綴，避免重複
        DIY_NAME="${MODEL_DIR_NAME#dinov3}"

    elif [ -n "${CUSTOM_DATA_NAME}" ]; then

        DATA_NAME="${CUSTOM_DATA_NAME}"

    fi

    # 手動指定 DIY_NAME 優先
    if [ -n "${CUSTOM_DIY_NAME}" ]; then
        DIY_NAME="${CUSTOM_DIY_NAME}"
    fi

    # --------------------------------------------------------
    # 沒有指定權重時，搜尋預設模型資料夾
    # --------------------------------------------------------

    if [ -z "${CKPT_FOUND}" ]; then

        local MODEL_ROOT="./logs/${DATA_NAME}/dinov3${DIY_NAME}"

        CKPT_FOUND="$(
            find "${MODEL_ROOT}" \
                -type f \
                -name 'train-step*.pth.tar' \
                2>/dev/null | sort | tail -n 1
        )"

        if [ -z "${CKPT_FOUND}" ]; then
            echo "ERROR: 找不到模型權重：${MODEL_ROOT}"
            exit 1
        fi

        CKPT_FOUND="$(realpath "${CKPT_FOUND}")"

    fi


    # ========================================================
    # 8. FoundAD Inference
    # ========================================================

    echo ""
    echo "========================================================"
    echo "[FoundAD Old Inference / No-Split]"
    echo "Dataset     : ${FULL_POOL_DIR}"
    echo "DATA_NAME   : ${DATA_NAME}"
    echo "DIY_NAME    : ${DIY_NAME}"
    echo "CKPT_STEP   : ${ACTIVE_CKPT_STEP}"
    echo "Checkpoint  : ${CKPT_FOUND}"
    echo "Threshold   : ${THRESH_MODE} / ${THRESHOLD}"
    echo "Output      : ${SAVE_BASE_DIR}"
    echo "========================================================"

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


    # ========================================================
    # 9. 完整 Dataset 統計（不進行 Split）
    # ========================================================

    python - \
        "${SAVE_BASE_DIR}" \
        "${CAT_NAME}" \
        "${UNIQUE_TAG}" \
        "${BATCH_SUMMARY_TXT}" \
        "${BATCH_SUMMARY_CSV}" \
        "${CKPT_FOUND}" \
        "${DATA_NAME}" \
        "${DIY_NAME}" \
        "${THRESH_MODE}" \
        "${THRESHOLD}" \
        "${FULL_POOL_DIR}" <<'PY'

import csv
import glob
import os
import re
import sys
import numpy as np

(
    save_base,
    cat_name,
    tag,
    txt_path,
    csv_path,
    checkpoint,
    data_name,
    diy_name,
    mode,
    threshold,
    pool,
) = sys.argv[1:]

# ---------------------------------------------------------
# 讀取最新 Inference Ranking
# ---------------------------------------------------------

candidates = sorted(
    glob.glob(
        os.path.join(
            save_base,
            "*",
            "escape_ranking_report.csv"
        )
    )
)

if not candidates:
    raise FileNotFoundError(
        f"Inference 完成後找不到 Ranking CSV：{save_base}"
    )

report = candidates[-1]
run_dir = os.path.dirname(report)

with open(report, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError(f"Ranking CSV 沒有資料：{report}")

# ---------------------------------------------------------
# Golden / Normal / Abnormal
# ---------------------------------------------------------

golden = [
    r for r in rows
    if str(r.get("Status", "")).startswith("GOLDEN")
]

non_golden = [
    r for r in rows
    if not str(r.get("Status", "")).startswith("GOLDEN")
]

flagged = [
    r for r in non_golden
    if "ANOMALY" in str(r.get("Status", ""))
]

passed = [
    r for r in non_golden
    if "ANOMALY" not in str(r.get("Status", ""))
]

normal = passed or non_golden or rows

scores = np.array(
    [float(r["Max_Score"]) for r in normal],
    dtype=float
)

all_scores = np.array(
    [float(r["Max_Score"]) for r in (non_golden or rows)],
    dtype=float
)

if not np.all(np.isfinite(scores)) or \
   not np.all(np.isfinite(all_scores)):
    raise ValueError("Max_Score 包含無效數值")

# ---------------------------------------------------------
# 分數分布統計
# ---------------------------------------------------------

mean = float(scores.mean())
std = float(scores.std())
median = float(np.median(scores))

max_score = float(scores.max())
p99 = float(np.percentile(scores, 99))
p95 = float(np.percentile(scores, 95))

med_all = float(np.median(all_scores))
mad_all = float(np.median(np.abs(all_scores - med_all)))

th_3sigma = float(
    all_scores.mean() + 3.0 * all_scores.std()
)

th_modz = float(
    med_all + (3.5 / 0.6745) * mad_all
)

anom_pct = (
    100.0 * len(flagged) / len(non_golden)
    if non_golden else 0.0
)

# 優先從原始報表讀取實際有效 Threshold
actual_threshold = float(threshold)

report_txt = report.replace(".csv", ".txt")

if os.path.isfile(report_txt):
    with open(report_txt, encoding="utf-8") as f:
        content = f.read(2500)

    match = re.search(
        r"Effective Threshold:\s*([0-9.]+)",
        content
    )

    if match:
        actual_threshold = float(match.group(1))

# ---------------------------------------------------------
# 終端機顯示
# ---------------------------------------------------------

lines = [
    "=" * 85,
    f"[{tag}] FoundAD Old — No-Split 完整資料集統計",
    "-" * 85,
    f"Dataset            : {pool}",
    f"DATA_NAME          : {data_name}",
    f"DIY_NAME           : {diy_name}",
    f"指定 Checkpoint    : {checkpoint}",
    f"Checkpoint Folder  : {os.path.dirname(checkpoint)}",
    f"Inference Folder   : {run_dir}",
    f"Ranking CSV        : {os.path.abspath(report)}",
    f"Threshold          : {mode} / {actual_threshold:.4f}",
    "-" * 85,
    f"Total Images       : {len(rows)}",
    f"Golden 標記數      : {len(golden)}",
    f"Non-Golden Images  : {len(non_golden)}",
    f"Flagged Anomalies  : {len(flagged)}",
    f"Anomaly Rate       : {anom_pct:.2f}%",
    "-" * 85,
    (
        f"Score Distribution : Max={max_score:.4f} | "
        f"P99={p99:.4f} | P95={p95:.4f} | "
        f"Median={median:.4f} | Mean={mean:.4f} | Std={std:.4f}"
    ),
    (
        f"Reference Threshold: "
        f"3-Sigma={th_3sigma:.4f} | "
        f"Mod-Z(3.5)={th_modz:.4f}"
    ),
]

if flagged:
    top5 = sorted(
        flagged,
        key=lambda r: float(r["Max_Score"]),
        reverse=True
    )[:5]

    names = [
        f"{r.get('Original_Filename', '')}"
        f"({float(r['Max_Score']):.3f})"
        for r in top5
    ]

    lines.append(
        "Top-5 Anomalies    : " + ", ".join(names)
    )


# ============================================================
# Threshold Reference Comparison
# 僅統計候選數量，不改變實際 Inference 結果
# ============================================================

# 排除 Golden 後的全部 Wafer 分數
ref_scores = all_scores

n = len(ref_scores)
ref_median = float(np.median(ref_scores))
ref_mean = float(np.mean(ref_scores))
ref_std = float(np.std(ref_scores))
ref_p95 = float(np.percentile(ref_scores, 95))

# 建立不同 Threshold 的比較表
threshold_refs = [
    ("Effective Threshold", actual_threshold),
    ("Median + 2 Sigma", ref_median + 2 * ref_std),
    ("Median + 3 Sigma", ref_median + 3 * ref_std),
    ("Median + 3.5 Sigma", ref_median + 3.5 * ref_std),
    ("Mean + 3 Sigma", th_3sigma),
    ("Modified Z-Score 3.5", th_modz),
    ("P95", ref_p95),
]

lines.extend([
    "-" * 85,
    "Threshold Reference Comparison",
    f"Score Population : Non-Golden Wafer (N={n})",
    f"Median={ref_median:.4f} | "
    f"Mean={ref_mean:.4f} | Std={ref_std:.4f}",
    "-" * 85,
    f"{'Method':<26} {'Threshold':>10} "
    f"{'Candidates':>12} {'Rate':>10} {'Rank Cut':>12}",
])

comparison_rows = []

for method, th in threshold_refs:

    count = int(np.sum(ref_scores >= th))
    rate = 100.0 * count / n if n else 0.0

    rank_cut = f"Top {count}" if count else "None"

    lines.append(
        f"{method:<26} {th:>10.4f} "
        f"{count:>8}/{n:<3} "
        f"{rate:>9.2f}% "
        f"{rank_cut:>12}"
    )

    comparison_rows.append([
        method,
        f"{th:.6f}",
        count,
        n,
        f"{rate:.4f}",
        rank_cut,
    ])

# 真正完成 Bounding Box 篩選的結果
lines.extend([
    "-" * 85,
    f"FINAL Effective Threshold : {actual_threshold:.4f}",
    f"FINAL Score Candidates    : "
    f"{int(np.sum(ref_scores >= actual_threshold))}/{n}",
    f"FINAL BBox Anomalies      : {len(flagged)}/{n}",
    f"FINAL Anomaly Rate        : {anom_pct:.2f}%",
])

# 額外匯出門檻比較 CSV
comparison_csv = os.path.join(
    run_dir,
    "threshold_reference_comparison.csv"
)

with open(
    comparison_csv, "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Threshold_Method",
        "Threshold_Value",
        "Score_Candidate_Count",
        "Total_NonGolden",
        "Candidate_Rate_Pct",
        "Rank_Cut",
    ])

    writer.writerows(comparison_rows)

lines.append(
    f"Threshold Comparison CSV : {comparison_csv}"
)


lines.append("=" * 85)

summary = "\n".join(lines)

print("\n" + summary)

# ---------------------------------------------------------
# 寫入 TXT Summary
# ---------------------------------------------------------

with open(txt_path, "a", encoding="utf-8") as f:
    f.write(summary + "\n")

# ---------------------------------------------------------
# 寫入 CSV Summary
# ---------------------------------------------------------

with open(
    csv_path,
    "a",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        tag,
        cat_name,
        len(rows),
        len(golden),
        len(non_golden),
        len(flagged),
        f"{anom_pct:.2f}",
        mode,
        f"{actual_threshold:.4f}",
        f"{median:.4f}",
        f"{mean:.4f}",
        f"{std:.4f}",
        data_name,
        diy_name,
        checkpoint,
        os.path.dirname(checkpoint),
        os.path.abspath(run_dir),
        os.path.abspath(report),
    ])

PY

}


# ============================================================
# 10. 依序執行多個 Dataset
# ============================================================

for dataset_path in "${TARGET_POOL_DIRS[@]}"; do

    run_one_pool "${dataset_path}"

done


# ============================================================
# 11. 完成與 Summary 輸出
# ============================================================

cp -f \
    "${BATCH_SUMMARY_TXT}" \
    "${BATCH_SUMMARY_LATEST}"

echo ""
echo "========================================================"
echo "FoundAD Old Inference 完成"
echo "模式：純推論 + 完整資料集統計（No-Split）"
echo "Dataset 數量：${#TARGET_POOL_DIRS[@]}"
echo ""
echo "Summary TXT：${BATCH_SUMMARY_TXT}"
echo "Latest TXT ：${BATCH_SUMMARY_LATEST}"
echo "Latest CSV ：${BATCH_SUMMARY_CSV}"
echo "========================================================"
