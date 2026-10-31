# 檔案位置: FoundAD/1_run_public_with_gt.py (全選覆蓋並存檔)
import os
import re
import time
import stat
import glob
import csv
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import List, Tuple


def format_duration(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS.xx 與總秒數。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f} ({seconds:.2f} 秒)"


def log_and_print(msg: str, log_files: List[str]):
    """同時在 Terminal 顯示並寫入 log.txt。"""
    print(msg, flush=True)
    for lf in log_files:
        if lf:
            with open(lf, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def force_writable_and_remove(target_dir: str):
    """解除唯讀權限並安全清空舊資料夾，防止 PermissionError。"""
    if not os.path.exists(target_dir):
        return

    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            func(path)
        except Exception:
            pass

    for root, dirs, files in os.walk(target_dir):
        for d in dirs:
            p = os.path.join(root, d)
            try:
                os.chmod(p, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                pass
        for f in files:
            p = os.path.join(root, f)
            try:
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass

    shutil.rmtree(target_dir, onerror=_onerror)


def make_tree_writable(target_dir: str):
    """確保抽樣後的資料夾與圖片皆具備寫入權限 (0644/0755)。"""
    if not os.path.exists(target_dir):
        return
    for root, dirs, files in os.walk(target_dir):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o755)
            except OSError:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o644)
            except OSError:
                pass


def find_latest_ckpt_info(data_name: str, diy_name: str, req_step: int, log_files: List[str]) -> Tuple[int, List[str]]:
    """尋找最新訓練步數並回傳 (步數, 權重檔完整路徑清單)。"""
    pattern = f"./logs/{data_name}/*{diy_name}*/train-step*.pth.tar"
    files = glob.glob(pattern) or glob.glob(f"./logs/{data_name}/**/train-step*.pth.tar", recursive=True)
    steps = [
        int(re.search(r"train-step(\d+)\.pth\.tar", f).group(1))
        for f in files if re.search(r"train-step(\d+)\.pth\.tar", f)
    ]
    if not steps:
        return req_step, []

    chosen_step = req_step if req_step in steps else max(steps)
    if chosen_step != req_step:
        log_and_print(f"[Auto-Ckpt] 自動對齊最新訓練步數 ckpt_step={chosen_step}", log_files)

    matched_ckpts = sorted([
        os.path.abspath(f) for f in files
        if os.path.basename(f) == f"train-step{chosen_step}.pth.tar"
    ])
    return chosen_step, matched_ckpts


def run_and_log(cmd: str, log_files: List[str]) -> str:
    header = f"\n[EXEC] {cmd}\n" + "-" * 60
    print(header, flush=True)
    output_lines = []

    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"

    files_to_write = [open(lf, "a", encoding="utf-8") for lf in log_files if lf]
    try:
        for f in files_to_write:
            f.write(f"\n{'='*60}\n[CMD] {cmd}\n{'='*60}\n")
            f.flush()

        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            for f in files_to_write:
                f.write(line)
                f.flush()
            output_lines.append(line)
        process.wait()
    finally:
        for f in files_to_write:
            f.close()

    full_out = "".join(output_lines)
    if process.returncode != 0 or "Traceback (most recent call last):" in full_out:
        raise RuntimeError(f"指令執行過程發生錯誤，請檢查上方 Traceback: {cmd}")
    return full_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--test_root", type=str, default="./datasets/mvtec")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_4shot")
    parser.add_argument("--classnames", type=str, default="[bottle,cable]")
    parser.add_argument("--app_train", type=str, default="train_dinov3")
    parser.add_argument("--app_test", type=str, default="test")
    parser.add_argument("--diy_name", type=str, default="dbug")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--max_batch_size", type=int, default=16, help="限制最大訓練 Batch Size，防止高解析度下 OOM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_sample", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./output/1_public_with_gt")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.abspath(args.save_dir)
    run_save_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)

    fixed_log_txt = os.path.join(run_save_dir, "log.txt")
    parent_latest_log = os.path.join(base_save_dir, "latest_log.txt")
    log_files = [fixed_log_txt, parent_latest_log]

    summary_txt = os.path.join(run_save_dir, "metrics_summary.txt")
    summary_csv = os.path.join(run_save_dir, "metrics_summary.csv")

    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    train_root_path = os.path.join(abs_datasets_dir, args.data_name)
    abs_test_root = os.path.abspath(args.test_root)

    pipeline_start_t = time.perf_counter()
    start_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_and_print(
        f"\n{'='*70}\n"
        f"[Task 1 啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_save_dir}\n"
        f"{'='*70}",
        log_files,
    )

    # =========================================================================
    # [階段一：整體訓練階段] (包含少樣本抽樣 + FoundAD 模型訓練)
    # =========================================================================
    sample_duration = 0.0
    if not args.skip_sample:
        t_sample_start = time.perf_counter()
        force_writable_and_remove(train_root_path)
        sample_cmd = (
            f"python foundad/src/sample.py source={abs_test_root} "
            f"target={train_root_path} num_samples={args.shots} seed={args.seed}"
        )
        run_and_log(sample_cmd, log_files)
        make_tree_writable(train_root_path)
        sample_duration = time.perf_counter() - t_sample_start
        log_and_print(f"[⏱️ 計時] 少樣本抽樣耗時: {format_duration(sample_duration)}", log_files)

    num_classes = len([c.strip() for c in args.classnames.strip("[]").split(",") if c.strip()])
    total_batch_size = min(args.max_batch_size, max(args.shots, num_classes * args.shots))

    train_duration = 0.0
    if not args.skip_train:
        t_train_start = time.perf_counter()
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={total_batch_size} "
            f"data.dataset={args.dataset} data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={train_root_path} "
            f"data.mvtec_classnames='{args.classnames}' "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        run_and_log(train_cmd, log_files)
        train_duration = time.perf_counter() - t_train_start
    else:
        log_and_print("[⏱️ 計時] 已設定 --skip_train，跳過訓練階段 (0.00 秒)", log_files)

    total_train_time = sample_duration + train_duration
    actual_step, ckpt_paths = find_latest_ckpt_info(args.data_name, args.diy_name, args.ckpt_step, log_files)
    ckpt_dirs = sorted(list({os.path.dirname(p) for p in ckpt_paths}))
    default_ckpt_root = os.path.abspath(f"./logs/{args.data_name}")

    train_done_msg = [
        f"\n{'='*70}",
        f" [✅ 訓練階段完成 — 時間與權重檔儲存位置]",
        f"{'-'*70}",
        f"  ⏱️ 少樣本抽樣時間   : {format_duration(sample_duration)}",
        f"  ⏱️ 模型訓練時間     : {format_duration(train_duration)}{' (Skipped)' if args.skip_train else ''}",
        f"  ⏱️ 整體訓練總時間   : {format_duration(total_train_time)}",
        f"  📁 訓練樣本資料夾   : {train_root_path}",
        f"  📦 權重檔主資料夾   : {ckpt_dirs[0] if len(ckpt_dirs) == 1 else default_ckpt_root}",
    ]
    if ckpt_paths:
        train_done_msg.append("  📦 產生的權重檔案   :")
        for cp in ckpt_paths:
            train_done_msg.append(f"      └── {cp}")
    train_done_msg.append(f"{'='*70}")
    log_and_print("\n".join(train_done_msg), log_files)

    # =========================================================================
    # [階段二：整體推論階段] (執行 mode=AD 推論與分割圖輸出)
    # =========================================================================
    t_infer_start = time.perf_counter()
    test_cmd = (
        f"python foundad/main.py mode=AD data.dataset={args.dataset} "
        f"data.data_name={args.data_name} data.mvtec_classnames='{args.classnames}' "
        f"diy_name={args.diy_name} data.test_root={abs_test_root} "
        f"app={args.app_test} app.ckpt_step={actual_step}"
    )
    test_output = run_and_log(test_cmd, log_files)
    infer_duration = time.perf_counter() - t_infer_start

    seg_dst = os.path.join(run_save_dir, "segmentation")
    copied_seg = False
    for seg_src in glob.glob(f"./logs/{args.data_name}/**/segmentation", recursive=True):
        if os.path.isdir(seg_src):
            force_writable_and_remove(seg_dst)
            shutil.copytree(seg_src, seg_dst, dirs_exist_ok=True)
            make_tree_writable(seg_dst)
            copied_seg = True

    total_wall_time = time.perf_counter() - pipeline_start_t

    metric_lines = [
        line.strip() for line in test_output.splitlines()
        if any(k in line for k in ["AUROC_i", "PRO-AUC", "Evaluating", "Mean |"])
    ]

    infer_done_msg = (
        f"\n{'='*70}\n"
        f" [✅ 推論階段完成 — 整體時間與結果檔案儲存位置]\n"
        f"{'-'*70}\n"
        f"  ⏱️ 整體訓練總時間       : {format_duration(total_train_time)}\n"
        f"  ⏱️ 整體推論總時間       : {format_duration(infer_duration)}\n"
        f"  ⏱️ 任務全程總耗時       : {format_duration(total_wall_time)}\n"
        f"{'-'*70}\n"
        f"  📦 訓練權重檔所在資料夾 : {ckpt_dirs[0] if len(ckpt_dirs) == 1 else default_ckpt_root}\n"
        f"  📂 本次推論結果主資料夾 : {run_save_dir}\n"
        f"      ├── 分割結果圖片夾  : {seg_dst if copied_seg else '(未產生分割圖)'}\n"
        f"      ├── 評估指標報表TXT : {summary_txt}\n"
        f"      ├── 評估指標報表CSV : {summary_csv}\n"
        f"      └── 本次執行日誌檔  : {fixed_log_txt}\n"
        f"{'='*70}"
    )

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"=== FoundAD Metrics Report ({timestamp}) ===\n")
        f.write(f"Command: {test_cmd}\n" + "-" * 75 + "\n")
        for line in metric_lines:
            f.write(line + "\n")
        f.write("\n" + "\n".join(train_done_msg) + "\n")
        f.write(infer_done_msg + "\n")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class_or_Mean", "AUROC_i", "AUPR_i", "AUROC_p", "PRO_AUC"])
        for line in metric_lines:
            if "|" in line and "AUROC_i" in line:
                parts = [p.strip() for p in line.split("|")]
                cls_name = parts[0].split(":")[-1].strip()
                vals = [p.split()[-1] for p in parts[1:]]
                writer.writerow([cls_name] + vals)
        writer.writerow([])
        writer.writerow(["Item", "Formatted_Duration_or_Path", "Seconds"])
        writer.writerow(["Total_Train_Time", format_duration(total_train_time), f"{total_train_time:.2f}"])
        writer.writerow(["Model_Train_Only", format_duration(train_duration), f"{train_duration:.2f}"])
        writer.writerow(["Total_Inference_Time", format_duration(infer_duration), f"{infer_duration:.2f}"])
        writer.writerow(["Total_Wall_Time", format_duration(total_wall_time), f"{total_wall_time:.2f}"])
        writer.writerow(["Checkpoint_Folder", ckpt_dirs[0] if len(ckpt_dirs) == 1 else default_ckpt_root, ""])
        writer.writerow(["Inference_Result_Folder", run_save_dir, ""])

    log_and_print(infer_done_msg, log_files)


if __name__ == "__main__":
    main()
