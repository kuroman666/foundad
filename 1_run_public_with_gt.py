# 檔案位置: FoundAD/1_run_public_with_gt.py (全選覆蓋並存檔)
import os
import re
import stat
import glob
import csv
import shutil
import argparse
import subprocess
from datetime import datetime


def force_writable_and_remove(target_dir: str):
    """解除唯讀權限 (如 MVTec 預設 0444 權限) 並安全清空舊抽樣資料夾，防止第二次執行時發生 PermissionError。"""
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


def find_latest_ckpt_step(data_name: str, diy_name: str, req_step: int) -> int:
    pattern = f"./logs/{data_name}/*{diy_name}*/train-step*.pth.tar"
    files = glob.glob(pattern) or glob.glob(f"./logs/{data_name}/**/train-step*.pth.tar", recursive=True)
    steps = [
        int(re.search(r"train-step(\d+)\.pth\.tar", f).group(1))
        for f in files if re.search(r"train-step(\d+)\.pth\.tar", f)
    ]
    if not steps or req_step in steps:
        return req_step
    best_step = max(steps)
    print(f"[Auto-Ckpt] 自動對齊最新訓練步數 ckpt_step={best_step}")
    return best_step


def run_and_log(cmd: str, log_file: str) -> str:
    print(f"\n[EXEC] {cmd}\n" + "-" * 60)
    output_lines = []
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n[CMD] {cmd}\n{'='*60}\n")
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            output_lines.append(line)
        process.wait()
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

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_txt = os.path.join(args.save_dir, f"terminal_full_log_{timestamp}.txt")
    summary_txt = os.path.join(args.save_dir, f"metrics_summary_{timestamp}.txt")
    summary_csv = os.path.join(args.save_dir, f"metrics_summary_{timestamp}.csv")

    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    train_root_path = os.path.join(abs_datasets_dir, args.data_name)
    abs_test_root = os.path.abspath(args.test_root)

    if not args.skip_sample:
        # 抽樣前先解除舊資料夾中 MVTec 唯讀圖片的權限並清理，徹底避免 PermissionError
        force_writable_and_remove(train_root_path)
        sample_cmd = (
            f"python foundad/src/sample.py source={abs_test_root} "
            f"target={train_root_path} num_samples={args.shots} seed={args.seed}"
        )
        run_and_log(sample_cmd, full_log_txt)
        make_tree_writable(train_root_path)

    num_classes = len([c.strip() for c in args.classnames.strip("[]").split(",") if c.strip()])
    total_batch_size = min(args.max_batch_size, max(args.shots, num_classes * args.shots))

    if not args.skip_train:
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={total_batch_size} "
            f"data.dataset={args.dataset} data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={train_root_path} "
            f"data.mvtec_classnames='{args.classnames}' "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        run_and_log(train_cmd, full_log_txt)

    actual_step = find_latest_ckpt_step(args.data_name, args.diy_name, args.ckpt_step)
    test_cmd = (
        f"python foundad/main.py mode=AD data.dataset={args.dataset} "
        f"data.data_name={args.data_name} data.mvtec_classnames='{args.classnames}' "
        f"diy_name={args.diy_name} data.test_root={abs_test_root} "
        f"app={args.app_test} app.ckpt_step={actual_step}"
    )
    test_output = run_and_log(test_cmd, full_log_txt)

    metric_lines = [
        line.strip() for line in test_output.splitlines()
        if any(k in line for k in ["AUROC_i", "PRO-AUC", "Evaluating", "Mean |"])
    ]

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"=== FoundAD Metrics Report ({timestamp}) ===\n")
        f.write(f"Command: {test_cmd}\n" + "-" * 75 + "\n")
        for line in metric_lines:
            f.write(line + "\n")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class_or_Mean", "AUROC_i", "AUPR_i", "AUROC_p", "PRO_AUC"])
        for line in metric_lines:
            if "|" in line and "AUROC_i" in line:
                parts = [p.strip() for p in line.split("|")]
                cls_name = parts[0].split(":")[-1].strip()
                vals = [p.split()[-1] for p in parts[1:]]
                writer.writerow([cls_name] + vals)

    for seg_src in glob.glob(f"./logs/{args.data_name}/**/segmentation", recursive=True):
        if os.path.isdir(seg_src):
            seg_dst = os.path.join(args.save_dir, "segmentation")
            force_writable_and_remove(seg_dst)
            shutil.copytree(seg_src, seg_dst, dirs_exist_ok=True)
            make_tree_writable(seg_dst)

    print(f"[✓] Task 1 報表與圖片已輸出至: {args.save_dir}")


if __name__ == "__main__":
    main()