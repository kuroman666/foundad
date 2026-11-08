# 檔案位置: FoundAD/2_train_fab_no_gt.py (全選覆蓋此檔：含多資料集獨立命名與 Torch Hub 離線防 504 保護)
import shlex
import os
import re
import time
import glob
import json
import socket
import subprocess
import argparse
import importlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np

TZ_TAIPEI = timezone(timedelta(hours=8))

_ag = importlib.import_module("1_auto_golden")
format_duration = _ag.format_duration
log_and_print = _ag.log_and_print
safe_copy = _ag.safe_copy
scan_pool_images = _ag.scan_pool_images
prepare_mock_unlabeled_pool_from_mvtec = _ag.prepare_mock_unlabeled_pool_from_mvtec
load_or_prepare_golden_dataset = _ag.load_or_prepare_golden_dataset


def get_free_port() -> int:
    """由作業系統動態分配保證空閒的本地 Port，徹底避免 DDP Port 40112 衝突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ensure_offline_hub_patch_dir() -> str:
    """
    建立 sitecustomize.py 自動攔截 torch.hub.load(source='github')：
    優先從本地 ~/.cache/torch/hub 離線載入 DINOv3/DINOv2，徹底杜絕 GitHub HTTP 504 Gateway Time-out！
    """
    patch_dir = os.path.abspath("./.offline_hub_patch")
    os.makedirs(patch_dir, exist_ok=True)
    sc_path = os.path.join(patch_dir, "sitecustomize.py")
    code = '''import os
import glob
try:
    import torch.hub
    _orig_hub_load = torch.hub.load

    def _safe_hub_load(repo_or_dir, model, *args, **kwargs):
        if kwargs.get("source", "github") == "github" and isinstance(repo_or_dir, str) and "/" in repo_or_dir:
            hub_dir = torch.hub.get_dir()
            owner, repo = repo_or_dir.split("/")[:2]
            cands = sorted(glob.glob(os.path.join(hub_dir, f"{owner}_{repo}_*")))
            for c in cands:
                if os.path.isdir(c) and os.path.isfile(os.path.join(c, "hubconf.py")):
                    try:
                        kw_local = dict(kwargs)
                        kw_local["source"] = "local"
                        return _orig_hub_load(c, model, *args, **kw_local)
                    except Exception:
                        pass
        try:
            return _orig_hub_load(repo_or_dir, model, *args, **kwargs)
        except Exception:
            hub_dir = torch.hub.get_dir()
            for c in sorted(glob.glob(os.path.join(hub_dir, "*dino*"))):
                if os.path.isdir(c) and os.path.isfile(os.path.join(c, "hubconf.py")):
                    kw_local = dict(kwargs)
                    kw_local["source"] = "local"
                    return _orig_hub_load(c, model, *args, **kw_local)
            raise

    torch.hub.load = _safe_hub_load
except Exception:
    pass
'''
    if not os.path.isfile(sc_path):
        with open(sc_path, "w", encoding="utf-8") as f:
            f.write(code)
    return patch_dir


def run_cmd_and_log(cmd: str, log_files: List[str], env: Optional[Dict] = None) -> str:
    """執行子程序指令，自動注入空閒 FAB_MASTER_PORT 與離線 Hub Patch，同步寫入 log.txt。"""
    header = f"\n[EXEC] {cmd}\n" + "-" * 60
    print(header, flush=True)
    output_lines = []
    run_env = (env or os.environ).copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if "FAB_MASTER_PORT" not in run_env:
        run_env["FAB_MASTER_PORT"] = str(get_free_port())

    patch_dir = _ensure_offline_hub_patch_dir()
    old_pp = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = f"{patch_dir}:{old_pp}" if old_pp else patch_dir

    files_to_write = [open(lf, "a", encoding="utf-8") for lf in log_files if lf]
    try:
        for f in files_to_write:
            f.write(f"\n{'='*60}\n[CMD] {cmd} (PORT={run_env['FAB_MASTER_PORT']})\n{'='*60}\n")
            f.flush()
        process = subprocess.Popen(
            shlex.split(cmd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=run_env,
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


def _get_candidate_log_dirs(data_name: str, diy_name: str, shots: int = 4) -> List[str]:
    """嚴格搜尋當前 data_name 專屬的權重目錄。"""
    patterns = [
        f"./logs/{data_name}/*{diy_name}*",
        f"./logs/{data_name}/*",
    ]
    if data_name in (f"mvtec_sim_{shots}shot", f"mvtec_fab_{shots}shot"):
        patterns.extend([
            f"./logs/mvtec_sim_{shots}shot/*{diy_name}*",
            f"./logs/mvtec_sim_{shots}shot/*sim_escape*",
            f"./logs/mvtec_fab_{shots}shot/*{diy_name}*",
            f"./logs/mvtec_fab_{shots}shot/*fab_wafer*",
        ])

    seen = set()
    dirs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and os.path.abspath(d) not in seen:
                seen.add(os.path.abspath(d))
                dirs.append(d)
    return dirs


def find_existing_trained_ckpt(
    data_name: str,
    diy_name: str,
    req_step: int,
    shots: int = 4,
) -> Optional[str]:
    """檢查當前 data_name 是否已經有訓練好的權重檔 (優先找對應步數，其次找最大步數)。"""
    for d in _get_candidate_log_dirs(data_name, diy_name, shots):
        exact = os.path.join(d, f"train-step{req_step}.pth.tar")
        if os.path.isfile(exact):
            return os.path.abspath(exact)
        step_files = glob.glob(os.path.join(d, "train-step*.pth.tar"))
        if step_files:
            step_files.sort(key=lambda p: int(re.search(r"train-step(\d+)", p).group(1)))
            return os.path.abspath(step_files[-1])
        alias = os.path.join(d, "pretrained.pth.tar")
        if os.path.isfile(alias):
            return os.path.abspath(alias)
    return None


def prepare_ckpt_for_demo(
    data_name: str,
    diy_name: str,
    req_step: int,
    log_files: List[str],
    shots: int = 4,
) -> Tuple[int, List[str], List[str]]:
    """確認權重已生成並複製為 pretrained.pth.tar，回傳 (步數, 權重資料夾清單, 權重檔案清單)。"""
    log_dirs = _get_candidate_log_dirs(data_name, diy_name, shots)
    chosen_step = req_step
    ckpt_dirs = []
    ckpt_files = []
    for d in log_dirs:
        if not os.path.isdir(d):
            continue
        target_ckpt = os.path.join(d, f"train-step{req_step}.pth.tar")
        if not os.path.exists(target_ckpt):
            step_files = glob.glob(os.path.join(d, "train-step*.pth.tar"))
            if step_files:
                steps = [int(re.search(r"train-step(\d+)", f).group(1)) for f in step_files]
                chosen_step = max(steps)
                target_ckpt = os.path.join(d, f"train-step{chosen_step}.pth.tar")
            elif os.path.exists(os.path.join(d, "pretrained.pth.tar")):
                target_ckpt = os.path.join(d, "pretrained.pth.tar")

        if os.path.exists(target_ckpt):
            pretrained_alias = os.path.join(d, "pretrained.pth.tar")
            if os.path.abspath(target_ckpt) != os.path.abspath(pretrained_alias):
                safe_copy(target_ckpt, pretrained_alias)
            abs_d = os.path.abspath(d)
            abs_target = os.path.abspath(target_ckpt)
            abs_alias = os.path.abspath(pretrained_alias)
            ckpt_dirs.append(abs_d)
            ckpt_files.extend([abs_target, abs_alias])
            log_and_print(f"[Auto-Ckpt] 已將訓練權重 {abs_target} 對接至 {abs_alias}", log_files)
    if not ckpt_dirs:
        raise FileNotFoundError(
            f"在 ./logs/{data_name}/ 下找不到任何 train-step*.pth.tar 權重檔！代表訓練階段未成功完成。"
        )
    return chosen_step, ckpt_dirs, ckpt_files


def main():
    parser = argparse.ArgumentParser(description="FoundAD 無GT高解析度訓練主程式 (Stage 1: Training)")
    parser.add_argument("--pool_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="./output/2_fab_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot")
    parser.add_argument("--category", type=str, default="auto")
    parser.add_argument("--app_train", type=str, default="train_dinov3")
    parser.add_argument("--diy_name", type=str, default="sim_escape")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--candidate_pool_size", type=int, default=16)
    parser.add_argument("--max_coarse_scan", type=int, default=250)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--rebuild_golden", action="store_true")
    parser.add_argument("--mock_from_mvtec", type=str, default="")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--max_train_tiles", type=int, default=48)
    parser.add_argument("--min_tile_trigger", type=int, default=1280)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--bg_std_thresh", type=float, default=1.5)
    parser.add_argument("--disable_tiling", action="store_true")
    args = parser.parse_args()

    if not args.category or args.category.lower() == "auto":
        if args.mock_from_mvtec:
            args.category = os.path.basename(os.path.normpath(args.mock_from_mvtec))
        else:
            args.category = os.path.basename(os.path.normpath(args.pool_dir))

    if args.data_name == "mvtec_sim_4shot" and args.category:
        args.data_name = f"mvtec_sim_{args.category}_{args.shots}shot"
    if args.diy_name == "sim_escape" and args.category:
        args.diy_name = f"sim_escape_{args.category}"

    now_tpe = datetime.now(TZ_TAIPEI)
    timestamp = now_tpe.strftime("%Y%m%d_%H%M%S_%f")[:19]
    base_out_dir = os.path.abspath(args.out_dir)
    safe_cat_tag = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{args.data_name}_{args.category}")
    run_out_dir = os.path.join(base_out_dir, f"train_{safe_cat_tag}_{timestamp}")
    os.makedirs(run_out_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)
    abs_datasets_dir = os.path.abspath(args.datasets_dir)

    shot_dir = os.path.join(abs_datasets_dir, args.data_name)

    fixed_log_txt = os.path.join(run_out_dir, "train_log.txt")
    cat_latest_log = os.path.join(base_out_dir, f"latest_train_log_{args.category}.txt")
    parent_latest_log = os.path.join(base_out_dir, "latest_train_log.txt")
    log_files = [fixed_log_txt, cat_latest_log, parent_latest_log]

    start_dt_str = now_tpe.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    log_and_print(
        f"\n{'='*75}\n"
        f"[Task 2 無GT訓練階段啟動時間] {start_dt_str}\n"
        f"[資料集與類別識別] DATA_NAME={args.data_name} | CATEGORY={args.category}\n"
        f"[本次訓練輸出與日誌目錄] {run_out_dir}\n"
        f"{'='*75}",
        log_files,
    )

    if args.mock_from_mvtec:
        prepare_mock_unlabeled_pool_from_mvtec(
            args.mock_from_mvtec,
            args.pool_dir,
            log_files,
            defect_limit=8,
            rebuild=args.rebuild_golden,
        )

    golden_filenames, num_train_samples, used_existing_golden, golden_select_sec, train_tile_sec = (
        load_or_prepare_golden_dataset(
            pool_dir=args.pool_dir,
            shot_dir=shot_dir,
            category=args.category,
            ckpt_path=args.ckpt,
            shots=args.shots,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_train_tiles=args.max_train_tiles,
            bg_std_thresh=args.bg_std_thresh,
            disable_tiling=args.disable_tiling,
            rebuild_golden=args.rebuild_golden,
            report_dir=run_out_dir,
            log_files=log_files,
            min_tile_trigger=args.min_tile_trigger,
            candidate_pool_size=args.candidate_pool_size,
            max_coarse_scan=args.max_coarse_scan,
        )
    )

    effective_batch_size = min(max(1, num_train_samples), args.train_batch_size)
    iters_per_epoch = max(1, int(np.ceil(num_train_samples / float(effective_batch_size))))
    effective_epochs = (
        max(400, int(np.ceil(args.ckpt_step / float(iters_per_epoch))))
        if iters_per_epoch > 1
        else args.ckpt_step
    )
    total_grad_updates = effective_epochs * iters_per_epoch

    train_meta_path = os.path.join(shot_dir, f"train_meta_{args.category}.json")
    prev_trained_samples = None
    if os.path.isfile(train_meta_path):
        try:
            with open(train_meta_path, "r", encoding="utf-8") as f:
                prev_meta = json.load(f)
            prev_trained_samples = prev_meta.get("num_train_samples", None)
        except Exception:
            pass

    existing_ckpt = find_existing_trained_ckpt(args.data_name, args.diy_name, effective_epochs, shots=args.shots)
    structure_changed = (prev_trained_samples is not None and int(prev_trained_samples) != int(num_train_samples))

    if existing_ckpt is None:
        if args.skip_train:
            log_and_print(
                f"[⚠️ 自動切換訓練] 雖設定了 --skip_train，但在 ./logs/{args.data_name}/ 下找不到任何權重檔，將自動開始訓練！",
                log_files,
            )
            args.skip_train = False
        else:
            log_and_print(
                f"[🚀 啟動新訓練] 在 ./logs/{args.data_name}/ 下尚未發現權重檔，自動開始執行模型訓練...",
                log_files,
            )
    elif structure_changed and not args.skip_train:
        log_and_print(
            f"[🔄 訓練集結構升級] 訓練集由舊版 {prev_trained_samples} 張變更為 {num_train_samples} 張影像，"
            f"將自動重新訓練一次以確保空間對齊精度！",
            log_files,
        )

    auto_skipped_train = False
    if (
        not args.skip_train
        and not args.force_train
        and not args.rebuild_golden
        and not structure_changed
        and existing_ckpt is not None
    ):
        args.skip_train = True
        auto_skipped_train = True
        log_and_print(
            f"[⚡ 防呆略過模型重訓] 偵測到已存在訓練完成的權重檔: {existing_ckpt}\n"
            f"                     自動跳過訓練階段！(若欲強制重新訓練請加上 --force_train)",
            log_files,
        )

    model_train_sec = 0.0
    if not args.skip_train:
        t_model_train_start = time.perf_counter()
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={effective_batch_size} "
            f"data.dataset=mvtec data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={shot_dir} "
            f"data.mvtec_classnames='[{args.category}]' "
            f"optimization.epochs={effective_epochs} "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        log_and_print(
            f"\n[Step 1] 執行 FoundAD 高解析度訓練 "
            f"(train_samples={num_train_samples}, batch_size={effective_batch_size}, "
            f"iters_per_epoch={iters_per_epoch}, epochs={effective_epochs}, total_updates={total_grad_updates}):",
            log_files,
        )
        run_cmd_and_log(train_cmd, log_files)
        model_train_sec = time.perf_counter() - t_model_train_start
    elif not auto_skipped_train:
        log_and_print("[⏱️ 計時] 已設定 --skip_train，跳過 Step 1 模型訓練階段 (0.00 秒)", log_files)

    total_train_time = golden_select_sec + train_tile_sec + model_train_sec

    actual_step, ckpt_dirs, ckpt_files = prepare_ckpt_for_demo(
        args.data_name,
        args.diy_name,
        effective_epochs if not args.skip_train else args.ckpt_step,
        log_files,
        shots=args.shots,
    )
    train_tiles_dir = os.path.abspath(os.path.join(shot_dir, "train", args.category))
    golden_backup_dir = os.path.abspath(os.path.join(shot_dir, "golden_full_res_backup", args.category))

    train_meta = {
        "data_name": args.data_name,
        "category": args.category,
        "diy_name": args.diy_name,
        "actual_step": actual_step,
        "golden_filenames": golden_filenames,
        "num_train_samples": num_train_samples,
        "candidate_pool_size": args.candidate_pool_size,
        "max_coarse_scan": args.max_coarse_scan,
        "golden_select_sec": golden_select_sec,
        "train_tile_sec": train_tile_sec,
        "model_train_sec": model_train_sec,
        "total_train_time": total_train_time,
        "ckpt_dirs": ckpt_dirs,
        "ckpt_files": ckpt_files,
        "run_out_dir": run_out_dir,
        "timestamp": timestamp,
        "timezone": "Asia/Taipei (UTC+8)",
    }
    os.makedirs(shot_dir, exist_ok=True)
    with open(train_meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)
    safe_copy(train_meta_path, os.path.join(run_out_dir, f"train_meta_{args.category}.json"))

    train_done_lines = [
        f"\n{'='*75}",
        f" [✅ 訓練階段完成 ({args.data_name} / {args.category}) — 時間統計與權重檔儲存位置]",
        f"{'-'*75}",
        f"  ⏱️ Auto-Golden 候選池切塊投票耗時        : {format_duration(golden_select_sec + train_tile_sec)}{' (Loaded Existing Golden)' if used_existing_golden else ''}",
        f"  ⏱️ Step 1 FoundAD 模型訓練耗時           : {format_duration(model_train_sec)}{' (Skipped Existing Ckpt)' if args.skip_train else ''}",
        f"  ⏱️ 整體訓練階段總時間                   : {format_duration(total_train_time)}",
        f"{'-'*75}",
        f"  🏆 最終勝出 Golden 原圖名單              : {', '.join(golden_filenames)}",
        f"  📁 Golden 訓練樣本資料夾                : {train_tiles_dir}",
        f"  📁 Golden 原圖備份資料夾                : {golden_backup_dir}",
        f"  📄 本次訓練專屬輸出資料夾               : {run_out_dir}",
        f"  📄 訓練階段 Meta 紀錄檔                 : {os.path.abspath(train_meta_path)}",
        f"  📦 訓練權重檔儲存資料夾                 : {ckpt_dirs[0]}",
    ]
    for cf in ckpt_files:
        train_done_lines.append(f"      └── {cf}")
    train_done_lines.append(f"{'='*75}")
    log_and_print("\n".join(train_done_lines), log_files)


if __name__ == "__main__":
    main()