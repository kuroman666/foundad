# 檔案位置: FoundAD/3_infer_fab_no_gt.py
# (全選覆蓋此檔：支援 --golden_src 與 --trained_ckpt 直接指定舊資料集 Golden Image 與訓練權重推論新資料集)
import shlex
import os
import re
import gc
import csv
import time
import glob
import json
import socket
import hashlib
import subprocess
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import importlib

TZ_TAIPEI = timezone(timedelta(hours=8))

_ag = importlib.import_module("1_auto_golden")
IMG_EXTENSIONS = _ag.IMG_EXTENSIONS
format_duration = _ag.format_duration
log_and_print = _ag.log_and_print
safe_rmtree = _ag.safe_rmtree
safe_copy = _ag.safe_copy
scan_pool_images = _ag.scan_pool_images
compute_pool_signature = _ag.compute_pool_signature
load_image_rgb_uint8 = _ag.load_image_rgb_uint8
get_axis_positions = _ag.get_axis_positions
is_background_tile = _ag.is_background_tile
prepare_tiled_golden_trainset = _ag.prepare_tiled_golden_trainset
load_or_prepare_golden_dataset = _ag.load_or_prepare_golden_dataset


def get_free_port() -> int:
    """由作業系統動態分配保證空閒的本地 Port，徹底避免 DDP Port 40112 衝突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_cmd_and_log(cmd: str, log_files: List[str], env: Optional[Dict] = None) -> str:
    """執行子程序指令，自動注入空閒 FAB_MASTER_PORT，即時在 Terminal 顯示並同步寫入 log.txt。"""
    header = f"\n[EXEC] {cmd}\n" + "-" * 60
    print(header, flush=True)
    output_lines = []
    run_env = (env or os.environ).copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if "FAB_MASTER_PORT" not in run_env:
        run_env["FAB_MASTER_PORT"] = str(get_free_port())

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


def load_image_bgr_uint8(img_path: str) -> np.ndarray:
    """安全讀取影像並統一轉為 (H, W, 3) uint8 BGR 格式供 OpenCV 畫圖與疊圖使用。"""
    rgb = load_image_rgb_uint8(img_path)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# =====================================================================
# [新增] 跨資料集推論：解析與對接指定的訓練權重檔 (--trained_ckpt)
# =====================================================================
def _resolve_custom_trained_ckpt_file(trained_ckpt: str, req_step: int) -> Tuple[str, int, str]:
    """
    解析使用者指定的 --trained_ckpt (可為 .pth.tar 檔案或權重目錄)，
    回傳 (權重檔絕對路徑, 對應步數 chosen_step, 模型前綴 model_prefix)。
    """
    ckpt_path = os.path.abspath(os.path.expanduser(trained_ckpt.strip()))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到指定的訓練權重路徑 (--trained_ckpt): {ckpt_path}")

    chosen_step = req_step
    target_file = ""

    if os.path.isfile(ckpt_path):
        target_file = ckpt_path
        m = re.search(r"train-step(\d+)\.pth", os.path.basename(target_file))
        if m:
            chosen_step = int(m.group(1))
        else:
            parent_dir = os.path.dirname(target_file)
            step_files = glob.glob(os.path.join(parent_dir, "train-step*.pth.tar"))
            if step_files:
                steps = [int(re.search(r"train-step(\d+)", f).group(1)) for f in step_files]
                if req_step in steps:
                    chosen_step = req_step
                else:
                    chosen_step = max(steps)
    else:
        exact_candidates = glob.glob(
            os.path.join(ckpt_path, "**", f"train-step{req_step}.pth.tar"), recursive=True
        )
        if exact_candidates:
            target_file = sorted(exact_candidates)[-1]
            chosen_step = req_step
        else:
            step_files = glob.glob(os.path.join(ckpt_path, "**", "train-step*.pth.tar"), recursive=True)
            if step_files:
                step_files.sort(key=lambda p: int(re.search(r"train-step(\d+)", p).group(1)))
                target_file = step_files[-1]
                chosen_step = int(re.search(r"train-step(\d+)", target_file).group(1))
            else:
                alias_files = glob.glob(os.path.join(ckpt_path, "**", "pretrained.pth.tar"), recursive=True)
                if not alias_files:
                    alias_files = glob.glob(os.path.join(ckpt_path, "**", "*.pth*"), recursive=True)
                if alias_files:
                    target_file = sorted(alias_files)[-1]

    if not target_file or not os.path.isfile(target_file):
        raise FileNotFoundError(f"在指定的 --trained_ckpt ({ckpt_path}) 中找不到任何有效的 .pth.tar 權重檔！")

    parent_name = os.path.basename(os.path.dirname(target_file))
    model_prefix = "dinov3"
    for prefix_cand in ("dinov3", "dinov2", "dino", "vit"):
        if parent_name.lower().startswith(prefix_cand):
            model_prefix = prefix_cand
            break

    return os.path.abspath(target_file), chosen_step, model_prefix


def prepare_ckpt_for_demo(
    data_name: str,
    diy_name: str,
    req_step: int,
    log_files: List[str],
    trained_ckpt: str = "",
) -> Tuple[int, List[str], List[str]]:
    """
    確認權重已生成並複製為 pretrained.pth.tar 與 train-step{chosen_step}.pth.tar。
    若指定了 trained_ckpt，則自動將該權重檔對接至當前 data_name / diy_name 的推論目錄。
    """
    if trained_ckpt and trained_ckpt.strip():
        src_ckpt_file, chosen_step, model_prefix = _resolve_custom_trained_ckpt_file(trained_ckpt, req_step)
        existing_dirs = [
            d for d in (glob.glob(f"./logs/{data_name}/*{diy_name}*") or glob.glob(f"./logs/{data_name}/*"))
            if os.path.isdir(d)
        ]
        default_target_dir = os.path.join("./logs", data_name, f"{model_prefix}{diy_name}")
        if default_target_dir not in existing_dirs:
            existing_dirs.insert(0, default_target_dir)
        dinov3_target_dir = os.path.join("./logs", data_name, f"dinov3{diy_name}")
        if dinov3_target_dir not in existing_dirs:
            existing_dirs.append(dinov3_target_dir)

        ckpt_dirs = []
        ckpt_files = []
        for d in existing_dirs:
            os.makedirs(d, exist_ok=True)
            dst_step_ckpt = os.path.join(d, f"train-step{chosen_step}.pth.tar")
            dst_alias_ckpt = os.path.join(d, "pretrained.pth.tar")
            if os.path.abspath(src_ckpt_file) != os.path.abspath(dst_step_ckpt):
                safe_copy(src_ckpt_file, dst_step_ckpt)
            if os.path.abspath(src_ckpt_file) != os.path.abspath(dst_alias_ckpt):
                safe_copy(src_ckpt_file, dst_alias_ckpt)
            abs_d = os.path.abspath(d)
            ckpt_dirs.append(abs_d)
            ckpt_files.extend([os.path.abspath(dst_step_ckpt), os.path.abspath(dst_alias_ckpt)])

        log_and_print(
            f"[Custom-Ckpt] 已載入指定訓練權重: {src_ckpt_file} (step={chosen_step})\n"
            f"              並自動對接至推論目錄: {ckpt_dirs[0]}",
            log_files,
        )
        return chosen_step, ckpt_dirs, ckpt_files

    log_dirs = glob.glob(f"./logs/{data_name}/*{diy_name}*") or glob.glob(f"./logs/{data_name}/*")
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
            f"在 ./logs/{data_name}/ 下找不到任何 train-step*.pth.tar 權重檔！"
            f"請先執行 2_train_fab_no_gt.py 完成訓練，或透過 --trained_ckpt 指定舊資料集訓練好的權重檔。"
        )
    return chosen_step, ckpt_dirs, ckpt_files


# =====================================================================
# [新增] 跨資料集推論：解析與載入指定的 Golden Image (--golden_src)
# =====================================================================
def resolve_custom_golden_source(
    golden_src: str,
    category: str,
    fallback_pool_dir: str = "",
) -> Tuple[List[str], Optional[str]]:
    """
    解析使用者指定的 --golden_src，回傳 (Golden 全解析度原圖絕對路徑清單, 既有訓練圖塊目錄或 None)。
    支援 4 種輸入格式：
    1. 舊資料集的 shot_dir 根目錄 (如 ./datasets/wafer_TMQB28_4shot)
    2. 含有 Golden 圖片的任意資料夾路徑 (如 .../golden_full_res_backup/wafer 或自訂資料夾)
    3. selected_4_golden_images.txt 紀錄檔或純文字路徑清單 (.txt)
    4. 以逗號 (,) 分隔的多張圖片路徑字串
    """
    raw_spec = golden_src.strip()
    if not raw_spec:
        raise ValueError("--golden_src 不可為空字串！")

    # 格式 4：以逗號分隔的多張圖片路徑，或單一圖片檔案
    if "," in raw_spec or (os.path.isfile(raw_spec) and raw_spec.lower().endswith(IMG_EXTENSIONS)):
        parts = [p.strip() for p in raw_spec.split(",") if p.strip()]
        resolved = []
        for p in parts:
            cand = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(cand) and fallback_pool_dir:
                cand2 = os.path.abspath(os.path.join(fallback_pool_dir, p))
                if os.path.isfile(cand2):
                    cand = cand2
            if not os.path.isfile(cand):
                raise FileNotFoundError(f"找不到指定的 Golden 圖片檔案: {p}")
            resolved.append(cand)
        return resolved, None

    src_path = os.path.abspath(os.path.expanduser(raw_spec))
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到指定的 Golden 來源路徑 (--golden_src): {src_path}")

    # 格式 3：.txt 紀錄檔 (例如 selected_4_golden_images.txt 或每行一個路徑的清單)
    if os.path.isfile(src_path) and src_path.lower().endswith(".txt"):
        txt_dir = os.path.dirname(src_path)
        parsed_rel_names = []
        direct_paths = []
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("==="):
                    continue
                m = re.search(r":\s*([^|]+?)\s*(?:\||$)", s)
                if m and "Golden #" in s:
                    parsed_rel_names.append(m.group(1).strip())
                elif s.lower().endswith(IMG_EXTENSIONS):
                    direct_paths.append(s)

        resolved = []
        for p in direct_paths + parsed_rel_names:
            for base_d in ("", txt_dir, fallback_pool_dir):
                cand = os.path.abspath(os.path.join(base_d, p)) if base_d else os.path.abspath(p)
                if os.path.isfile(cand):
                    resolved.append(cand)
                    break
        if resolved:
            return resolved, None

        # 若 .txt 內記載的是舊盲池相對路徑，退回檢查同目錄下的 golden_full_res_backup
        src_path = txt_dir

    # 格式 1 & 2：資料夾路徑 (可為舊 shot_dir 或任意圖片資料夾)
    if os.path.isdir(src_path):
        existing_train_tiles_dir = None
        train_cat_dir = os.path.join(src_path, "train", category)
        if not os.path.isdir(train_cat_dir) and os.path.isdir(os.path.join(src_path, "train")):
            sub_trains = [
                os.path.join(src_path, "train", d)
                for d in sorted(os.listdir(os.path.join(src_path, "train")))
                if os.path.isdir(os.path.join(src_path, "train", d))
            ]
            if sub_trains:
                train_cat_dir = sub_trains[0]
        if os.path.isdir(train_cat_dir) and scan_pool_images(train_cat_dir):
            existing_train_tiles_dir = train_cat_dir

        # 優先尋找 golden_full_res_backup/<category> 內的完整解析度 Golden 原圖
        backup_cat_dir = os.path.join(src_path, "golden_full_res_backup", category)
        if not os.path.isdir(backup_cat_dir) and os.path.isdir(os.path.join(src_path, "golden_full_res_backup")):
            sub_backups = [
                os.path.join(src_path, "golden_full_res_backup", d)
                for d in sorted(os.listdir(os.path.join(src_path, "golden_full_res_backup")))
                if os.path.isdir(os.path.join(src_path, "golden_full_res_backup", d))
            ]
            if sub_backups:
                backup_cat_dir = sub_backups[0]
        if os.path.isdir(backup_cat_dir):
            backup_imgs = scan_pool_images(backup_cat_dir)
            if backup_imgs:
                return backup_imgs, existing_train_tiles_dir

        # 若使用者直接傳入 train/<category> 目錄，嘗試往上找其對應的 golden_full_res_backup
        parent_dir = os.path.dirname(src_path)
        grandparent_dir = os.path.dirname(parent_dir)
        if os.path.basename(parent_dir) == "train":
            alt_backup = os.path.join(grandparent_dir, "golden_full_res_backup", os.path.basename(src_path))
            if os.path.isdir(alt_backup):
                backup_imgs = scan_pool_images(alt_backup)
                if backup_imgs:
                    return backup_imgs, src_path

        # 直接掃描指定資料夾內的圖片 (優先取非碎片 _tile_y 的原圖)
        all_found = scan_pool_images(src_path)
        if all_found:
            non_tile_imgs = [p for p in all_found if "_tile_y" not in os.path.basename(p)]
            chosen_imgs = non_tile_imgs if non_tile_imgs else all_found
            return chosen_imgs, existing_train_tiles_dir

    raise FileNotFoundError(f"在指定的 --golden_src ({golden_src}) 中找不到任何有效的 Golden 圖片！")


def setup_custom_golden_for_infer(
    golden_src: str,
    pool_dir: str,
    shot_dir: str,
    category: str,
    ckpt_path: str,
    shots: int,
    tile_size: int,
    tile_overlap: int,
    max_train_tiles: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    min_tile_trigger: int,
    run_out_dir: str,
    log_files: List[str],
) -> Tuple[List[str], List[str]]:
    """
    將使用者指定的 Golden Image 同步配置到當前 shot_dir，並檢查這些 Golden Image 是否屬於外部舊資料集：
    - 回傳 (golden_filenames, external_golden_paths)
    - 若 Golden Image 來自外部舊資料集 (不在本次 --pool_dir 內)，將透過 external_golden_paths
      注入推論圖塊池以計算 Golden 空間容差遮罩與動態門檻基準，但不混入新資料集的測試排名報表中。
    """
    custom_golden_paths, existing_train_tiles_dir = resolve_custom_golden_source(
        golden_src=golden_src,
        category=category,
        fallback_pool_dir=pool_dir,
    )

    cat_train_dir = os.path.join(shot_dir, "train", category)
    full_backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    os.makedirs(cat_train_dir, exist_ok=True)
    os.makedirs(full_backup_dir, exist_ok=True)

    # 1. 將指定的 Golden 全解析度原圖同步至 golden_full_res_backup/<category>
    if os.path.abspath(os.path.dirname(custom_golden_paths[0])) != os.path.abspath(full_backup_dir):
        safe_rmtree(full_backup_dir)
        os.makedirs(full_backup_dir, exist_ok=True)
        for i, src_p in enumerate(custom_golden_paths):
            ext = os.path.splitext(src_p)[1].lower()
            dst_p = os.path.join(full_backup_dir, f"{i:03d}_{os.path.basename(src_p)}" if not re.match(r"^\d{3}", os.path.basename(src_p)) else os.path.basename(src_p))
            safe_copy(src_p, dst_p)

    # 2. 確保 shot_dir/train/<category> 具備對應的訓練圖塊或原圖
    if existing_train_tiles_dir and os.path.abspath(existing_train_tiles_dir) != os.path.abspath(cat_train_dir):
        safe_rmtree(cat_train_dir)
        os.makedirs(cat_train_dir, exist_ok=True)
        for f in sorted(os.listdir(existing_train_tiles_dir)):
            if f.lower().endswith(IMG_EXTENSIONS):
                safe_copy(os.path.join(existing_train_tiles_dir, f), os.path.join(cat_train_dir, f))
    elif not existing_train_tiles_dir:
        safe_rmtree(cat_train_dir)
        os.makedirs(cat_train_dir, exist_ok=True)
        for i, src_p in enumerate(custom_golden_paths):
            ext = os.path.splitext(src_p)[1].lower()
            safe_copy(src_p, os.path.join(cat_train_dir, f"{i:03d}{ext}"))
        if not disable_tiling:
            try:
                sample_img = load_image_rgb_uint8(custom_golden_paths[0])
                sh, sw = sample_img.shape[:2]
                del sample_img
                if max(sh, sw) > max(tile_size, min_tile_trigger):
                    prepare_tiled_golden_trainset(
                        shot_dir=shot_dir,
                        category=category,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        max_train_tiles=max_train_tiles,
                        bg_std_thresh=bg_std_thresh,
                        log_files=log_files,
                        ckpt_path=ckpt_path,
                        report_dir=run_out_dir,
                        min_tile_trigger=min_tile_trigger,
                        shots=len(custom_golden_paths),
                    )
            except Exception as e:
                log_and_print(f"[Custom-Golden] 備註：略過訓練圖塊重建，直接沿用原圖 ({e})", log_files)

    # 3. 判斷指定的 Golden Image 是否已在新的 --pool_dir 內，或屬於跨資料集外部 Golden
    pool_imgs = scan_pool_images(pool_dir)
    pool_abs_map = {os.path.abspath(p): os.path.relpath(p, pool_dir).replace("\\", "/") for p in pool_imgs}
    pool_base_map = {os.path.basename(p): os.path.relpath(p, pool_dir).replace("\\", "/") for p in pool_imgs}

    golden_filenames = []
    external_golden_paths = []
    for idx, g_path in enumerate(custom_golden_paths):
        abs_g = os.path.abspath(g_path)
        base_g = os.path.basename(g_path)
        if abs_g in pool_abs_map:
            golden_filenames.append(pool_abs_map[abs_g])
        elif base_g in pool_base_map and not re.match(r"^\d{3}\.[a-zA-Z0-9]+$", base_g):
            golden_filenames.append(pool_base_map[base_g])
        else:
            ext_key = f"[GOLDEN_REF]/{idx:02d}_{base_g}"
            golden_filenames.append(ext_key)
            external_golden_paths.append(abs_g)

    record_txt = os.path.join(run_out_dir, "selected_custom_golden_images.txt")
    with open(record_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Custom Specified Golden Images (--golden_src: {golden_src}) ===\n")
        for i, (g_p, g_key) in enumerate(zip(custom_golden_paths, golden_filenames), 1):
            f.write(f"Golden #{i:02d}: {g_p} -> Manifest_Key: {g_key}\n")

    log_and_print(
        f"[Custom-Golden] 已載入指定的 {len(custom_golden_paths)} 張 Golden Image 來源: {golden_src}\n"
        f"                ├── 清單: {', '.join([os.path.basename(p) for p in custom_golden_paths])}\n"
        f"                └── 跨資料集外部基準圖數量: {len(external_golden_paths)} 張 (將自動注入推論池以校準動態門檻與結構容差)",
        log_files,
    )
    return golden_filenames, external_golden_paths


def load_golden_filenames_for_infer(
    pool_dir: str,
    shot_dir: str,
    category: str,
    ckpt_path: str,
    shots: int,
    tile_size: int,
    tile_overlap: int,
    max_train_tiles: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    min_tile_trigger: int,
    run_out_dir: str,
    log_files: List[str],
    golden_src: str = "",
) -> Tuple[List[str], List[str], Optional[int], float]:
    """
    1. 若有指定 --golden_src，直接載入指定的 Golden Image (絕不對新資料集重跑 Auto-Golden)。
    2. 否則優先從 2_train_fab_no_gt.py 產生的 train_meta_{category}.json 讀取 golden_filenames；
       若連舊紀錄也無，才從 auto_golden 快取或備份資料夾載入。
    """
    if golden_src and golden_src.strip():
        golden_filenames, external_golden_paths = setup_custom_golden_for_infer(
            golden_src=golden_src,
            pool_dir=pool_dir,
            shot_dir=shot_dir,
            category=category,
            ckpt_path=ckpt_path,
            shots=shots,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_train_tiles=max_train_tiles,
            bg_std_thresh=bg_std_thresh,
            disable_tiling=disable_tiling,
            min_tile_trigger=min_tile_trigger,
            run_out_dir=run_out_dir,
            log_files=log_files,
        )
        return golden_filenames, external_golden_paths, None, 0.0

    train_meta_path = os.path.join(shot_dir, f"train_meta_{category}.json")
    if os.path.isfile(train_meta_path):
        try:
            with open(train_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            golden_filenames = meta.get("golden_filenames", [])
            saved_step = meta.get("actual_step", None)
            total_train_time = float(meta.get("total_train_time", 0.0))
            if golden_filenames:
                purity_src = os.path.join(shot_dir, "golden_tile_purity_report.txt")
                if os.path.isfile(purity_src):
                    safe_copy(purity_src, os.path.join(run_out_dir, "golden_tile_purity_report.txt"))

                # 若使用者未傳 --golden_src，但把舊的 shot_dir 直接拿來推論新 pool_dir，
                # 自動檢查 backup 內的 Golden 原圖是否不在新 pool_dir 中並自動補入外部基準
                external_golden_paths = []
                pool_rel_set = {
                    os.path.relpath(p, pool_dir).replace("\\", "/") for p in scan_pool_images(pool_dir)
                }
                pool_base_set = {os.path.basename(p) for p in pool_rel_set}
                if not any((gf in pool_rel_set or os.path.basename(gf) in pool_base_set) for gf in golden_filenames):
                    backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
                    if os.path.isdir(backup_dir):
                        b_imgs = scan_pool_images(backup_dir)
                        if b_imgs:
                            external_golden_paths = [os.path.abspath(p) for p in b_imgs]
                            golden_filenames = [
                                f"[GOLDEN_REF]/{idx:02d}_{os.path.basename(p)}"
                                for idx, p in enumerate(external_golden_paths)
                            ]
                            log_and_print(
                                f"[Golden-AutoRef] 偵測到新測試集不含原訓練 Golden 圖片，"
                                f"已自動從備份載入 {len(external_golden_paths)} 張舊 Golden 原圖作為跨資料集基準！",
                                log_files,
                            )

                log_and_print(
                    f"[Golden-Meta] 已從 {os.path.abspath(train_meta_path)} 載入 {len(golden_filenames)} 張 Golden 清單",
                    log_files,
                )
                return golden_filenames, external_golden_paths, saved_step, total_train_time
        except Exception:
            pass

    golden_filenames, _, _, golden_select_sec, train_tile_sec = load_or_prepare_golden_dataset(
        pool_dir=pool_dir,
        shot_dir=shot_dir,
        category=category,
        ckpt_path=ckpt_path,
        shots=shots,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        max_train_tiles=max_train_tiles,
        bg_std_thresh=bg_std_thresh,
        disable_tiling=disable_tiling,
        min_tile_trigger=min_tile_trigger,
        rebuild_golden=False,
        report_dir=run_out_dir,
        log_files=log_files,
    )
    return golden_filenames, [], None, (golden_select_sec + train_tile_sec)


# =====================================================================
# 測試池高解析度切塊 (Tiling) 與無縫縫合 (Feathering Stitching) 模組
# =====================================================================
def _tile_single_image_to_pool(
    fpath: str,
    rel_fname: str,
    stem: str,
    tiled_pool_dir: str,
    tile_size: int,
    stride: int,
    min_tile_trigger: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    is_external_golden: bool = False,
) -> Tuple[Dict, int, int, bool]:
    """將單張原圖切分為推論圖塊並存入 tiled_pool_dir。"""
    img = load_image_rgb_uint8(fpath)
    h, w = img.shape[:2]

    if disable_tiling or (max(h, w) <= min_tile_trigger) or (h <= tile_size and w <= tile_size):
        ys, xs = [0], [0]
        eff_tile_h, eff_tile_w = h, w
        is_1to1 = True
    else:
        ys = get_axis_positions(h, tile_size, stride)
        xs = get_axis_positions(w, tile_size, stride)
        eff_tile_h, eff_tile_w = tile_size, tile_size
        is_1to1 = False

    tiles_info = []
    skipped_bg = 0
    for y in ys:
        for x in xs:
            ch = min(eff_tile_h, h - y)
            cw = min(eff_tile_w, w - x)
            crop = img[y:y + ch, x:x + cw]

            if is_background_tile(crop, bg_std_thresh) and (len(ys) * len(xs) > 1):
                skipped_bg += 1
                continue

            tile_stem = f"{stem}__tile_y{y:04d}_x{x:04d}_h{ch:04d}_w{cw:04d}"
            tile_fname = f"{tile_stem}.png"
            Image.fromarray(crop).save(os.path.join(tiled_pool_dir, tile_fname))
            tiles_info.append(
                {
                    "tile_stem": tile_stem,
                    "tile_fname": tile_fname,
                    "y": int(y),
                    "x": int(x),
                    "h": int(ch),
                    "w": int(cw),
                }
            )

    if not tiles_info:
        cy, cx = max(0, (h - eff_tile_h) // 2), max(0, (w - eff_tile_w) // 2)
        ch, cw = min(eff_tile_h, h - cy), min(eff_tile_w, w - cx)
        crop = img[cy:cy + ch, cx:cx + cw]
        tile_stem = f"{stem}__tile_y{cy:04d}_x{cx:04d}_h{ch:04d}_w{cw:04d}"
        tile_fname = f"{tile_stem}.png"
        Image.fromarray(crop).save(os.path.join(tiled_pool_dir, tile_fname))
        tiles_info.append(
            {
                "tile_stem": tile_stem,
                "tile_fname": tile_fname,
                "y": int(cy),
                "x": int(cx),
                "h": int(ch),
                "w": int(cw),
            }
        )

    del img
    entry = {
        "stem": stem,
        "path": os.path.abspath(fpath),
        "height": int(h),
        "width": int(w),
        "is_external_golden": bool(is_external_golden),
        "tiles": tiles_info,
    }
    return entry, len(tiles_info), skipped_bg, is_1to1


def prepare_tiled_test_pool(
    pool_dir: str,
    tiled_pool_dir: str,
    manifest_save_dir: str,
    tile_size: int,
    tile_overlap: int,
    min_tile_trigger: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    pool_signature: str,
    tile_config_key: str,
    cached_manifest_path: str,
    log_files: List[str],
    external_golden_paths: Optional[List[str]] = None,
) -> Dict:
    """
    遞迴搜尋測試盲池 (含所有子資料夾) 中的所有圖片，並切塊供模型推論。
    若存在跨資料集指定的外部 Golden 原圖 (external_golden_paths)，自動一併切塊注入推論池以供基準校準。
    """
    safe_rmtree(tiled_pool_dir)
    os.makedirs(tiled_pool_dir, exist_ok=True)
    stride = max(1, tile_size - tile_overlap)
    img_paths = scan_pool_images(pool_dir)
    if not img_paths:
        raise FileNotFoundError(f"在 {pool_dir} (含子資料夾) 中找不到任何圖片！")

    manifest = {
        "pool_signature": pool_signature,
        "tile_config_key": tile_config_key,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "min_tile_trigger": min_tile_trigger,
        "images": {},
    }
    total_tiles = 0
    skipped_bg_tiles = 0
    full_1to1_imgs = 0
    tiled_highres_imgs = 0

    for fpath in img_paths:
        rel_fname = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_fname)[0]
        stem = re.sub(r"[\\/]+", "__", rel_no_ext)
        entry, n_t, n_skip, is_1to1 = _tile_single_image_to_pool(
            fpath=fpath,
            rel_fname=rel_fname,
            stem=stem,
            tiled_pool_dir=tiled_pool_dir,
            tile_size=tile_size,
            stride=stride,
            min_tile_trigger=min_tile_trigger,
            bg_std_thresh=bg_std_thresh,
            disable_tiling=disable_tiling,
            is_external_golden=False,
        )
        manifest["images"][rel_fname] = entry
        total_tiles += n_t
        skipped_bg_tiles += n_skip
        if is_1to1:
            full_1to1_imgs += 1
        else:
            tiled_highres_imgs += 1
        gc.collect()

    ext_golden_tiles = 0
    if external_golden_paths:
        for g_idx, g_path in enumerate(external_golden_paths):
            base_g = os.path.basename(g_path)
            rel_fname = f"[GOLDEN_REF]/{g_idx:02d}_{base_g}"
            base_no_ext = re.sub(r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(base_g)[0])
            stem = f"GOLDEN_REF_{g_idx:02d}__{base_no_ext}"
            entry, n_t, n_skip, _ = _tile_single_image_to_pool(
                fpath=g_path,
                rel_fname=rel_fname,
                stem=stem,
                tiled_pool_dir=tiled_pool_dir,
                tile_size=tile_size,
                stride=stride,
                min_tile_trigger=min_tile_trigger,
                bg_std_thresh=bg_std_thresh,
                disable_tiling=disable_tiling,
                is_external_golden=True,
            )
            manifest["images"][rel_fname] = entry
            total_tiles += n_t
            ext_golden_tiles += n_t
            skipped_bg_tiles += n_skip
        gc.collect()

    manifest["total_tiles"] = total_tiles
    manifest["num_images"] = len(img_paths)
    manifest["num_external_golden"] = len(external_golden_paths) if external_golden_paths else 0
    manifest_path = os.path.join(manifest_save_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if cached_manifest_path:
        os.makedirs(os.path.dirname(cached_manifest_path), exist_ok=True)
        safe_copy(manifest_path, cached_manifest_path)

    ext_note = (
        f", 另含 {len(external_golden_paths)} 張跨資料集 Golden 基準圖 [{ext_golden_tiles} 塊]"
        if external_golden_paths
        else ""
    )
    log_and_print(
        f"[Tiling-Test] 已遞迴將 {len(img_paths)} 張新測試圖準備為 {total_tiles} 張推論圖塊 "
        f"(1:1整張推論={full_1to1_imgs}張 [<= {min_tile_trigger}px], 大圖切塊={tiled_highres_imgs}張"
        f"{ext_note}, 自動跳過 {skipped_bg_tiles} 張純背景角落塊) -> {os.path.abspath(tiled_pool_dir)}",
        log_files,
    )
    return manifest


def build_2d_feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """建立 2D 餘弦平滑權重遮罩 (Raised-Cosine Window)，消除重疊切塊拼接時的邊界接縫。"""
    def _1d_ramp(length: int, ov: int) -> np.ndarray:
        w1d = np.ones(length, dtype=np.float32)
        ramp = min(max(ov, 16), length // 3)
        if ramp > 1:
            t = np.linspace(0.0, np.pi, ramp, dtype=np.float32)
            rising = 0.1 + 0.9 * (0.5 * (1.0 - np.cos(t)))
            w1d[:ramp] = rising
            w1d[-ramp:] = rising[::-1]
        return w1d

    wy = _1d_ramp(h, overlap)
    wx = _1d_ramp(w, overlap)
    return np.outer(wy, wx).astype(np.float32)


def find_tile_npy(tile_amap_dir: str, tile_stem: str, tile_fname: str) -> str:
    """容錯尋找 AD.py 輸出的子圖塊 .npy 熱力圖檔案。"""
    candidates = [
        os.path.join(tile_amap_dir, f"{tile_stem}.npy"),
        os.path.join(tile_amap_dir, f"{tile_fname}.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    matches = glob.glob(os.path.join(tile_amap_dir, f"*{tile_stem}*.npy"), recursive=True)
    return matches[0] if matches else ""


def is_tiled_png_pool_valid(
    tiled_pool_dir: str,
    cached_manifest_path: str,
    pool_signature: str,
    tile_config_key: str,
) -> Optional[Dict]:
    """檢查既有測試池切塊 PNG 與 cached_tile_manifest.json 是否完整且指紋相符。"""
    if not os.path.isdir(tiled_pool_dir) or not os.path.isfile(cached_manifest_path):
        return None
    try:
        with open(cached_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (
            manifest.get("pool_signature") != pool_signature
            or manifest.get("tile_config_key") != tile_config_key
            or not manifest.get("images")
        ):
            return None
        for meta in manifest["images"].values():
            if not os.path.exists(meta.get("path", "")):
                return None
            for t in meta.get("tiles", []):
                if not os.path.exists(os.path.join(tiled_pool_dir, t["tile_fname"])):
                    return None
        return manifest
    except Exception:
        return None


def stitch_single_image_amap(
    img_meta: Dict,
    tile_amap_dir: str,
    tile_overlap: int,
    delete_tiles: bool = True,
) -> np.ndarray:
    """
    將單張大圖的所有子圖塊 .npy 直接於記憶體 (RAM) 內無縫縫合回原圖完整解析度 (H, W)，
    並在拼回整張圖後立刻自動銷毀該圖所有切成碎片的 Tile .npy 檔案。
    """
    h, w = img_meta["height"], img_meta["width"]
    tiles = img_meta["tiles"]
    canvas_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    weight_cache: Dict[Tuple[int, int], np.ndarray] = {}
    consumed_tile_npys: List[str] = []
    is_single_full_tile = len(tiles) == 1 and tiles[0]["h"] == h and tiles[0]["w"] == w

    for t in tiles:
        npy_path = find_tile_npy(tile_amap_dir, t["tile_stem"], t["tile_fname"])
        if not npy_path:
            continue
        amap = np.load(npy_path).astype(np.float32)
        consumed_tile_npys.append(npy_path)
        amap = np.squeeze(amap)
        if amap.ndim != 2:
            continue

        th, tw = t["h"], t["w"]
        if amap.shape[0] != th or amap.shape[1] != tw:
            amap = cv2.resize(amap, (tw, th), interpolation=cv2.INTER_CUBIC)

        if is_single_full_tile:
            w2d = np.ones((th, tw), dtype=np.float32)
        else:
            key = (th, tw)
            if key not in weight_cache:
                weight_cache[key] = build_2d_feather_weight(th, tw, tile_overlap)
            w2d = weight_cache[key]

        y, x = t["y"], t["x"]
        canvas_sum[y:y + th, x:x + tw] += amap * w2d
        weight_sum[y:y + th, x:x + tw] += w2d

    valid_mask = weight_sum > 1e-6
    full_amap = np.zeros((h, w), dtype=np.float32)
    full_amap[valid_mask] = canvas_sum[valid_mask] / weight_sum[valid_mask]
    if np.any(valid_mask) and not np.all(valid_mask):
        baseline = float(np.percentile(full_amap[valid_mask], 5.0))
        full_amap[~valid_mask] = baseline

    if delete_tiles:
        for p in consumed_tile_npys:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    return full_amap


# =====================================================================
# 防誤報核心演算法：晶圓圓周邊緣抑制 + 微瑕疵白頂帽增強 + 動態門檻對齊
# =====================================================================
def detect_circular_wafer_geometry(
    img_path: str,
    h: int,
    w: int,
    wafer_edge_mode: str,
    category: str,
) -> Tuple[bool, float, float, float]:
    """
    偵測影像是否為圓形晶圓/圓形工件，並回傳 (是否啟用圓周抑制, 圓心 cy, 圓心 cx, 半徑 R)。
    """
    mode = wafer_edge_mode.lower().strip()
    if mode == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5

    cat_lower = category.lower()
    is_non_circular_cat = any(
        k in cat_lower for k in ["capsule", "pill", "toothbrush", "transistor", "screw", "wood", "zipper"]
    )
    is_circular_cat = (not is_non_circular_cat) and any(
        k in cat_lower for k in ["wafer", "bottle", "cable", "coin", "lens", "circle"]
    )

    scale = 256.0 / max(h, w)
    sh, sw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
    try:
        full_rgb = load_image_rgb_uint8(img_path)
        small_rgb = cv2.resize(full_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        del full_rgb
        gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        corner_sz = max(4, min(sh, sw) // 10)
        corners = np.concatenate(
            [
                blur[:corner_sz, :corner_sz].reshape(-1),
                blur[:corner_sz, -corner_sz:].reshape(-1),
                blur[-corner_sz:, :corner_sz].reshape(-1),
                blur[-corner_sz:, -corner_sz:].reshape(-1),
            ]
        )
        center_patch = blur[sh // 3:2 * sh // 3, sw // 3:2 * sw // 3].reshape(-1)
        corner_mean = float(np.mean(corners))
        center_mean = float(np.mean(center_patch))
        corner_std = float(np.std(corners))

        if abs(center_mean - corner_mean) > 12.0:
            thresh_type = cv2.THRESH_BINARY if center_mean > corner_mean else cv2.THRESH_BINARY_INV
            _, bw = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = float(cv2.contourArea(c))
                if area > 0.18 * (sh * sw):
                    _, _, bw_rect, bh_rect = cv2.boundingRect(c)
                    aspect_ratio = float(min(bw_rect, bh_rect)) / max(float(max(bw_rect, bh_rect)), 1e-6)

                    perimeter = float(cv2.arcLength(c, True))
                    geom_circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6)

                    (cx_s, cy_s), r_s = cv2.minEnclosingCircle(c)
                    circle_area = float(np.pi * (r_s ** 2) + 1e-6)
                    circularity_fill = area / circle_area
                    circularity_score = max(geom_circularity, circularity_fill)

                    if (
                        aspect_ratio >= 0.85
                        and circularity_score >= 0.82
                        and 0.78 <= circularity_fill <= 1.15
                        and abs(cx_s - sw * 0.5) < 0.18 * sw
                        and abs(cy_s - sh * 0.5) < 0.18 * sh
                        and r_s <= 0.58 * max(sh, sw)
                    ):
                        if mode == "on" or (not is_non_circular_cat and (is_circular_cat or corner_std < 28.0)):
                            cy_full = float(cy_s / scale)
                            cx_full = float(cx_s / scale)
                            r_full = float(r_s / scale)
                            return True, cy_full, cx_full, r_full
    except Exception:
        pass

    if mode == "on" or is_circular_cat:
        return True, h * 0.5, w * 0.5, min(h, w) * 0.48
    return False, h * 0.5, w * 0.5, min(h, w) * 0.5


def refine_amap_for_wafer_and_noise(
    full_amap: np.ndarray,
    img_path: str,
    category: str,
    border_ratio: float,
    wafer_edge_mode: str,
    wafer_edge_margin: float,
    speckle_filter_ratio: float,
    golden_spatial_ref: Optional[np.ndarray] = None,
    golden_tol_weight: float = 0.15,
    micro_defect_mode: bool = False,
    micro_boost_weight: float = 0.35,
) -> np.ndarray:
    """
    四合一防誤報熱力圖純化器 (支援 --micro_defect_mode 微瑕疵白頂帽局部峰值增強)。
    """
    h, w = full_amap.shape[:2]
    work_map = full_amap.astype(np.float32).copy()
    img_median = float(np.median(work_map))

    eff_tol_weight = min(golden_tol_weight, 0.08) if micro_defect_mode else golden_tol_weight
    if golden_spatial_ref is not None and eff_tol_weight > 0:
        if golden_spatial_ref.shape[:2] != (h, w):
            g_ref = cv2.resize(golden_spatial_ref, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            g_ref = golden_spatial_ref
        g_med = float(np.median(g_ref))
        structural_excess = np.maximum(0.0, g_ref - g_med)
        work_map = img_median + np.maximum(0.0, (work_map - img_median) - eff_tol_weight * structural_excess)

    if micro_defect_mode:
        smooth_3x3 = cv2.GaussianBlur(work_map, (3, 3), 0)
        k_bg = max(15, min(35, int(round(max(h, w) * 0.006)) | 1))
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg, k_bg))
        local_bg = cv2.morphologyEx(smooth_3x3, cv2.MORPH_OPEN, bg_kernel)
        local_bg = cv2.GaussianBlur(local_bg, (k_bg, k_bg), 0)
        tophat_peak = np.maximum(0.0, smooth_3x3 - local_bg)
        work_map = smooth_3x3 + float(micro_boost_weight) * tophat_peak
    elif speckle_filter_ratio > 0:
        light_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_3x3 = cv2.morphologyEx(work_map, cv2.MORPH_OPEN, light_kernel)
        smooth_3x3 = cv2.GaussianBlur(work_map, (3, 3), 0)
        work_map = np.maximum(opened_3x3, 0.85 * work_map + 0.15 * smooth_3x3)

    if border_ratio > 0:
        by = max(2, int(round(h * border_ratio)))
        bx = max(2, int(round(w * border_ratio)))
        work_map[:by, :] = img_median
        work_map[-by:, :] = img_median
        work_map[:, :bx] = img_median
        work_map[:, -bx:] = img_median

    use_circle, cy, cx, radius = detect_circular_wafer_geometry(
        img_path=img_path,
        h=h,
        w=w,
        wafer_edge_mode=wafer_edge_mode,
        category=category,
    )
    if use_circle and wafer_edge_margin > 0:
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.hypot(yy - cy, xx - cx).astype(np.float32)
        inner_r = radius * (1.0 - wafer_edge_margin)
        outer_r = radius * (1.0 - wafer_edge_margin * 0.15)
        radial_weight = np.ones((h, w), dtype=np.float32)
        transition_mask = (dist_from_center > inner_r) & (dist_from_center < outer_r)
        if np.any(transition_mask):
            t = (dist_from_center[transition_mask] - inner_r) / max(outer_r - inner_r, 1.0)
            radial_weight[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * t))
        radial_weight[dist_from_center >= outer_r] = 0.0
        work_map = img_median + radial_weight * (work_map - img_median)

    return work_map


def merge_nearby_bboxes(
    bboxes: List[Tuple[int, int, int, int, float, int]],
    merge_gap: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    """自動合併重疊或距離極近的碎片紅框，避免單一瑕疵碎成多個小框。"""
    if len(bboxes) <= 1:
        return bboxes

    boxes = [list(b) for b in bboxes]
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2, score, area = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2, bscore, barea = boxes[j]
                if not (
                    x2 + merge_gap < bx1
                    or bx2 + merge_gap < x1
                    or y2 + merge_gap < by1
                    or by2 + merge_gap < y1
                ):
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    score = max(score, bscore)
                    area += barea
                    used[j] = True
                    merged = True
            new_boxes.append([x1, y1, x2, y2, score, area])
            used[i] = True
        boxes = new_boxes

    boxes.sort(key=lambda item: item[4], reverse=True)
    return [tuple(b) for b in boxes]


def extract_tight_bboxes_for_anomaly_image(
    refined_amap: np.ndarray,
    img_threshold: float,
    core_ratio: float = 0.55,
    min_area_ratio: float = 0.00012,
    raw_score_mode: bool = False,
    micro_defect_mode: bool = False,
) -> Tuple[List[Tuple[int, int, int, int, float, int]], float]:
    """峰值自適應核心收斂 (Peak-Adaptive Core Thresholding) 演算法。"""
    h, w = refined_amap.shape[:2]
    work_map = refined_amap.copy()
    img_median = float(np.median(work_map))
    max_score = float(np.max(work_map))
    if max_score < img_threshold:
        return [], img_threshold

    eff_core_ratio = min(core_ratio, 0.45) if micro_defect_mode else core_ratio
    core_thresh = img_median + eff_core_ratio * (max_score - img_median)
    thresh_relax = 0.88 if micro_defect_mode else 0.92
    pixel_box_thresh = max(float(img_threshold * thresh_relax), float(core_thresh))

    binary_mask = (work_map >= pixel_box_thresh).astype(np.uint8) * 255
    if not raw_score_mode:
        if micro_defect_mode:
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_k)
        else:
            k_size = max(3, int(round(max(h, w) / 500)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if raw_score_mode:
        min_area = 1
    elif micro_defect_mode:
        eff_min_ratio = min(min_area_ratio, 0.000004)
        min_area = max(4, int(round((h * w) * eff_min_ratio)))
    else:
        min_area = max(16, int(round((h * w) * min_area_ratio)))

    vis_scale = max(1.0, max(h, w) / 900.0)
    pad = max(4, int(round(5 * vis_scale))) if micro_defect_mode else max(3, int(round(4 * vis_scale)))

    raw_bboxes = []
    for lbl_idx in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl_idx]
        if area < min_area:
            continue
        region_mask = (labels[by:by + bh, bx:bx + bw] == lbl_idx)
        region_pixels = work_map[by:by + bh, bx:bx + bw][region_mask]
        if region_pixels.size == 0:
            continue
        if raw_score_mode:
            region_score = float(np.max(region_pixels))
        elif micro_defect_mode:
            top_n = max(1, min(16, int(round(region_pixels.size * 0.10))))
            region_score = float(np.mean(np.partition(region_pixels, -top_n)[-top_n:]))
        else:
            top_n = max(1, int(round(region_pixels.size * 0.25)))
            region_score = float(np.mean(np.partition(region_pixels, -top_n)[-top_n:]))
        if region_score < img_threshold:
            continue
        x1, y1 = max(0, int(bx - pad)), max(0, int(by - pad))
        x2, y2 = min(w - 1, int(bx + bw + pad)), min(h - 1, int(by + bh + pad))
        raw_bboxes.append((x1, y1, x2, y2, region_score, int(area)))

    if not raw_bboxes:
        return [], pixel_box_thresh

    merge_gap = max(6, int(round(max(h, w) * 0.015)))
    merged_bboxes = merge_nearby_bboxes(raw_bboxes, merge_gap=merge_gap)
    return merged_bboxes, pixel_box_thresh


def estimate_blind_normal_core_stats(
    non_golden_scores: List[float],
    golden_scores: List[float],
) -> Tuple[float, float, float, np.ndarray]:
    """真無 GT 盲池左半部正常核心估計器 (Left-Half Normal Core Estimator, sigma_left)。"""
    pool_arr = np.array(non_golden_scores if non_golden_scores else golden_scores, dtype=np.float32)
    g_arr = np.array(golden_scores, dtype=np.float32) if golden_scores else np.array([], dtype=np.float32)
    if pool_arr.size == 0:
        return 0.20, 0.015, 0.20, np.array([0.20], dtype=np.float32)

    sorted_pool = np.sort(pool_arr)
    n = sorted_pool.size

    half_n = max(1, (n + 1) // 2)
    lower_half = sorted_pool[:half_n]
    med_pool = float(np.median(sorted_pool))

    if g_arr.size > 0:
        g_med = float(np.median(g_arr))
        g_max = float(np.max(g_arr))
        g_std = float(np.std(g_arr)) if g_arr.size > 1 else 0.015
        core_anchor = min(med_pool, max(g_med * 1.25, g_max + 2.0 * max(g_std, 0.012)))
    else:
        g_med = float(np.median(lower_half))
        core_anchor = med_pool

    left_samples = sorted_pool[sorted_pool <= core_anchor]
    if left_samples.size < max(3, n // 5):
        left_samples = lower_half

    mu_core = float(np.percentile(left_samples, 75.0)) if left_samples.size >= 4 else float(np.median(left_samples))
    if g_arr.size > 0:
        mu_core = float(np.clip(mu_core, g_med, core_anchor))

    left_diffs = mu_core - sorted_pool[sorted_pool <= mu_core]
    if left_diffs.size >= 2:
        sigma_left_rms = float(np.sqrt(np.mean(np.square(left_diffs))))
        sigma_left_mad = float(1.4826 * np.median(np.abs(left_diffs)))
        sigma_left = max(sigma_left_rms, sigma_left_mad)
    else:
        sigma_left = float(np.std(left_samples)) if left_samples.size > 1 else 0.015

    if g_arr.size > 1:
        g_iqr = float((np.percentile(g_arr, 75.0) - np.percentile(g_arr, 25.0)) / 1.349)
        sigma_left = max(sigma_left, float(np.std(g_arr)), g_iqr)

    sigma_left = float(np.clip(sigma_left, 0.010, 0.065))

    est_normal_mask = sorted_pool <= (mu_core + 2.5 * sigma_left)
    est_normal_arr = sorted_pool[est_normal_mask] if np.any(est_normal_mask) else left_samples
    return mu_core, sigma_left, g_med, est_normal_arr


def compute_dynamic_effective_threshold(
    non_golden_scores: List[float],
    golden_scores: List[float],
    thresh_mode: str,
    user_threshold: float,
    min_threshold: float,
    auto_percentile: float,
    sigma_k: float,
    top_k: int,
) -> Tuple[float, str]:
    """計算真無 GT 盲池最終異常判定門檻。"""
    mode = thresh_mode.lower().strip()
    scores_arr = np.array(non_golden_scores if non_golden_scores else golden_scores, dtype=np.float32)
    g_arr = np.array(golden_scores, dtype=np.float32) if golden_scores else np.array([], dtype=np.float32)
    if scores_arr.size == 0:
        return max(user_threshold, min_threshold), "無樣本保底門檻"

    mu_core, sigma_left, g_med, est_normal_arr = estimate_blind_normal_core_stats(
        non_golden_scores=non_golden_scores,
        golden_scores=golden_scores,
    )

    left_stat_bound = float(mu_core + sigma_k * sigma_left)

    if g_arr.size > 0:
        g_mean = float(np.mean(g_arr))
        g_max = float(np.max(g_arr))
        g_std = max(float(np.std(g_arr)) if g_arr.size > 1 else 0.012, 0.012)
        golden_ref_bound = max(g_mean + sigma_k * max(g_std, sigma_left), g_max * 1.08 + 0.015)
        golden_safety_cap = max(
            g_max * 1.35 + 0.05,
            mu_core + (sigma_k + 1.5) * sigma_left,
            float(min_threshold) + 0.06,
        )
    else:
        golden_ref_bound = left_stat_bound
        golden_safety_cap = max(mu_core + (sigma_k + 2.0) * sigma_left, float(min_threshold) + 0.08)

    calibrated_bound = min(max(left_stat_bound, golden_ref_bound), golden_safety_cap)

    p_clip = float(np.clip(auto_percentile, 50.0, 99.9))
    p_val_core = float(np.percentile(est_normal_arr, p_clip))
    p_val_raw = float(np.percentile(scores_arr, p_clip))

    topk_bound = 0.0
    if top_k > 0 and scores_arr.size > top_k:
        sorted_desc = np.sort(scores_arr)[::-1]
        topk_bound = float(0.5 * (sorted_desc[top_k - 1] + sorted_desc[top_k]))

    floor_val = max(float(min_threshold), float(user_threshold))

    if mode == "fixed":
        eff = float(user_threshold)
        desc = f"Fixed 固定門檻 = {eff:.4f}"
    elif mode == "mean_std":
        eff = max(float(min_threshold), calibrated_bound, topk_bound) if top_k > 0 else max(float(min_threshold), calibrated_bound)
        desc = (
            f"Left-Core + Golden 雙重校準統計門檻 (mu_core+{sigma_k:g}*σ_left) = {eff:.4f} "
            f"[mu_core={mu_core:.4f}, σ_left={sigma_left:.4f}, Golden_Med={g_med:.4f}, "
            f"LeftBound={left_stat_bound:.4f}, GoldenBound={golden_ref_bound:.4f}"
            f"{f', TopK({top_k})={topk_bound:.4f}' if top_k > 0 else ''}]"
        )
    elif mode == "percentile":
        eff = max(float(min_threshold), min(p_val_raw, golden_safety_cap), topk_bound)
        desc = (
            f"Auto-P{auto_percentile:g} 動態分位數門檻 = {eff:.4f} "
            f"(Core_P{auto_percentile:g}={p_val_core:.4f}, Raw_P{auto_percentile:g}={p_val_raw:.4f}, "
            f"SafetyCap={golden_safety_cap:.4f})"
        )
    elif mode == "top_k":
        eff = max(float(min_threshold), topk_bound if top_k > 0 else calibrated_bound)
        desc = f"Auto-TopK (K={top_k}) 動態門檻 = {eff:.4f} (TopK_Bound={topk_bound:.4f}, MinFloor={min_threshold:.4f})"
    else:
        hybrid_stat = max(floor_val, calibrated_bound, min(p_val_core, golden_safety_cap))
        eff = max(hybrid_stat, topk_bound) if top_k > 0 else hybrid_stat
        desc = (
            f"Hybrid 真無GT雙重校準動態門檻 = {eff:.4f} "
            f"[左半核心上界(mu+{sigma_k:g}*σ_left)={left_stat_bound:.4f} (mu={mu_core:.4f}, σ_left={sigma_left:.4f}), "
            f"Golden基準上界={golden_ref_bound:.4f}, Core_P{auto_percentile:g}={p_val_core:.4f}, "
            f"MinFloor={floor_val:.4f}"
            f"{f', TopK({top_k})={topk_bound:.4f}' if top_k > 0 else ''}]"
        )
    return eff, desc


def evaluate_overfitting_diagnostics(
    golden_scores: List[float],
    non_golden_normal_scores: List[float],
) -> Tuple[float, float, float, str, str]:
    """無監督過擬合 / 跨資料集偏移自檢指標 (Unsupervised Overfitting & Domain-Shift Self-Diagnostic)。"""
    if not golden_scores or not non_golden_normal_scores:
        return 0.0, 1.0, 0.0, "🟢 正常 (無對照組)", ""

    g_mean = float(np.mean(golden_scores))
    ng_med = float(np.median(non_golden_normal_scores))
    overfit_ratio = ng_med / max(g_mean, 1e-6)
    golden_gap = ng_med - g_mean

    if overfit_ratio >= 1.45 and golden_gap >= 0.040:
        status_badge = "🔴 警告: 偵測到明顯過擬合或跨資料集分佈偏移 (Overfit / Domain-Shift Warning)"
        advice_msg = (
            f"  [⚠️ 過擬合/分佈偏移自檢建議] 新測試池正常核心中位數 ({ng_med:.4f}) 顯著高於 Golden 基準圖平均 ({g_mean:.4f}) "
            f"[Overfit_Ratio={overfit_ratio:.2f}x, Gap=+{golden_gap:.4f}]！\n"
            f"                               代表舊權重可能過度記憶舊 Golden 微結構或新舊批號有光照偏移。若正常晶圓誤報偏多，\n"
            f"                               建議調高門檻倍數 (例如設定 --thresh_mode mean_std --sigma_k 4.0)。"
        )
    elif overfit_ratio >= 1.25 and golden_gap >= 0.025:
        status_badge = "🟡 注意: 輕微過擬合/批次偏移傾向 (Mild Shift — 已由左半核心動態門檻自動補償)"
        advice_msg = ""
    else:
        status_badge = "🟢 良好: 無過擬合現象 (Healthy Generalization — 舊Golden基準與新資料集正常分佈一致)"
        advice_msg = ""

    return g_mean, overfit_ratio, golden_gap, status_badge, advice_msg


def stitch_and_filter_anomalies(
    manifest: Dict,
    tile_amap_dir: str,
    final_amap_dir: str,
    out_dir: str,
    category: str,
    golden_filenames: List[str],
    thresh_mode: str,
    user_threshold: float,
    min_threshold: float,
    auto_percentile: float,
    top_k: int,
    sigma_k: float,
    core_ratio: float,
    border_ratio: float,
    wafer_edge_mode: str,
    wafer_edge_margin: float,
    speckle_filter_ratio: float,
    min_area_ratio: float,
    raw_score_mode: bool,
    micro_defect_mode: bool,
    micro_boost_weight: float,
    micro_peak_pixels: int,
    log_files: List[str],
) -> Tuple[int, int, float, str, str, str]:
    """三階段 RAM 串流縫合、碎片即時銷毀與異常紅框篩選輸出 (支援跨資料集外部 Golden 基準校準)。"""
    os.makedirs(final_amap_dir, exist_ok=True)
    anomaly_dir = os.path.join(out_dir, "anomaly_images")
    anomaly_fullres_bbox_dir = os.path.join(anomaly_dir, "full_res_bbox_only")
    refined_cache_dir = os.path.join(out_dir, "_refined_cache_tmp")
    safe_rmtree(refined_cache_dir)
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(anomaly_fullres_bbox_dir, exist_ok=True)
    os.makedirs(refined_cache_dir, exist_ok=True)

    tile_overlap = manifest["tile_overlap"]
    golden_set = set(golden_filenames)
    num_test_imgs = manifest.get("num_images", len(manifest["images"]))
    num_ext_golden = manifest.get("num_external_golden", 0)

    if raw_score_mode:
        score_mode_label = "RAW_UNFILTERED (未經四合一濾波 + 單點最大值 np.max)"
        score_mode_code = "RAW_UNFILTERED"
    elif micro_defect_mode:
        score_mode_label = (
            f"MICRO_DEFECT_SENSITIVE (微瑕疵白頂帽增強 [boost={micro_boost_weight:g}] + "
            f"Top-{micro_peak_pixels}px 防稀釋局部尖峰 + 5x5閉運算微框保護)"
        )
        score_mode_code = "MICRO_DEFECT_SENSITIVE"
    else:
        score_mode_label = "FILTERED_ROBUST (四合一防誤報純化 [3x3輕量濾微塵+tol=0.15] + Top-0.015% 抗噪點峰值)"
        score_mode_code = "FILTERED_ROBUST"

    ext_msg = f" + {num_ext_golden} 張跨資料集 Golden 基準圖" if num_ext_golden > 0 else ""
    log_and_print(
        f"\n[Step 3.1] 啟動 RAM 即時縫合與碎片銷毀模式 (共 {num_test_imgs} 張測試原圖{ext_msg}) | 計分模式: {score_mode_label}",
        log_files,
    )

    golden_raw_cache: Dict[str, np.ndarray] = {}
    golden_maps: List[np.ndarray] = []
    if not raw_score_mode:
        for fname, meta in manifest["images"].items():
            if meta.get("is_external_golden", False) or fname in golden_set or os.path.basename(fname) in golden_set:
                g_amap = stitch_single_image_amap(
                    meta,
                    tile_amap_dir,
                    tile_overlap,
                    delete_tiles=True,
                )
                golden_raw_cache[fname] = g_amap
                golden_maps.append(g_amap)

    golden_spatial_ref: Optional[np.ndarray] = None
    if golden_maps:
        ref_h, ref_w = golden_maps[0].shape[:2]
        aligned_g = [
            cv2.resize(gm, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            if gm.shape[:2] != (ref_h, ref_w)
            else gm
            for gm in golden_maps
        ]
        g_max_map = np.max(np.stack(aligned_g, axis=0), axis=0)
        blur_k = max(15, int(round(max(ref_h, ref_w) * 0.035)) | 1)
        golden_spatial_ref = cv2.GaussianBlur(g_max_map, (blur_k, blur_k), 0)
        del aligned_g, g_max_map
    del golden_maps
    gc.collect()

    stitched_info = []
    golden_max_scores = []
    non_golden_all_scores = []
    ext_golden_score_logs = []

    for idx, (fname, meta) in enumerate(manifest["images"].items(), 1):
        is_ext_golden = bool(meta.get("is_external_golden", False))
        is_golden_img = is_ext_golden or (fname in golden_set) or (os.path.basename(fname) in golden_set)
        if fname in golden_raw_cache:
            raw_amap = golden_raw_cache.pop(fname)
        else:
            raw_amap = stitch_single_image_amap(
                meta,
                tile_amap_dir,
                tile_overlap,
                delete_tiles=True,
            )

        raw_flat = raw_amap.reshape(-1)
        raw_single_max = float(np.max(raw_flat))

        if micro_defect_mode:
            peak_k = max(6, min(int(micro_peak_pixels), int(round(raw_flat.size * 0.00002))))
        else:
            peak_k = max(8, int(round(raw_flat.size * 0.00015)))

        raw_top_peak = float(np.mean(np.partition(raw_flat, -peak_k)[-peak_k:]))

        refined_amap = refine_amap_for_wafer_and_noise(
            full_amap=raw_amap,
            img_path=meta["path"],
            category=category,
            border_ratio=border_ratio,
            wafer_edge_mode=wafer_edge_mode,
            wafer_edge_margin=wafer_edge_margin,
            speckle_filter_ratio=speckle_filter_ratio,
            golden_spatial_ref=golden_spatial_ref if not is_golden_img else None,
            golden_tol_weight=0.15,
            micro_defect_mode=micro_defect_mode,
            micro_boost_weight=micro_boost_weight,
        )
        ref_flat = refined_amap.reshape(-1)
        filtered_robust_max = float(np.mean(np.partition(ref_flat, -peak_k)[-peak_k:]))

        if raw_score_mode:
            active_amap = raw_amap
            active_max_score = raw_single_max
            active_flat = raw_flat
        else:
            active_amap = refined_amap
            active_max_score = filtered_robust_max
            active_flat = ref_flat

        top_k_count = max(20, int(active_flat.size * 0.0005))
        top_k_score = float(np.mean(np.partition(active_flat, -top_k_count)[-top_k_count:]))

        if is_golden_img:
            golden_max_scores.append(active_max_score)
        else:
            non_golden_all_scores.append(active_max_score)

        # 若為跨資料集外部指定的 Golden Image，僅做空間容差與門檻基準校準，不混入新測試集的排名與紅框輸出
        if is_ext_golden:
            ext_golden_score_logs.append(f"{os.path.basename(meta['path'])}={active_max_score:.4f}")
            del raw_amap, refined_amap, active_amap
            continue

        cache_npy_path = os.path.join(refined_cache_dir, f"{idx:05d}_{meta['stem']}.npy")
        try:
            np.save(cache_npy_path, active_amap.astype(np.float16))
        except Exception:
            cache_npy_path = ""

        del raw_amap, refined_amap, active_amap
        stitched_info.append(
            {
                "filename": fname,
                "stem": meta["stem"],
                "meta": meta,
                "cache_npy_path": cache_npy_path,
                "max_score": active_max_score,
                "raw_max_score": raw_single_max,
                "raw_top_peak_score": raw_top_peak,
                "filtered_max_score": filtered_robust_max,
                "filter_delta": filtered_robust_max - raw_single_max,
                "top_k_score": top_k_score,
            }
        )
        if idx % 25 == 0:
            gc.collect()

    golden_raw_cache.clear()
    safe_rmtree(tile_amap_dir)
    log_and_print(
        f"[🧹 碎片清理完成] 所有子圖塊已於 RAM 內拼回整張 Wafer，已自動銷毀全部碎片 Tile .npy -> {os.path.abspath(tile_amap_dir)}",
        log_files,
    )
    if ext_golden_score_logs:
        log_and_print(
            f"[📏 跨資料集 Golden 基準分數] {', '.join(ext_golden_score_logs)} "
            f"(Mean={float(np.mean(golden_max_scores)):.4f}, Max={float(np.max(golden_max_scores)):.4f})",
            log_files,
        )

    effective_threshold, thresh_mode_str = compute_dynamic_effective_threshold(
        non_golden_scores=non_golden_all_scores,
        golden_scores=golden_max_scores,
        thresh_mode=thresh_mode,
        user_threshold=user_threshold,
        min_threshold=min_threshold,
        auto_percentile=auto_percentile,
        sigma_k=sigma_k,
        top_k=top_k,
    )

    _, _, _, est_normal_arr = estimate_blind_normal_core_stats(
        non_golden_scores=non_golden_all_scores,
        golden_scores=golden_max_scores,
    )
    ref_normal_scores = est_normal_arr.tolist() if est_normal_arr.size > 0 else non_golden_all_scores

    g_mean_score, overfit_ratio, golden_gap, overfit_badge, overfit_advice = evaluate_overfitting_diagnostics(
        golden_scores=golden_max_scores,
        non_golden_normal_scores=ref_normal_scores,
    )

    log_and_print(
        f"[Step 3.2] 異常判定門檻設定: {thresh_mode_str}\n"
        f"           (計分模式={score_mode_code}, "
        f"圓周邊緣抑制={'OFF(Raw)' if raw_score_mode else wafer_edge_mode}, "
        f"邊界縮減={0.0 if raw_score_mode else wafer_edge_margin*100:.1f}%)\n"
        f"[Step 3.2b] 過擬合自檢診斷 : {overfit_badge} "
        f"[Golden平均={g_mean_score:.4f} | 盲池正常核心中位數={float(np.median(ref_normal_scores)) if ref_normal_scores else 0.0:.4f} | "
        f"Overfit_Ratio={overfit_ratio:.2f}x (Gap={golden_gap:+.4f})]",
        log_files,
    )
    if overfit_advice:
        log_and_print(overfit_advice, log_files)

    stitched_info.sort(key=lambda item: item["max_score"], reverse=True)
    ranking_records = []
    anomaly_count = 0
    skipped_normal_count = 0

    for rank_idx, item in enumerate(stitched_info, 1):
        fname = item["filename"]
        stem = item["stem"]
        meta = item["meta"]
        cache_npy_path = item["cache_npy_path"]
        max_score = item["max_score"]
        raw_max_score = item["raw_max_score"]
        filtered_max_score = item["filtered_max_score"]
        filter_delta = item["filter_delta"]
        top_k_score = item["top_k_score"]
        h, w = meta["height"], meta["width"]
        is_golden = (fname in golden_set) or (os.path.basename(fname) in golden_set)

        if max_score < effective_threshold:
            if cache_npy_path and os.path.exists(cache_npy_path):
                try:
                    os.remove(cache_npy_path)
                except OSError:
                    pass
            skipped_normal_count += 1
            status_tag = "GOLDEN_PASS" if is_golden else "PASS_SKIPPED"
            ranking_records.append(
                {
                    "rank": rank_idx,
                    "filename": fname,
                    "status": status_tag,
                    "max_score": max_score,
                    "raw_max_score": raw_max_score,
                    "filtered_max_score": filtered_max_score,
                    "filter_delta": filter_delta,
                    "top_k_score": top_k_score,
                    "num_bboxes": 0,
                    "saved_file": "(未超過動態門檻，不輸出圖片)",
                    "resolution": f"{w}x{h}",
                }
            )
            continue

        if cache_npy_path and os.path.exists(cache_npy_path):
            active_amap = np.load(cache_npy_path).astype(np.float32)
            try:
                os.remove(cache_npy_path)
            except OSError:
                pass
        else:
            active_amap = np.zeros((h, w), dtype=np.float32)

        bboxes, _ = extract_tight_bboxes_for_anomaly_image(
            refined_amap=active_amap,
            img_threshold=effective_threshold,
            core_ratio=core_ratio,
            min_area_ratio=min_area_ratio,
            raw_score_mode=raw_score_mode,
            micro_defect_mode=micro_defect_mode,
        )

        if not bboxes:
            del active_amap
            skipped_normal_count += 1
            status_tag = "GOLDEN_PASS" if is_golden else "PASS_NOISE_FILTERED"
            ranking_records.append(
                {
                    "rank": rank_idx,
                    "filename": fname,
                    "status": status_tag,
                    "max_score": max_score,
                    "raw_max_score": raw_max_score,
                    "filtered_max_score": filtered_max_score,
                    "filter_delta": filter_delta,
                    "top_k_score": top_k_score,
                    "num_bboxes": 0,
                    "saved_file": "(微小單點噪點已自動濾除，不輸出圖片)",
                    "resolution": f"{w}x{h}",
                }
            )
            continue

        anomaly_count += 1
        save_dtype = np.float16 if max(active_amap.shape) > 2048 else np.float32
        np.save(
            os.path.join(final_amap_dir, f"Rank{rank_idx:03d}_Score{max_score:.4f}_{stem}.npy"),
            active_amap.astype(save_dtype),
        )

        orig_bgr = load_image_bgr_uint8(meta["path"])
        vis_scale = max(1.0, max(h, w) / 900.0)
        box_thickness = max(2, int(round(2.5 * vis_scale)))
        font_scale = max(0.5, 0.55 * vis_scale)
        font_thickness = max(1, int(round(1.5 * vis_scale)))

        bbox_canvas = orig_bgr.copy()
        for (x1, y1, x2, y2, region_max, _) in bboxes:
            cv2.rectangle(bbox_canvas, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            label_txt = f"{region_max:.3f}"
            (tw, th_txt), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            ty1 = max(0, y1 - th_txt - 8)
            cv2.rectangle(bbox_canvas, (x1, ty1), (x1 + tw + 6, ty1 + th_txt + 6), (0, 0, 255), -1)
            cv2.putText(
                bbox_canvas,
                label_txt,
                (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        img_med = float(np.median(active_amap))
        norm_map = np.clip((active_amap - img_med) / max(float(np.max(active_amap)) - img_med, 1e-6), 0.0, 1.0)
        hm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(orig_bgr, 0.55, hm_color, 0.45, 0)

        safe_out_fname = re.sub(r"[\\/]+", "__", fname)
        prefixed_img_name = f"Rank{rank_idx:03d}_Score{max_score:.4f}_Box{len(bboxes):02d}_{safe_out_fname}"

        fullres_out_path = os.path.join(anomaly_fullres_bbox_dir, prefixed_img_name)
        cv2.imwrite(fullres_out_path, bbox_canvas)

        panel_h = min(900, h)
        panel_w = int(round(w * (panel_h / float(h))))
        p1 = cv2.resize(orig_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(overlay_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(bbox_canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        tri_panel = np.hstack([p1, p2, p3])

        banner_h = max(36, int(round(panel_h * 0.055)))
        banner = np.zeros((banner_h, tri_panel.shape[1], 3), dtype=np.uint8)
        mode_tag = "RAW" if raw_score_mode else ("MICRO" if micro_defect_mode else "FILT")
        banner_txt = (
            f"[{mode_tag}] Orig: {fname} | Score: {max_score:.4f} (Raw:{raw_max_score:.4f}, Filt:{filtered_max_score:.4f}, "
            f"EffThresh:{effective_threshold:.4f}) | Status: ANOMALY (Boxes:{len(bboxes)})"
        )
        cv2.putText(
            banner,
            banner_txt,
            (12, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, banner_h / 55.0),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        final_vis = np.vstack([banner, tri_panel])
        vis_out_path = os.path.join(anomaly_dir, prefixed_img_name)
        cv2.imwrite(vis_out_path, final_vis)

        del active_amap, orig_bgr, bbox_canvas, hm_color, overlay_bgr, tri_panel, final_vis
        gc.collect()

        status_tag = "GOLDEN_ANOMALY" if is_golden else "ANOMALY"
        ranking_records.append(
            {
                "rank": rank_idx,
                "filename": fname,
                "status": status_tag,
                "max_score": max_score,
                "raw_max_score": raw_max_score,
                "filtered_max_score": filtered_max_score,
                "filter_delta": filter_delta,
                "top_k_score": top_k_score,
                "num_bboxes": len(bboxes),
                "saved_file": prefixed_img_name,
                "resolution": f"{w}x{h}",
            }
        )

    safe_rmtree(refined_cache_dir)
    del golden_spatial_ref
    gc.collect()

    non_golden_recs = [r for r in ranking_records if not r["status"].startswith("GOLDEN")]
    passed_normal_recs = [r for r in non_golden_recs if r["status"].startswith("PASS")]
    target_recs = passed_normal_recs if passed_normal_recs else non_golden_recs
    if not target_recs:
        target_recs = ranking_records

    dist_scores = [float(r["max_score"]) for r in target_recs]
    raw_dist_scores = [float(r["raw_max_score"]) for r in target_recs]
    filt_dist_scores = [float(r["filtered_max_score"]) for r in target_recs]

    max_s = float(np.max(dist_scores))
    p99_s = float(np.percentile(dist_scores, 99))
    p95_s = float(np.percentile(dist_scores, 95))
    median_s = float(np.median(dist_scores))
    mean_s = float(np.mean(dist_scores))
    std_s = float(np.std(dist_scores))
    m1std = mean_s + 1.0 * std_s
    m2std = mean_s + 2.0 * std_s
    m3std = mean_s + 3.0 * std_s

    raw_max_s = float(np.max(raw_dist_scores))
    raw_mean_s = float(np.mean(raw_dist_scores))
    raw_std_s = float(np.std(raw_dist_scores))
    filt_max_s = float(np.max(filt_dist_scores))
    filt_mean_s = float(np.mean(filt_dist_scores))
    filt_std_s = float(np.std(filt_dist_scores))

    dist_summary_lines = (
        f"      📊 盲池正常核心分數分佈 ({score_mode_code}) : "
        f"Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
        f"      📐 盲池正常核心門檻參考值      : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
        f"      🔬 濾波前後正常圖對照 (Ablation): "
        f"[未濾波 Raw] Max={raw_max_s:.4f}, Mean={raw_mean_s:.4f}, Std={raw_std_s:.4f}  vs  "
        f"[當前模式] Max={filt_max_s:.4f}, Mean={filt_mean_s:.4f}, Std={filt_std_s:.4f}\n"
        f"      🩺 模型過擬合自檢 (Overfit)    : {overfit_badge}\n"
        f"                                       [Golden平均={g_mean_score:.4f} vs 盲池正常中位數={median_s:.4f} | Overfit_Ratio={overfit_ratio:.2f}x | Gap={golden_gap:+.4f}]"
    )
    if overfit_advice:
        dist_summary_lines += f"\n{overfit_advice}"

    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== FoundAD High-Res Anomaly Detection Report ===\n")
        f.write(
            f"Score Mode: {score_mode_code} | "
            f"Threshold Mode: {thresh_mode} | Effective Threshold: {effective_threshold:.4f} | "
            f"Wafer Edge Mode: {'off(raw)' if raw_score_mode else wafer_edge_mode} (Margin: {wafer_edge_margin:.3f})\n"
            f"Core Box Ratio: {core_ratio:.2f} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_normal_count}\n"
            f"Blind Normal Core Score Dist: Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | "
            f"Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
            f"Ablation Compare (Normal)   : Raw_Max={raw_max_s:.4f}, Raw_Mean={raw_mean_s:.4f} vs "
            f"Filtered_Max={filt_max_s:.4f}, Filtered_Mean={filt_mean_s:.4f}\n"
            f"Threshold Ref (Mean+k*Std)  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
            f"Overfit Diagnostic          : {overfit_badge} "
            f"(Golden_Mean={g_mean_score:.4f}, BlindNormal_Med={median_s:.4f}, Ratio={overfit_ratio:.2f}x, Gap={golden_gap:+.4f})\n"
        )
        if ext_golden_score_logs:
            f.write(f"External Golden Ref Scores  : {', '.join(ext_golden_score_logs)}\n")
        f.write("-" * 145 + "\n")
        f.write(
            f"{'Rank':<5} | {'Original_Filename':<26} | {'Status':<19} | {'Max_Score':<9} | "
            f"{'Raw_Score':<9} | {'Filt_Score':<10} | {'Delta':<8} | {'BBoxes':<6} | {'Saved_Anomaly_Filename'}\n"
        )
        f.write("-" * 145 + "\n")
        for r in ranking_records:
            f.write(
                f"{r['rank']:<5} | {r['filename']:<26} | {r['status']:<19} | "
                f"{r['max_score']:<9.4f} | {r['raw_max_score']:<9.4f} | {r['filtered_max_score']:<10.4f} | "
                f"{r['filter_delta']:<+8.4f} | {r['num_bboxes']:<6} | {r['saved_file']}\n"
            )

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Rank",
                "Original_Filename",
                "Status",
                "Max_Score",
                "Raw_Max_Score",
                "Filtered_Max_Score",
                "Filter_Delta",
                "TopK_Mean_Score",
                "Num_BBoxes",
                "Saved_Anomaly_Filename",
                "Resolution",
            ]
        )
        for r in ranking_records:
            writer.writerow(
                [
                    r["rank"],
                    r["filename"],
                    r["status"],
                    f"{r['max_score']:.6f}",
                    f"{r['raw_max_score']:.6f}",
                    f"{r['filtered_max_score']:.6f}",
                    f"{r['filter_delta']:+.6f}",
                    f"{r['top_k_score']:.6f}",
                    r["num_bboxes"],
                    r["saved_file"],
                    r["resolution"],
                ]
            )

    log_and_print(
        f"[Step 3.3] 篩選與畫框完成：共 {anomaly_count} 張超過有效門檻 ({effective_threshold:.4f}) 已畫框並將整張 Wafer .npy 存於 output；"
        f"其餘 {skipped_normal_count} 張正常圖已自動略過不輸出。",
        log_files,
    )
    return anomaly_count, skipped_normal_count, effective_threshold, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines


def main():
    parser = argparse.ArgumentParser(description="FoundAD 無GT高解析度推論主程式 (Stage 2: Inference)")
    parser.add_argument("--pool_dir", type=str, required=True, help="欲進行推論的新資料集圖片資料夾路徑")
    parser.add_argument("--ckpt", type=str, default="auto", help="DINO 預訓練 Backbone 權重路徑")
    parser.add_argument(
        "--golden_src",
        "--custom_golden_dir",
        dest="golden_src",
        type=str,
        default="",
        help=(
            "直接指定舊資料集挑過的 Golden Image 來源（支援：1. Golden 圖片資料夾路徑、"
            "2. 舊資料集的 shot_dir 如 ./datasets/wafer_TMQB28_4shot、"
            "3. selected_4_golden_images.txt、4. 逗號分隔的多張圖片路徑）"
        ),
    )
    parser.add_argument(
        "--trained_ckpt",
        "--custom_ckpt",
        dest="trained_ckpt",
        type=str,
        default="",
        help=(
            "直接指定舊資料集訓練過的 FoundAD 權重檔路徑（支援直接傳入 .pth.tar 檔案路徑 "
            "如 ./logs/.../train-step2000.pth.tar 或 pretrained.pth.tar，或傳入權重所在資料夾路徑）"
        ),
    )
    parser.add_argument("--out_dir", type=str, default="./output/2_fab_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot")
    parser.add_argument("--category", type=str, default="auto")
    parser.add_argument("--app_test", type=str, default="test")
    parser.add_argument("--diy_name", type=str, default="sim_escape")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=4)

    # 門檻與動態對齊參數
    parser.add_argument(
        "--thresh_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "mean_std", "percentile", "top_k", "fixed"],
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min_threshold", type=float, default=0.27)
    parser.add_argument("--auto_percentile", type=float, default=95.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--sigma_k", type=float, default=3.5)

    # 晶圓圓周邊緣抑制、微塵濾波與「微瑕疵高靈敏模式 (--micro_defect_mode)」開關
    parser.add_argument("--raw_score_mode", action="store_true", help="完全不經過濾波之原始最大值模式 (消融實驗對照組)")
    parser.add_argument(
        "--micro_defect_mode",
        action="store_true",
        help="一鍵啟動微瑕疵高靈敏模式：形態學白頂帽局部對比度增強 + Top-24px 防稀釋峰值計分 + 5x5閉運算微框保護",
    )
    parser.add_argument(
        "--micro_boost_weight",
        type=float,
        default=0.35,
        help="micro_defect_mode 下白頂帽局部尖峰增強權重 (預設 0.35)",
    )
    parser.add_argument(
        "--micro_peak_pixels",
        type=int,
        default=24,
        help="micro_defect_mode 下計算全圖異常分數的局部尖峰像素數上限 (預設 24，防止微瑕疵被背景稀釋)",
    )
    parser.add_argument("--core_ratio", type=float, default=0.55)
    parser.add_argument("--border_ratio", type=float, default=0.02)
    parser.add_argument("--wafer_edge_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--wafer_edge_margin", type=float, default=0.04)
    parser.add_argument("--speckle_filter_ratio", type=float, default=0.008)
    parser.add_argument("--min_area_ratio", type=float, default=0.00012)

    # 推論與切塊控制參數 (同步 min_tile_trigger=1280)
    parser.add_argument("--force_infer", action="store_true", help="強制重新切塊測試池並重跑 GPU 推論")
    parser.add_argument("--mock_from_mvtec", type=str, default="")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument(
        "--min_tile_trigger",
        type=int,
        default=1280,
        help="影像最大邊 <= min_tile_trigger (預設 1280px) 時直接 1:1 整張推論，> 1280px 才啟動重疊切塊與無縫縫合",
    )
    parser.add_argument("--max_train_tiles", type=int, default=48)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--bg_std_thresh", type=float, default=1.5)
    parser.add_argument("--disable_tiling", action="store_true")
    args = parser.parse_args()

    if not args.category or args.category.lower() == "auto":
        if args.mock_from_mvtec:
            args.category = os.path.basename(os.path.normpath(args.mock_from_mvtec))
        else:
            args.category = os.path.basename(os.path.normpath(args.pool_dir))

    now_tpe = datetime.now(TZ_TAIPEI)
    timestamp = now_tpe.strftime("%Y%m%d_%H%M%S")
    base_out_dir = os.path.abspath(args.out_dir)
    run_out_dir = os.path.join(base_out_dir, timestamp)
    os.makedirs(run_out_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)
    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    shot_dir = os.path.join(abs_datasets_dir, args.data_name)

    fixed_log_txt = os.path.join(run_out_dir, "log.txt")
    parent_latest_log = os.path.join(base_out_dir, "latest_log.txt")
    log_files = [fixed_log_txt, parent_latest_log]

    pipeline_start_t = time.perf_counter()
    start_dt_str = now_tpe.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    log_and_print(
        f"\n{'='*75}\n"
        f"[Task 2 無GT推論階段啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_out_dir}\n"
        f"[指定 Golden 來源 (--golden_src)] {args.golden_src if args.golden_src else '(預設沿用 shot_dir)'}\n"
        f"[指定訓練權重 (--trained_ckpt)] {args.trained_ckpt if args.trained_ckpt else '(預設搜尋 ./logs/' + args.data_name + ')'}\n"
        f"[微瑕疵高靈敏模式狀態] {'ON (已啟動白頂帽增強與防稀釋局部尖峰計分)' if args.micro_defect_mode else 'OFF (標準四合一濾波模式)'}\n"
        f"{'='*75}",
        log_files,
    )

    # 1. 載入 Golden 檔案清單與對接訓練權重
    golden_filenames, external_golden_paths, meta_step, recorded_train_time = load_golden_filenames_for_infer(
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
        min_tile_trigger=args.min_tile_trigger,
        run_out_dir=run_out_dir,
        log_files=log_files,
        golden_src=args.golden_src,
    )

    req_step = meta_step if (meta_step is not None and not args.trained_ckpt) else args.ckpt_step
    actual_step, ckpt_dirs, _ = prepare_ckpt_for_demo(
        data_name=args.data_name,
        diy_name=args.diy_name,
        req_step=req_step,
        log_files=log_files,
        trained_ckpt=args.trained_ckpt,
    )

    # 2. 測試池高解析度切塊與快取檢查 (將外部 Golden 清單納入快取指紋，確保更換 Golden 時自動更新)
    pool_signature = compute_pool_signature(args.pool_dir)
    ext_golden_sig = ""
    if external_golden_paths:
        h_ext = hashlib.md5()
        for gp in external_golden_paths:
            try:
                sz = os.path.getsize(gp)
            except OSError:
                sz = 0
            h_ext.update(f"{os.path.abspath(gp)}:{sz};".encode("utf-8"))
        ext_golden_sig = f"_extG{h_ext.hexdigest()[:10]}"

    tile_config_key = (
        f"t{args.tile_size}_ov{args.tile_overlap}_trig{args.min_tile_trigger}_"
        f"bg{args.bg_std_thresh:g}_dt{int(args.disable_tiling)}{ext_golden_sig}"
    )

    amap_dump_dir = os.path.join(run_out_dir, "amap_npy")
    report_txt_path = os.path.join(run_out_dir, "escape_ranking_report.txt")
    report_csv_path = os.path.join(run_out_dir, "escape_ranking_report.csv")

    tiled_pool_dir = os.path.join(abs_datasets_dir, f"{args.data_name}_tiled_pool")
    cached_manifest_path = os.path.join(abs_datasets_dir, f"{args.data_name}_cached_tile_manifest.json")
    tile_amap_tmp_dir = os.path.join(run_out_dir, "_tmp_tile_amap_npy")

    legacy_tile_cache_root = os.path.join(abs_datasets_dir, f"{args.data_name}_tile_amap_cache")
    if os.path.exists(legacy_tile_cache_root):
        safe_rmtree(legacy_tile_cache_root)

    test_tile_sec = 0.0
    model_infer_sec = 0.0
    used_png_tile_cache = False
    manifest = None

    if not args.force_infer:
        manifest = is_tiled_png_pool_valid(
            tiled_pool_dir=tiled_pool_dir,
            cached_manifest_path=cached_manifest_path,
            pool_signature=pool_signature,
            tile_config_key=tile_config_key,
        )

    if manifest is not None:
        used_png_tile_cache = True
        safe_copy(cached_manifest_path, os.path.join(run_out_dir, "tile_manifest.json"))
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)
        log_and_print(
            f"\n[⚡ 測試池切塊快取命中] 測試圖片池、Golden 基準與切塊參數未變動 (min_tile_trigger={args.min_tile_trigger}px)，"
            f"直接沿用既有 {total_test_tiles} 張推論圖塊 PNG (0.00 秒)",
            log_files,
        )
    else:
        t_test_tile_start = time.perf_counter()
        manifest = prepare_tiled_test_pool(
            pool_dir=args.pool_dir,
            tiled_pool_dir=tiled_pool_dir,
            manifest_save_dir=run_out_dir,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            min_tile_trigger=args.min_tile_trigger,
            bg_std_thresh=args.bg_std_thresh,
            disable_tiling=args.disable_tiling,
            pool_signature=pool_signature,
            tile_config_key=tile_config_key,
            cached_manifest_path=cached_manifest_path,
            log_files=log_files,
            external_golden_paths=external_golden_paths,
        )
        test_tile_sec = time.perf_counter() - t_test_tile_start
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)

    safe_rmtree(tile_amap_tmp_dir)
    os.makedirs(tile_amap_tmp_dir, exist_ok=True)

    # 3. 執行 FoundAD 子圖塊批次推論
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FAB_AMAP_DUMP_DIR"] = os.path.abspath(tile_amap_tmp_dir)
    env["FAB_SKIP_TILE_HEATMAP_PNG"] = "1"
    env["FAB_INFER_BATCH_SIZE"] = str(args.infer_batch_size)
    env["FAB_BBOX_OUT_DIR"] = ""
    env["FAB_SIGMA_K"] = str(args.sigma_k)
    env["FAB_GOLDEN_NAMES"] = ""

    t_infer_start = time.perf_counter()
    demo_cmd = (
        f"python foundad/main.py mode=demo data.dataset=mvtec "
        f"data.data_name={args.data_name} diy_name={args.diy_name} "
        f"data.test_root={os.path.abspath(tiled_pool_dir)} app={args.app_test} app.ckpt_step={actual_step}"
    )
    log_and_print(
        f"\n[Step 2] 執行 FoundAD 批次推論 "
        f"(共 {total_test_tiles} 張推論圖塊, BatchSize={args.infer_batch_size}, "
        f"暫存於 {os.path.abspath(tile_amap_tmp_dir)} 並將於 Step 3 RAM 拼接後立即銷毀):",
        log_files,
    )
    run_cmd_and_log(demo_cmd, log_files, env=env)
    model_infer_sec = time.perf_counter() - t_infer_start

    for old_viz in glob.glob(f"./logs/{args.data_name}/**/no_gt_results", recursive=True):
        safe_rmtree(old_viz)
    for old_hm in glob.glob(f"./logs/{args.data_name}/**/demo/heatmaps", recursive=True):
        safe_rmtree(old_hm)

    # 4. RAM 即時縫合、碎片銷毀與四合一防誤報篩選
    t_stitch_start = time.perf_counter()
    anomaly_count, skipped_count, eff_thresh, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines = (
        stitch_and_filter_anomalies(
            manifest=manifest,
            tile_amap_dir=tile_amap_tmp_dir,
            final_amap_dir=amap_dump_dir,
            out_dir=run_out_dir,
            category=args.category,
            golden_filenames=golden_filenames,
            thresh_mode=args.thresh_mode,
            user_threshold=args.threshold,
            min_threshold=args.min_threshold,
            auto_percentile=args.auto_percentile,
            top_k=args.top_k,
            sigma_k=args.sigma_k,
            core_ratio=args.core_ratio,
            border_ratio=args.border_ratio,
            wafer_edge_mode=args.wafer_edge_mode,
            wafer_edge_margin=args.wafer_edge_margin,
            speckle_filter_ratio=args.speckle_filter_ratio,
            min_area_ratio=args.min_area_ratio,
            raw_score_mode=args.raw_score_mode,
            micro_defect_mode=args.micro_defect_mode,
            micro_boost_weight=args.micro_boost_weight,
            micro_peak_pixels=args.micro_peak_pixels,
            log_files=log_files,
        )
    )
    stitch_vis_sec = time.perf_counter() - t_stitch_start

    total_infer_time = test_tile_sec + model_infer_sec + stitch_vis_sec
    avg_infer_per_img = total_infer_time / max(1, num_test_images)
    total_wall_time = time.perf_counter() - pipeline_start_t

    mode_summary_str = "RAW" if args.raw_score_mode else ("MICRO_DEFECT" if args.micro_defect_mode else "FILTERED")
    infer_done_summary = (
        f"\n{'='*75}\n"
        f" [✅ 推論階段完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-'*75}\n"
        f"  [1] 紀錄之訓練階段耗時 (Recorded Train Time)  : {format_duration(recorded_train_time)} (ckpt_step={actual_step})\n"
        f"{'-'*75}\n"
        f"  [2] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ Step 2.1 測試池圖塊準備 ({total_test_tiles} 塊, trigger={args.min_tile_trigger}px) : {format_duration(test_tile_sec)}"
        f"{' (Cached)' if used_png_tile_cache else ''}\n"
        f"      ├─ Step 2.2 FoundAD 模型推論               : {format_duration(model_infer_sec)}\n"
        f"      ├─ Step 3 RAM 縫合銷毀碎片與{mode_summary_str}畫框 : {format_duration(stitch_vis_sec)}\n"
        f"      └─ 平均每張原圖推論耗時 ({num_test_images} 張原圖)         : {avg_infer_per_img:.2f} 秒/張\n"
        f"{'-'*75}\n"
        f"  [3] 本次推論腳本總執行時間 (Total Wall Time)  : {format_duration(total_wall_time)}\n"
        f"{'-'*75}\n"
        f"  [4] 門檻篩選統計、分數分佈與過擬合自檢 (Score={mode_summary_str}, Mode={args.thresh_mode}, Eff_Thresh={eff_thresh:.4f}):\n"
        f"      🚨 超過門檻判定異常並畫框輸出 : {anomaly_count} 張\n"
        f"      ✅ 未達門檻或微塵濾除直接略過 : {skipped_count} 張 (不輸出圖片)\n"
        f"{dist_summary_lines}\n"
        f"{'-'*75}\n"
        f"  [5] 權重檔與本次推論結果儲存資料夾清單 (Saved Directories & Files):\n"
        f"      📦 使用之訓練權重資料夾       : {ckpt_dirs[0]}\n"
        f"      🏆 指定之 Golden 來源         : {args.golden_src if args.golden_src else shot_dir}\n"
        f"      🧹 子圖塊碎片 Tile .npy 狀態  : 已於 RAM 拼接完成後立即自動銷毀 (0 殘留)\n"
        f"      📂 本次時間戳記推論主資料夾   : {run_out_dir}\n"
        f"          ├── 異常紅框三合一對照圖  : {anomaly_dir}\n"
        f"          ├── 異常1:1全解析度紅框圖 : {anomaly_fullres_bbox_dir}\n"
        f"          ├── 整張Wafer熱力圖(.npy) : {amap_dump_dir}\n"
        f"          ├── 完整排名報表 (TXT)    : {report_txt_path}\n"
        f"          ├── 完整排名報表 (CSV)    : {report_csv_path}\n"
        f"          └── 本次執行日誌 (log.txt): {fixed_log_txt}\n"
        f"{'='*75}"
    )
    log_and_print(infer_done_summary, log_files)

    if os.path.exists(report_txt_path):
        with open(report_txt_path, "a", encoding="utf-8") as f:
            f.write(infer_done_summary + "\n")


if __name__ == "__main__":
    main()