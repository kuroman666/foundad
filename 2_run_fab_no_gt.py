# 檔案位置: FoundAD/2_run_fab_no_gt.py (全選覆蓋並存檔)
import os
import re
import csv
import time
import stat
import glob
import json
import shutil
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from fab_tools.auto_golden import select_golden_images, IMG_EXTENSIONS


def format_duration(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS.xx 與總秒數。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f} ({seconds:.2f} 秒)"


def log_and_print(msg: str, log_files: List[str]):
    """同時在 Terminal 顯示並附加寫入 log.txt 檔案。"""
    print(msg, flush=True)
    for lf in log_files:
        if lf:
            with open(lf, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def run_cmd_and_log(cmd: str, log_files: List[str], env: Optional[Dict] = None) -> str:
    """執行子程序指令，即時在 Terminal 顯示輸出並同步寫入 log.txt。"""
    header = f"\n[EXEC] {cmd}\n" + "-" * 60
    print(header, flush=True)
    output_lines = []

    run_env = (env or os.environ).copy()
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


def safe_rmtree(target_dir: str):
    """安全刪除可能含有唯讀檔案的資料夾。"""
    if not target_dir or not os.path.exists(target_dir):
        return

    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IRWXU)
            func(path)
        except Exception:
            pass

    shutil.rmtree(target_dir, onerror=_onerror)


def safe_copy(src: str, dst: str):
    """複製檔案內容但不複製來源的唯讀鎖定屬性，並確保目標檔案可讀寫 (0644)。"""
    if os.path.exists(dst):
        try:
            os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    shutil.copyfile(src, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass


def prepare_ckpt_for_demo(
    data_name: str, diy_name: str, req_step: int, log_files: List[str]
) -> Tuple[int, List[str], List[str]]:
    """確認權重已生成並複製為 pretrained.pth.tar，回傳 (步數, 權重資料夾清單, 權重檔案清單)。"""
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

        if os.path.exists(target_ckpt):
            pretrained_alias = os.path.join(d, "pretrained.pth.tar")
            safe_copy(target_ckpt, pretrained_alias)
            abs_d = os.path.abspath(d)
            abs_target = os.path.abspath(target_ckpt)
            abs_alias = os.path.abspath(pretrained_alias)
            ckpt_dirs.append(abs_d)
            ckpt_files.extend([abs_target, abs_alias])
            log_and_print(f"[Auto-Ckpt] 已將訓練權重 {abs_target} 對接至 {abs_alias}", log_files)

    if not ckpt_dirs:
        raise FileNotFoundError(
            f"在 ./logs/{data_name}/ 下找不到任何 train-step*.pth.tar 權重檔！代表 Step 2 訓練未成功完成。"
        )
    return chosen_step, ckpt_dirs, ckpt_files


def prepare_mock_unlabeled_pool_from_mvtec(
    mvtec_cat_dir: str, mock_pool_dir: str, log_files: List[str], defect_limit: int = 8
):
    """
    從 MVTec 建立模擬無 GT 盲池：
    保留原始圖片名稱 (例如 000.png)，並在前方補上說明前綴：
    - 正常樣本: clean_<原始檔名> (例如 clean_000.png)
    - 瑕疵樣本: escape_<瑕疵類別>_<原始檔名> (例如 escape_crack_025.png)
    """
    safe_rmtree(mock_pool_dir)
    os.makedirs(mock_pool_dir, exist_ok=True)
    test_dir = os.path.join(mvtec_cat_dir, "test")

    total_count = 0
    good_dir = os.path.join(test_dir, "good")
    for f in sorted(os.listdir(good_dir)):
        if f.lower().endswith(IMG_EXTENSIONS):
            new_name = f"clean_{f}"
            safe_copy(os.path.join(good_dir, f), os.path.join(mock_pool_dir, new_name))
            total_count += 1

    defect_count = 0
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good":
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for f in sorted(os.listdir(sub_dir)):
            if f.lower().endswith(IMG_EXTENSIONS) and defect_count < defect_limit:
                new_name = f"escape_{sub}_{f}"
                safe_copy(os.path.join(sub_dir, f), os.path.join(mock_pool_dir, new_name))
                defect_count += 1
                total_count += 1

    log_and_print(
        f"[模擬無GT盲池建立完成] 共 {total_count} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵，保留原圖檔名並加上前綴) "
        f"-> {os.path.abspath(mock_pool_dir)}",
        log_files,
    )


# =====================================================================
# 高解析度 5000x5000 切塊 (Tiling) 與無縫縫合 (Feathering Stitching) 模組
# =====================================================================

def get_axis_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """計算滑動視窗座標，最後一格自動回推對齊邊界，確保所有切塊維持固定正方形不變形。"""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size, stride))
    if not positions or positions[-1] + tile_size < length:
        positions.append(length - tile_size)
    return positions


def is_background_tile(tile_rgb: np.ndarray, bg_std_thresh: float) -> bool:
    """判斷是否為純黑/純白無資訊背景塊。"""
    if bg_std_thresh <= 0:
        return False
    return float(np.std(tile_rgb)) < bg_std_thresh


def prepare_tiled_golden_trainset(
    shot_dir: str,
    category: str,
    tile_size: int,
    tile_overlap: int,
    max_train_tiles: int,
    bg_std_thresh: float,
    log_files: List[str],
) -> int:
    """
    將 Step 1 挑出的 4 張高解析度 Golden Image 切分為原生解析度子圖塊，
    並在相同空間位置跨 Golden 圖取中位數純淨塊，再分層抽樣出代表性訓練集。
    """
    cat_train_dir = os.path.join(shot_dir, "train", category)
    golden_files = sorted([
        os.path.join(cat_train_dir, f) for f in os.listdir(cat_train_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])
    if not golden_files:
        raise FileNotFoundError(f"在 {cat_train_dir} 中找不到 Golden 圖片！")

    full_backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    os.makedirs(full_backup_dir, exist_ok=True)
    golden_imgs = []
    for p in golden_files:
        safe_copy(p, os.path.join(full_backup_dir, os.path.basename(p)))
        img = np.array(Image.open(p).convert("RGB"))
        golden_imgs.append(img)

    h0, w0 = golden_imgs[0].shape[:2]
    if h0 <= tile_size and w0 <= tile_size:
        log_and_print(
            f"[Tiling] 原始圖片尺寸 ({w0}x{h0}) <= tile_size ({tile_size})，保持原圖訓練。",
            log_files,
        )
        return len(golden_files)

    stride = max(1, tile_size - tile_overlap)
    ys = get_axis_positions(h0, tile_size, stride)
    xs = get_axis_positions(w0, tile_size, stride)
    grid_coords = [(y, x) for y in ys for x in xs]

    valid_candidates = []
    for (y, x) in grid_coords:
        patches = []
        for g_img in golden_imgs:
            gh, gw = g_img.shape[:2]
            py = min(y, max(0, gh - tile_size))
            px = min(x, max(0, gw - tile_size))
            ph = min(tile_size, gh - py)
            pw = min(tile_size, gw - px)
            crop = g_img[py:py + ph, px:px + pw]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            patches.append(crop)

        non_bg = [p for p in patches if not is_background_tile(p, bg_std_thresh)]
        if not non_bg:
            continue

        if len(non_bg) == 1:
            best_patch = non_bg[0]
        else:
            small = np.stack([
                cv2.resize(p, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
                for p in non_bg
            ], axis=0)
            dist_mat = np.abs(small[:, None, :] - small[None, :, :]).mean(axis=-1)
            medoid_idx = int(np.argmin(dist_mat.sum(axis=1)))
            best_patch = non_bg[medoid_idx]

        cy, cx = y + tile_size * 0.5, x + tile_size * 0.5
        radius = np.hypot(cy - h0 * 0.5, cx - w0 * 0.5)
        valid_candidates.append((radius, y, x, best_patch))

    if not valid_candidates:
        log_and_print("[Tiling] 警告：所有切塊皆低於背景門檻，退回保留全部切塊。", log_files)
        for (y, x) in grid_coords:
            crop = golden_imgs[0][y:y + tile_size, x:x + tile_size]
            valid_candidates.append((0.0, y, x, crop))

    valid_candidates.sort(key=lambda item: item[0])
    if len(valid_candidates) > max_train_tiles:
        indices = np.linspace(0, len(valid_candidates) - 1, max_train_tiles, dtype=int)
        selected_tiles = [valid_candidates[i] for i in indices]
    else:
        selected_tiles = valid_candidates

    if len(selected_tiles) < min(max_train_tiles, len(golden_imgs) * len(valid_candidates)):
        extra_needed = max_train_tiles - len(selected_tiles)
        for g_img in golden_imgs[1:]:
            for (_, y, x, _) in valid_candidates:
                if extra_needed <= 0:
                    break
                crop = g_img[y:y + tile_size, x:x + tile_size]
                if not is_background_tile(crop, bg_std_thresh):
                    selected_tiles.append((0.0, y, x, crop))
                    extra_needed -= 1

    for f in os.listdir(cat_train_dir):
        if f.lower().endswith(IMG_EXTENSIONS):
            p = os.path.join(cat_train_dir, f)
            try:
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            os.remove(p)

    for idx, (_, y, x, patch) in enumerate(selected_tiles):
        out_path = os.path.join(cat_train_dir, f"{idx:03d}_tile_y{y:04d}_x{x:04d}.png")
        Image.fromarray(patch).save(out_path)

    log_and_print(
        f"[Tiling-Train] 已將 {len(golden_files)} 張 ({w0}x{h0}) Golden 大圖無損切分為 "
        f"{len(selected_tiles)} 張 ({tile_size}x{tile_size}) 高解析度代表性訓練圖塊 -> {os.path.abspath(cat_train_dir)}",
        log_files,
    )
    return len(selected_tiles)


def prepare_tiled_test_pool(
    pool_dir: str,
    tiled_pool_dir: str,
    manifest_save_dir: str,
    tile_size: int,
    tile_overlap: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    log_files: List[str],
) -> Dict:
    """
    將測試盲池中的所有圖片切分為重疊子圖塊，並確保 tiled_pool_dir 內只有純圖片 (.png)，
    將座標映射表 tile_manifest.json 儲存於 manifest_save_dir 中。
    """
    safe_rmtree(tiled_pool_dir)
    os.makedirs(tiled_pool_dir, exist_ok=True)

    stride = max(1, tile_size - tile_overlap)
    img_files = sorted([
        f for f in os.listdir(pool_dir) if f.lower().endswith(IMG_EXTENSIONS)
    ])
    if not img_files:
        raise FileNotFoundError(f"在 {pool_dir} 中找不到任何圖片！")

    manifest = {
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "images": {},
    }
    total_tiles = 0
    skipped_bg_tiles = 0

    for fname in img_files:
        fpath = os.path.join(pool_dir, fname)
        stem = os.path.splitext(fname)[0]
        img = np.array(Image.open(fpath).convert("RGB"))
        h, w = img.shape[:2]

        if disable_tiling or (h <= tile_size and w <= tile_size):
            ys, xs = [0], [0]
            eff_tile_h, eff_tile_w = h, w
        else:
            ys = get_axis_positions(h, tile_size, stride)
            xs = get_axis_positions(w, tile_size, stride)
            eff_tile_h, eff_tile_w = tile_size, tile_size

        tiles_info = []
        for y in ys:
            for x in xs:
                ch = min(eff_tile_h, h - y)
                cw = min(eff_tile_w, w - x)
                crop = img[y:y + ch, x:x + cw]

                if is_background_tile(crop, bg_std_thresh) and (len(ys) * len(xs) > 1):
                    skipped_bg_tiles += 1
                    continue

                tile_stem = f"{stem}__tile_y{y:04d}_x{x:04d}_h{ch:04d}_w{cw:04d}"
                tile_fname = f"{tile_stem}.png"
                Image.fromarray(crop).save(os.path.join(tiled_pool_dir, tile_fname))

                tiles_info.append({
                    "tile_stem": tile_stem,
                    "tile_fname": tile_fname,
                    "y": int(y),
                    "x": int(x),
                    "h": int(ch),
                    "w": int(cw),
                })
                total_tiles += 1

        if not tiles_info:
            cy, cx = max(0, (h - eff_tile_h) // 2), max(0, (w - eff_tile_w) // 2)
            ch, cw = min(eff_tile_h, h - cy), min(eff_tile_w, w - cx)
            crop = img[cy:cy + ch, cx:cx + cw]
            tile_stem = f"{stem}__tile_y{cy:04d}_x{cx:04d}_h{ch:04d}_w{cw:04d}"
            tile_fname = f"{tile_stem}.png"
            Image.fromarray(crop).save(os.path.join(tiled_pool_dir, tile_fname))
            tiles_info.append({
                "tile_stem": tile_stem,
                "tile_fname": tile_fname,
                "y": int(cy),
                "x": int(cx),
                "h": int(ch),
                "w": int(cw),
            })
            total_tiles += 1

        manifest["images"][fname] = {
            "stem": stem,
            "path": os.path.abspath(fpath),
            "height": int(h),
            "width": int(w),
            "tiles": tiles_info,
        }

    manifest["total_tiles"] = total_tiles
    manifest["num_images"] = len(img_files)
    manifest_path = os.path.join(manifest_save_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log_and_print(
        f"[Tiling-Test] 已將 {len(img_files)} 張原圖準備為 {total_tiles} 張推論圖塊 "
        f"(自動跳過 {skipped_bg_tiles} 張純背景角落塊) -> {os.path.abspath(tiled_pool_dir)}",
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


def stitch_single_image_amap(
    img_meta: Dict,
    tile_amap_dir: str,
    tile_overlap: int,
) -> np.ndarray:
    """將單張大圖的所有子圖塊 .npy 無縫縫合回原圖完整解析度 (H, W)。"""
    h, w = img_meta["height"], img_meta["width"]
    canvas_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    weight_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for t in img_meta["tiles"]:
        npy_path = find_tile_npy(tile_amap_dir, t["tile_stem"], t["tile_fname"])
        if not npy_path:
            continue
        amap = np.load(npy_path).astype(np.float32)
        amap = np.squeeze(amap)
        if amap.ndim != 2:
            continue

        th, tw = t["h"], t["w"]
        if amap.shape[0] != th or amap.shape[1] != tw:
            amap = cv2.resize(amap, (tw, th), interpolation=cv2.INTER_CUBIC)

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

    return full_amap


def merge_nearby_bboxes(
    bboxes: List[Tuple[int, int, int, int, float, int]], merge_gap: int
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
    full_amap: np.ndarray,
    img_threshold: float,
    core_ratio: float = 0.55,
) -> Tuple[List[Tuple[int, int, int, int, float, int]], float]:
    """
    峰值自適應核心收斂 (Peak-Adaptive Core Thresholding) 演算法：
    1. 抑制影像最外圍 1.5% 邊框 Padding 偽影。
    2. 像素分割門檻 pixel_thresh = max(img_threshold, median + core_ratio * (max_score - median))，
       消除低分背景雜訊框，同時防止高斯平滑光暈把整顆物件包進同一個大框。
    3. 針對連通塊進行面積、峰值強度篩選，並自動合併鄰近框。
    """
    h, w = full_amap.shape[:2]
    work_map = full_amap.copy()

    border_y = max(2, int(round(h * 0.015)))
    border_x = max(2, int(round(w * 0.015)))
    img_median = float(np.median(work_map))
    work_map[:border_y, :] = img_median
    work_map[-border_y:, :] = img_median
    work_map[:, :border_x] = img_median
    work_map[:, -border_x:] = img_median

    max_score = float(np.max(work_map))
    if max_score < img_threshold:
        return [], img_threshold

    core_thresh = img_median + core_ratio * (max_score - img_median)
    pixel_box_thresh = max(float(img_threshold), float(core_thresh))

    binary_mask = (work_map >= pixel_box_thresh).astype(np.uint8) * 255
    k_size = max(3, int(round(max(h, w) / 600)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = max(12, int(round((h * w) * 0.00008)))
    vis_scale = max(1.0, max(h, w) / 900.0)
    pad = max(3, int(round(4 * vis_scale)))

    raw_bboxes = []
    for lbl_idx in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl_idx]
        if area < min_area:
            continue
        region_scores = work_map[by:by + bh, bx:bx + bw]
        region_max = float(np.max(region_scores))

        if region_max < max(img_threshold, img_median + 0.50 * (max_score - img_median)):
            continue

        x1, y1 = max(0, int(bx - pad)), max(0, int(by - pad))
        x2, y2 = min(w - 1, int(bx + bw + pad)), min(h - 1, int(by + bh + pad))
        raw_bboxes.append((x1, y1, x2, y2, region_max, int(area)))

    if not raw_bboxes:
        peak_y, peak_x = np.unravel_index(int(np.argmax(work_map)), work_map.shape)
        half_r = max(12, int(round(max(h, w) * 0.025)))
        x1, y1 = max(0, int(peak_x - half_r)), max(0, int(peak_y - half_r))
        x2, y2 = min(w - 1, int(peak_x + half_r)), min(h - 1, int(peak_y + half_r))
        raw_bboxes.append((x1, y1, x2, y2, max_score, (x2 - x1) * (y2 - y1)))

    merge_gap = max(6, int(round(max(h, w) * 0.015)))
    merged_bboxes = merge_nearby_bboxes(raw_bboxes, merge_gap=merge_gap)
    return merged_bboxes, pixel_box_thresh


def stitch_and_filter_anomalies(
    manifest: Dict,
    tile_amap_dir: str,
    final_amap_dir: str,
    out_dir: str,
    golden_filenames: List[str],
    user_threshold: float,
    sigma_k: float,
    core_ratio: float,
    log_files: List[str],
) -> Tuple[int, int, str, str]:
    """
    1. 縫合每張圖片的熱力圖。
    2. 判斷每張圖的最高分是否 >= user_threshold：
       - 若 >= threshold：判定為異常 (ANOMALY)，精準繪製紅框並存入 anomaly_images/ 資料夾。
       - 若 < threshold：判定為正常 (PASS)，直接不畫框、不輸出圖片。
    """
    os.makedirs(final_amap_dir, exist_ok=True)
    anomaly_dir = os.path.join(out_dir, "anomaly_images")
    anomaly_fullres_bbox_dir = os.path.join(anomaly_dir, "full_res_bbox_only")
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(anomaly_fullres_bbox_dir, exist_ok=True)

    tile_overlap = manifest["tile_overlap"]
    golden_set = set(golden_filenames)

    log_and_print("\n[Step 4.1] 正在將子圖塊無縫拼接回原圖完整解析度...", log_files)
    stitched_info = []
    golden_max_scores = []

    for fname, meta in manifest["images"].items():
        full_amap = stitch_single_image_amap(meta, tile_amap_dir, tile_overlap)
        h, w = meta["height"], meta["width"]
        border_y = max(2, int(round(h * 0.015)))
        border_x = max(2, int(round(w * 0.015)))
        inner_amap = full_amap[border_y:h - border_y, border_x:w - border_x]
        max_score = float(np.max(inner_amap)) if inner_amap.size > 0 else float(np.max(full_amap))

        top_k_count = max(10, int(full_amap.size * 0.0005))
        top_k_score = float(np.mean(np.partition(full_amap.reshape(-1), -top_k_count)[-top_k_count:]))

        if fname in golden_set:
            golden_max_scores.append(max_score)

        stitched_info.append({
            "filename": fname,
            "stem": meta["stem"],
            "meta": meta,
            "full_amap": full_amap,
            "max_score": max_score,
            "top_k_score": top_k_score,
        })

    if user_threshold > 0:
        effective_threshold = float(user_threshold)
        thresh_mode_str = f"使用者指定 Threshold = {effective_threshold:.4f}"
    else:
        if golden_max_scores:
            g_mu = float(np.mean(golden_max_scores))
            g_std = float(np.std(golden_max_scores)) + 1e-4
            effective_threshold = g_mu + sigma_k * g_std
        else:
            all_max = [x["max_score"] for x in stitched_info]
            effective_threshold = float(np.percentile(all_max, 75.0))
        thresh_mode_str = f"自動計算 Threshold = {effective_threshold:.4f}"

    log_and_print(f"[Step 4.2] 異常判定門檻設定: {thresh_mode_str} (低於此門檻之圖片將不畫框、不輸出)", log_files)

    stitched_info.sort(key=lambda item: item["max_score"], reverse=True)

    ranking_records = []
    anomaly_count = 0
    skipped_normal_count = 0

    for rank_idx, item in enumerate(stitched_info, 1):
        fname = item["filename"]
        stem = item["stem"]
        meta = item["meta"]
        full_amap = item["full_amap"]
        max_score = item["max_score"]
        top_k_score = item["top_k_score"]
        h, w = meta["height"], meta["width"]
        is_golden = fname in golden_set

        if max_score < effective_threshold:
            skipped_normal_count += 1
            status_tag = "GOLDEN_PASS" if is_golden else "PASS_SKIPPED"
            ranking_records.append({
                "rank": rank_idx,
                "filename": fname,
                "status": status_tag,
                "max_score": max_score,
                "top_k_score": top_k_score,
                "num_bboxes": 0,
                "saved_file": "(未超過 threshold，不輸出圖片)",
                "resolution": f"{w}x{h}",
            })
            continue

        anomaly_count += 1
        save_dtype = np.float16 if max(full_amap.shape) > 2048 else np.float32
        np.save(
            os.path.join(final_amap_dir, f"Rank{rank_idx:03d}_Score{max_score:.4f}_{stem}.npy"),
            full_amap.astype(save_dtype),
        )

        bboxes, pixel_box_thresh = extract_tight_bboxes_for_anomaly_image(
            full_amap=full_amap,
            img_threshold=effective_threshold,
            core_ratio=core_ratio,
        )

        orig_bgr = cv2.imread(meta["path"])
        if orig_bgr is None:
            orig_rgb = np.array(Image.open(meta["path"]).convert("RGB"))
            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

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
                bbox_canvas, label_txt, (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        img_med = float(np.median(full_amap))
        norm_map = np.clip((full_amap - img_med) / max(max_score - img_med, 1e-6), 0.0, 1.0)
        hm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(orig_bgr, 0.55, hm_color, 0.45, 0)

        prefixed_img_name = f"Rank{rank_idx:03d}_Score{max_score:.4f}_Box{len(bboxes):02d}_{fname}"

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
        banner_txt = (
            f"Orig: {fname} | MaxScore: {max_score:.4f} (ImgThresh: {effective_threshold:.4f}, "
            f"CoreBoxThresh: {pixel_box_thresh:.4f}) | Status: ANOMALY (Boxes: {len(bboxes)})"
        )
        cv2.putText(
            banner, banner_txt, (12, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, banner_h / 55.0), (0, 220, 255), 2, cv2.LINE_AA
        )
        final_vis = np.vstack([banner, tri_panel])
        vis_out_path = os.path.join(anomaly_dir, prefixed_img_name)
        cv2.imwrite(vis_out_path, final_vis)

        status_tag = "GOLDEN_ANOMALY" if is_golden else "ANOMALY"
        ranking_records.append({
            "rank": rank_idx,
            "filename": fname,
            "status": status_tag,
            "max_score": max_score,
            "top_k_score": top_k_score,
            "num_bboxes": len(bboxes),
            "saved_file": prefixed_img_name,
            "resolution": f"{w}x{h}",
        })

    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== FoundAD High-Res Anomaly Detection Report ===\n")
        f.write(
            f"Image Anomaly Threshold: {effective_threshold:.4f} | Core Box Ratio: {core_ratio:.2f} | "
            f"Exported Anomaly Images: {anomaly_count} | Skipped Normal Images: {skipped_normal_count}\n"
        )
        f.write("-" * 120 + "\n")
        f.write(
            f"{'Rank':<5} | {'Original_Filename':<28} | {'Status':<15} | {'Max_Score':<10} | "
            f"{'BBoxes':<6} | {'Saved_Anomaly_Filename'}\n"
        )
        f.write("-" * 120 + "\n")
        for r in ranking_records:
            f.write(
                f"{r['rank']:<5} | {r['filename']:<28} | {r['status']:<15} | "
                f"{r['max_score']:<10.4f} | {r['num_bboxes']:<6} | {r['saved_file']}\n"
            )

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Original_Filename", "Status", "Max_Score",
            "TopK_Mean_Score", "Num_BBoxes", "Saved_Anomaly_Filename", "Resolution"
        ])
        for r in ranking_records:
            writer.writerow([
                r["rank"], r["filename"], r["status"],
                f"{r['max_score']:.6f}", f"{r['top_k_score']:.6f}",
                r["num_bboxes"], r["saved_file"], r["resolution"],
            ])

    log_and_print(
        f"[Step 4.3] 篩選與畫框完成：共 {anomaly_count} 張超過門檻 ({effective_threshold:.4f}) 已畫框輸出；"
        f"其餘 {skipped_normal_count} 張正常圖已自動略過不輸出。",
        log_files,
    )
    return anomaly_count, skipped_normal_count, anomaly_dir, anomaly_fullres_bbox_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="./output/2_fab_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot")
    parser.add_argument("--category", type=str, default="auto")
    parser.add_argument("--app_train", type=str, default="train_dinov3")
    parser.add_argument("--app_test", type=str, default="test")
    parser.add_argument("--diy_name", type=str, default="sim_escape")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=4)

    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="判定異常的最高分數門檻 (超過此分數才畫框並輸出至異常資料夾；未超過則不輸出圖片)"
    )
    parser.add_argument(
        "--core_ratio", type=float, default=0.55,
        help="異常熱區峰值核心收斂比例 (0.45~0.65，越高紅框越緊密貼合最高分缺陷核心，避免大框包住整顆物件)"
    )
    parser.add_argument("--sigma_k", type=float, default=3.2, help="當 threshold<=0 時自動推算門檻的標準差倍數")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--mock_from_mvtec", type=str, default="")

    parser.add_argument("--tile_size", type=int, default=512, help="子圖塊像素大小 (建議 512 或 1024，避免大圖壓縮失真)")
    parser.add_argument("--tile_overlap", type=int, default=64, help="相鄰子圖塊重疊像素寬度 (用於無縫平滑拼接)")
    parser.add_argument("--max_train_tiles", type=int, default=16, help="從 Golden 大圖抽出的代表性高解析度訓練圖塊上限")
    parser.add_argument("--train_batch_size", type=int, default=8, help="訓練時的最大 Batch Size，防止高解析度圖塊 OOM")
    parser.add_argument("--bg_std_thresh", type=float, default=1.5, help="過濾純背景角落塊的標準差門檻")
    parser.add_argument("--disable_tiling", action="store_true", help="若設定此旗標則關閉切塊，退回整張推論模式")
    args = parser.parse_args()

    if not args.category or args.category.lower() == "auto":
        if args.mock_from_mvtec:
            args.category = os.path.basename(os.path.normpath(args.mock_from_mvtec))
        else:
            args.category = os.path.basename(os.path.normpath(args.pool_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    start_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_and_print(
        f"\n{'='*75}\n"
        f"[Task 2 無GT抓漏啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_out_dir}\n"
        f"{'='*75}",
        log_files,
    )

    if args.mock_from_mvtec:
        prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, log_files, defect_limit=8)

    safe_rmtree(shot_dir)

    # =========================================================================
    # [階段一：整體訓練階段] (包含 Golden 自動挑圖 + 高解析度切塊 + 模型訓練)
    # =========================================================================
    t_golden_start = time.perf_counter()
    golden_paths = select_golden_images(
        pool_dir=args.pool_dir,
        ckpt_path=args.ckpt,
        out_dir=shot_dir,
        category=args.category,
        shots=args.shots,
        report_dir=run_out_dir,
    )
    golden_select_sec = time.perf_counter() - t_golden_start
    golden_filenames = [os.path.basename(p) for p in golden_paths]
    log_and_print(
        f"[⏱️ 計時] Step 1 自動挑選 {len(golden_filenames)} 張 Golden Image 耗時: {format_duration(golden_select_sec)}",
        log_files,
    )

    t_train_tile_start = time.perf_counter()
    if not args.disable_tiling:
        num_train_samples = prepare_tiled_golden_trainset(
            shot_dir=shot_dir,
            category=args.category,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_train_tiles=args.max_train_tiles,
            bg_std_thresh=args.bg_std_thresh,
            log_files=log_files,
        )
    else:
        num_train_samples = args.shots
    train_tile_sec = time.perf_counter() - t_train_tile_start

    effective_batch_size = min(max(1, num_train_samples), args.train_batch_size)

    model_train_sec = 0.0
    if not args.skip_train:
        t_model_train_start = time.perf_counter()
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={effective_batch_size} "
            f"data.dataset=mvtec data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={shot_dir} "
            f"data.mvtec_classnames='[{args.category}]' "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        log_and_print(f"\n[Step 2] 執行 FoundAD 高解析度圖塊訓練 (batch_size={effective_batch_size}):", log_files)
        run_cmd_and_log(train_cmd, log_files)
        model_train_sec = time.perf_counter() - t_model_train_start
    else:
        log_and_print("[⏱️ 計時] 已設定 --skip_train，跳過 Step 2 模型訓練階段 (0.00 秒)", log_files)

    total_train_time = golden_select_sec + train_tile_sec + model_train_sec
    actual_step, ckpt_dirs, ckpt_files = prepare_ckpt_for_demo(
        args.data_name, args.diy_name, args.ckpt_step, log_files
    )

    train_tiles_dir = os.path.abspath(os.path.join(shot_dir, "train", args.category))
    golden_backup_dir = os.path.abspath(os.path.join(shot_dir, "golden_full_res_backup", args.category))

    train_done_lines = [
        f"\n{'='*75}",
        f" [✅ 訓練階段完成 — 時間統計與權重檔儲存位置]",
        f"{'-'*75}",
        f"  ⏱️ Step 1 自動挑選 Golden 圖耗時 : {format_duration(golden_select_sec)}",
        f"  ⏱️ Step 1.5 高解析度訓練切塊耗時 : {format_duration(train_tile_sec)}",
        f"  ⏱️ Step 2 FoundAD 模型訓練耗時   : {format_duration(model_train_sec)}{' (Skipped)' if args.skip_train else ''}",
        f"  ⏱️ 整體訓練階段總時間            : {format_duration(total_train_time)}",
        f"{'-'*75}",
        f"  📁 Golden 訓練圖塊資料夾         : {train_tiles_dir}",
        f"  📁 Golden 原圖備份資料夾         : {golden_backup_dir}",
        f"  📦 訓練權重檔儲存資料夾          : {ckpt_dirs[0]}",
    ]
    for cf in ckpt_files:
        train_done_lines.append(f"      └── {cf}")
    train_done_lines.append(f"{'='*75}")
    log_and_print("\n".join(train_done_lines), log_files)

    # =========================================================================
    # [階段二：整體推論階段] (包含測試池切塊 + 模型推論 + 門檻篩選與精準畫框)
    # =========================================================================
    amap_dump_dir = os.path.join(run_out_dir, "amap_npy")
    report_txt_path = os.path.join(run_out_dir, "escape_ranking_report.txt")
    report_csv_path = os.path.join(run_out_dir, "escape_ranking_report.csv")

    tiled_pool_dir = os.path.join(abs_datasets_dir, f"{args.data_name}_tiled_pool")
    tile_amap_dir = os.path.join(run_out_dir, "_tile_amap_tmp")
    tile_bbox_tmp = os.path.join(run_out_dir, "_tile_bbox_tmp")
    safe_rmtree(tile_amap_dir)
    safe_rmtree(tile_bbox_tmp)
    os.makedirs(tile_amap_dir, exist_ok=True)
    os.makedirs(tile_bbox_tmp, exist_ok=True)

    t_test_tile_start = time.perf_counter()
    manifest = prepare_tiled_test_pool(
        pool_dir=args.pool_dir,
        tiled_pool_dir=tiled_pool_dir,
        manifest_save_dir=run_out_dir,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        bg_std_thresh=args.bg_std_thresh,
        disable_tiling=args.disable_tiling,
        log_files=log_files,
    )
    test_tile_sec = time.perf_counter() - t_test_tile_start
    num_test_images = manifest.get("num_images", 1)
    total_test_tiles = manifest.get("total_tiles", num_test_images)

    # 關鍵修正：傳入合法的暫存目錄 tile_bbox_tmp，防止 AD.py 內部 os.makedirs("") 報錯崩潰
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FAB_AMAP_DUMP_DIR"] = os.path.abspath(tile_amap_dir)
    env["FAB_BBOX_OUT_DIR"] = os.path.abspath(tile_bbox_tmp)
    env["FAB_SIGMA_K"] = str(args.sigma_k)
    env["FAB_GOLDEN_NAMES"] = ""

    t_infer_start = time.perf_counter()
    demo_cmd = (
        f"python foundad/main.py mode=demo data.dataset=mvtec "
        f"data.data_name={args.data_name} diy_name={args.diy_name} "
        f"data.test_root={os.path.abspath(tiled_pool_dir)} app={args.app_test} app.ckpt_step={actual_step}"
    )
    log_and_print("\n[Step 3] 執行 FoundAD 子圖塊高解析度推論:", log_files)
    run_cmd_and_log(demo_cmd, log_files, env=env)
    model_infer_sec = time.perf_counter() - t_infer_start

    t_stitch_start = time.perf_counter()
    anomaly_count, skipped_count, anomaly_dir, anomaly_fullres_bbox_dir = stitch_and_filter_anomalies(
        manifest=manifest,
        tile_amap_dir=tile_amap_dir,
        final_amap_dir=amap_dump_dir,
        out_dir=run_out_dir,
        golden_filenames=golden_filenames,
        user_threshold=args.threshold,
        sigma_k=args.sigma_k,
        core_ratio=args.core_ratio,
        log_files=log_files,
    )
    stitch_vis_sec = time.perf_counter() - t_stitch_start

    # 清理子圖塊推論暫存檔
    safe_rmtree(tile_amap_dir)
    safe_rmtree(tile_bbox_tmp)
    for old_viz in glob.glob(f"./logs/{args.data_name}/**/bbox_visualizations", recursive=True):
        safe_rmtree(old_viz)

    total_infer_time = test_tile_sec + model_infer_sec + stitch_vis_sec
    avg_infer_per_img = total_infer_time / max(1, num_test_images)
    total_wall_time = time.perf_counter() - pipeline_start_t

    infer_done_summary = (
        f"\n{'='*75}\n"
        f" [✅ 推論階段完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-'*75}\n"
        f"  [1] 整體訓練階段總時間 (Total Training Time)  : {format_duration(total_train_time)}\n"
        f"      ├─ Step 1 自動挑選 Golden Image            : {format_duration(golden_select_sec)}\n"
        f"      ├─ Step 1.5 Golden 高解析度切塊            : {format_duration(train_tile_sec)}\n"
        f"      └─ Step 2 FoundAD 模型訓練 (2000 Epochs)   : {format_duration(model_train_sec)}"
        f"{' (Skipped)' if args.skip_train else ''}\n"
        f"{'-'*75}\n"
        f"  [2] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ Step 3.1 測試池高解析度切塊 ({total_test_tiles} 塊)   : {format_duration(test_tile_sec)}\n"
        f"      ├─ Step 3.2 FoundAD 模型推論               : {format_duration(model_infer_sec)}\n"
        f"      ├─ Step 4 全解析度縫合、門檻篩選與精準畫框 : {format_duration(stitch_vis_sec)}\n"
        f"      └─ 平均每張原圖推論耗時 ({num_test_images} 張原圖)         : {avg_infer_per_img:.2f} 秒/張\n"
        f"{'-'*75}\n"
        f"  [3] 任務全程總執行時間 (Total Wall Time)      : {format_duration(total_wall_time)}\n"
        f"{'-'*75}\n"
        f"  [4] 門檻篩選統計 (Threshold = {args.threshold:.4f}):\n"
        f"      🚨 超過門檻判定異常並畫框輸出 : {anomaly_count} 張\n"
        f"      ✅ 未達門檻判定正常直接略過   : {skipped_count} 張 (不輸出圖片)\n"
        f"{'-'*75}\n"
        f"  [5] 權重檔與本次推論結果儲存資料夾清單 (Saved Directories & Files):\n"
        f"      📦 訓練權重檔資料夾           : {ckpt_dirs[0]}\n"
        f"      📂 本次時間戳記推論主資料夾   : {run_out_dir}\n"
        f"          ├── 異常紅框三合一對照圖  : {anomaly_dir}\n"
        f"          ├── 異常1:1全解析度紅框圖 : {anomaly_fullres_bbox_dir}\n"
        f"          ├── 異常熱力圖矩陣(.npy)  : {amap_dump_dir}\n"
        f"          ├── 完整排名報表 (TXT)    : {report_txt_path}\n"
        f"          ├── 完整排名報表 (CSV)    : {report_csv_path}\n"
        f"          └── 本次執行日誌 (log.txt): {fixed_log_txt}\n"
        f"{'='*75}"
    )

    log_and_print(infer_done_summary, log_files)

    if os.path.exists(report_txt_path):
        with open(report_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(train_done_lines) + "\n")
            f.write(infer_done_summary + "\n")


if __name__ == "__main__":
    main()
