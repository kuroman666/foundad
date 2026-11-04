# 檔案位置: FoundAD/2_run_fab_no_gt.py (全選覆蓋並存檔)
import os
import re
import gc
import csv
import time
import stat
import glob
import json
import socket
import shutil
import hashlib
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from fab_tools.auto_golden import select_golden_images, scan_pool_images, IMG_EXTENSIONS


def format_duration(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS.xx 與總秒數。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f} ({seconds:.2f} 秒)"


def get_free_port() -> int:
    """由作業系統動態分配保證空閒的本地 Port，徹底避免 DDP Port 40112 衝突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def log_and_print(msg: str, log_files: List[str]):
    """同時在 Terminal 顯示並附加寫入 log.txt 檔案。"""
    print(msg, flush=True)
    for lf in log_files:
        if lf:
            with open(lf, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


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


def compute_pool_signature(pool_dir: str) -> str:
    """遞迴計算測試盲池所有有效影像之 (相對路徑, 檔案大小) MD5 指紋，用於判斷資料池是否變動。"""
    if not os.path.isdir(pool_dir):
        return ""
    img_paths = scan_pool_images(pool_dir)
    h = hashlib.md5()
    h.update(os.path.abspath(pool_dir).encode("utf-8"))
    for fpath in img_paths:
        rel_path = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        try:
            sz = os.path.getsize(fpath)
        except OSError:
            sz = 0
        h.update(f"{rel_path}:{sz};".encode("utf-8"))
    return h.hexdigest()


def compute_ckpt_signature(ckpt_files: List[str]) -> str:
    """計算實際載入之模型權重檔指紋 (路徑 + 檔案大小 + 修改時間)，確保重新訓練後自動更新推論快取。"""
    h = hashlib.md5()
    for p in sorted(ckpt_files):
        if os.path.basename(p) == "pretrained.pth.tar":
            continue
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{os.path.abspath(p)}:{st.st_size}:{int(st.st_mtime)};".encode("utf-8"))
    return h.hexdigest()


def _robust_normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """使用 P0.1 ~ P99.9 穩健百分位數將 16-bit / float 陣列轉為 uint8，防止單點飽和亮點壓扁對比度。"""
    arr_f = arr.astype(np.float32)
    p_low, p_high = np.percentile(arr_f, [0.1, 99.9])
    if p_high <= p_low:
        p_low, p_high = float(np.min(arr_f)), float(np.max(arr_f))
    if p_high > p_low:
        arr_f = (arr_f - p_low) / (p_high - p_low) * 255.0
    else:
        arr_f = np.zeros_like(arr_f, dtype=np.float32)
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def load_image_rgb_uint8(img_path: str) -> np.ndarray:
    """
    安全讀取影像並統一轉為 (H, W, 3) uint8 RGB 格式。
    自動相容晶圓廠常見之 16-bit TIFF (.tif/.tiff)、單通道灰階或 8-bit 圖檔。
    """
    try:
        with Image.open(img_path) as pil_img:
            if pil_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                arr_u8 = _robust_normalize_to_uint8(np.array(pil_img))
                return np.stack([arr_u8, arr_u8, arr_u8], axis=-1)
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)
    except Exception:
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"無法讀取影像檔案: {img_path}")
        if raw.dtype != np.uint8:
            raw = _robust_normalize_to_uint8(raw)
        if raw.ndim == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        if raw.shape[2] == 4:
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def load_image_bgr_uint8(img_path: str) -> np.ndarray:
    """安全讀取影像並統一轉為 (H, W, 3) uint8 BGR 格式供 OpenCV 畫圖與疊圖使用。"""
    rgb = load_image_rgb_uint8(img_path)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
    將 Step 1 挑出的 4 張高解析度 Golden Image 切分為原生解析度子圖塊：
    - 對每個空間座標 (y, x) 計算跨片 L1 距離矩陣，優先選出最純淨的 Medoid 圖塊。
    - 當需要補充跨片公差圖塊時，自動排除距離最遠的離群圖塊 (Outlier Rejection)，防止局部微瑕疵混入訓練集。
    - 處理完畢立即釋放 Golden 原圖陣列，降低記憶體佔用。
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
        img = load_image_rgb_uint8(p)
        golden_imgs.append(img)

    h0, w0 = golden_imgs[0].shape[:2]
    if h0 <= tile_size and w0 <= tile_size:
        log_and_print(
            f"[Tiling] 原始圖片尺寸 ({w0}x{h0}) <= tile_size ({tile_size})，保持原圖訓練。",
            log_files,
        )
        del golden_imgs
        gc.collect()
        return len(golden_files)

    stride = max(1, tile_size - tile_overlap)
    ys = get_axis_positions(h0, tile_size, stride)
    xs = get_axis_positions(w0, tile_size, stride)
    grid_coords = [(y, x) for y in ys for x in xs]

    valid_candidates = []
    for (y, x) in grid_coords:
        non_bg_items = []
        for g_idx, g_img in enumerate(golden_imgs):
            gh, gw = g_img.shape[:2]
            py = min(y, max(0, gh - tile_size))
            px = min(x, max(0, gw - tile_size))
            ph = min(tile_size, gh - py)
            pw = min(tile_size, gw - px)
            crop = g_img[py:py + ph, px:px + pw]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            if not is_background_tile(crop, bg_std_thresh):
                non_bg_items.append((g_idx, crop))

        if not non_bg_items:
            continue

        if len(non_bg_items) == 1:
            best_g_idx = non_bg_items[0][0]
            safe_extra_indices = []
        else:
            small = np.stack([
                cv2.resize(item[1], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
                for item in non_bg_items
            ], axis=0)
            dist_sum = np.abs(small[:, None, :] - small[None, :, :]).mean(axis=-1).sum(axis=1)
            sorted_order = np.argsort(dist_sum)
            best_g_idx = non_bg_items[int(sorted_order[0])][0]
            if len(sorted_order) >= 3:
                safe_extra_indices = [non_bg_items[int(i)][0] for i in sorted_order[1:-1]]
            else:
                safe_extra_indices = [non_bg_items[int(i)][0] for i in sorted_order[1:]]

        cy, cx = y + tile_size * 0.5, x + tile_size * 0.5
        radius = np.hypot(cy - h0 * 0.5, cx - w0 * 0.5)
        valid_candidates.append((radius, y, x, best_g_idx, safe_extra_indices))

    if not valid_candidates:
        log_and_print("[Tiling] 警告：所有切塊皆低於背景門檻，退回保留全部切塊。", log_files)
        for (y, x) in grid_coords:
            valid_candidates.append((0.0, y, x, 0, []))

    valid_candidates.sort(key=lambda item: item[0])
    if len(valid_candidates) > max_train_tiles:
        indices = np.linspace(0, len(valid_candidates) - 1, max_train_tiles, dtype=int)
        selected_specs = [
            (valid_candidates[i][1], valid_candidates[i][2], valid_candidates[i][3])
            for i in indices
        ]
    else:
        selected_specs = [(y, x, best_g_idx) for (_, y, x, best_g_idx, _) in valid_candidates]

    if len(selected_specs) < max_train_tiles:
        extra_needed = max_train_tiles - len(selected_specs)
        for (_, y, x, _, safe_extra_indices) in valid_candidates:
            if extra_needed <= 0:
                break
            for extra_g_idx in safe_extra_indices:
                if extra_needed <= 0:
                    break
                selected_specs.append((y, x, extra_g_idx))
                extra_needed -= 1

    for f in os.listdir(cat_train_dir):
        if f.lower().endswith(IMG_EXTENSIONS):
            p = os.path.join(cat_train_dir, f)
            try:
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            os.remove(p)

    for idx, (y, x, g_idx) in enumerate(selected_specs):
        g_img = golden_imgs[g_idx]
        gh, gw = g_img.shape[:2]
        py = min(y, max(0, gh - tile_size))
        px = min(x, max(0, gw - tile_size))
        ph = min(tile_size, gh - py)
        pw = min(tile_size, gw - px)
        patch = g_img[py:py + ph, px:px + pw]
        if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
            patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(cat_train_dir, f"{idx:03d}_g{g_idx}_tile_y{y:04d}_x{x:04d}.png")
        Image.fromarray(patch).save(out_path)

    num_saved = len(selected_specs)
    del golden_imgs
    gc.collect()

    log_and_print(
        f"[Tiling-Train] 已將 {len(golden_files)} 張 ({w0}x{h0}) Golden 大圖無損切分為 "
        f"{num_saved} 張 ({tile_size}x{tile_size}) 高解析度代表性訓練圖塊 (已啟用跨片離群瑕疵過濾) "
        f"-> {os.path.abspath(cat_train_dir)}",
        log_files,
    )
    return num_saved


def prepare_tiled_test_pool(
    pool_dir: str,
    tiled_pool_dir: str,
    manifest_save_dir: str,
    tile_size: int,
    tile_overlap: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    pool_signature: str,
    tile_config_key: str,
    cached_manifest_path: str,
    log_files: List[str],
) -> Dict:
    """
    遞迴搜尋測試盲池 (含所有子資料夾) 中的所有圖片並切分為重疊子圖塊，
    確保 tiled_pool_dir 內只有純圖片 (.png)，同時將座標映射表儲存於 run_out_dir 與持久化快取中。
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
        "images": {},
    }
    total_tiles = 0
    skipped_bg_tiles = 0

    for fpath in img_paths:
        rel_fname = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_fname)[0]
        stem = re.sub(r"[\\/]+", "__", rel_no_ext)

        img = load_image_rgb_uint8(fpath)
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

        manifest["images"][rel_fname] = {
            "stem": stem,
            "path": os.path.abspath(fpath),
            "height": int(h),
            "width": int(w),
            "tiles": tiles_info,
        }
        del img

    gc.collect()
    manifest["total_tiles"] = total_tiles
    manifest["num_images"] = len(img_paths)

    manifest_path = os.path.join(manifest_save_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if cached_manifest_path:
        os.makedirs(os.path.dirname(cached_manifest_path), exist_ok=True)
        safe_copy(manifest_path, cached_manifest_path)

    log_and_print(
        f"[Tiling-Test] 已遞迴將 {len(img_paths)} 張原圖準備為 {total_tiles} 張推論圖塊 "
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


def is_tile_amap_cache_valid(
    tile_amap_cache_dir: str,
    cached_manifest_path: str,
    pool_signature: str,
    tile_config_key: str,
    ckpt_signature: str,
    actual_step: int,
    diy_name: str,
) -> Optional[Dict]:
    """
    檢查子圖塊推論熱力圖 (.npy) 快取是否完整可用：
    驗證 (1) 測試池指紋 (2) 切塊參數 (3) 權重檔指紋與訓練步數 (4) 所有子圖塊 .npy 皆存在。
    """
    cache_info_path = os.path.join(tile_amap_cache_dir, "cache_info.json")
    if not os.path.isdir(tile_amap_cache_dir) or not os.path.isfile(cache_info_path) or not os.path.isfile(cached_manifest_path):
        return None
    try:
        with open(cache_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if (
            info.get("pool_signature") != pool_signature
            or info.get("tile_config_key") != tile_config_key
            or info.get("ckpt_signature") != ckpt_signature
            or int(info.get("actual_step", -1)) != int(actual_step)
            or info.get("diy_name") != diy_name
        ):
            return None

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
                if not find_tile_npy(tile_amap_cache_dir, t["tile_stem"], t["tile_fname"]):
                    return None
        return manifest
    except Exception:
        return None


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


# =====================================================================
# 防誤報核心演算法：晶圓圓周邊緣抑制 + 微塵單點尖峰濾除 + 動態門檻對齊
# =====================================================================

def detect_circular_wafer_geometry(
    img_path: str, h: int, w: int, wafer_edge_mode: str, category: str
) -> Tuple[bool, float, float, float]:
    """
    偵測影像是否為圓形晶圓/圓形工件，並回傳 (是否啟用圓周抑制, 圓心 cy, 圓心 cx, 半徑 R)。
    """
    mode = wafer_edge_mode.lower().strip()
    if mode == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5

    cat_lower = category.lower()
    is_circular_cat = any(k in cat_lower for k in ["wafer", "bottle", "cable", "coin", "lens", "circle"])

    scale = 256.0 / max(h, w)
    sh, sw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
    try:
        full_rgb = load_image_rgb_uint8(img_path)
        small_rgb = cv2.resize(full_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        del full_rgb
        gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        corner_sz = max(4, min(sh, sw) // 10)
        corners = np.concatenate([
            blur[:corner_sz, :corner_sz].reshape(-1),
            blur[:corner_sz, -corner_sz:].reshape(-1),
            blur[-corner_sz:, :corner_sz].reshape(-1),
            blur[-corner_sz:, -corner_sz:].reshape(-1),
        ])
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
                area = cv2.contourArea(c)
                if area > 0.18 * (sh * sw):
                    (cx_s, cy_s), r_s = cv2.minEnclosingCircle(c)
                    circle_area = np.pi * (r_s ** 2) + 1e-6
                    circularity_fill = area / circle_area
                    if (
                        0.60 <= circularity_fill <= 1.15
                        and abs(cx_s - sw * 0.5) < 0.18 * sw
                        and abs(cy_s - sh * 0.5) < 0.18 * sh
                        and r_s <= 0.58 * max(sh, sw)
                    ):
                        if mode == "on" or is_circular_cat or corner_std < 28.0:
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
    golden_tol_weight: float = 0.35,
) -> np.ndarray:
    """
    四合一防誤報熱力圖純化器：
    1. 矩形外圍 Padding 偽影抑制 (border_ratio)。
    2. 晶圓圓周邊緣 / 倒角區 (Wafer Bevel / Edge Ring) 自動餘弦衰減抑制。
    3. 灰階形態學開運算 (Grayscale Morphological Opening) 濾除單一 Patch 微塵尖峰噪點。
    4. 跨 Golden 結構複雜區容差補償。
    """
    h, w = full_amap.shape[:2]
    work_map = full_amap.astype(np.float32).copy()
    img_median = float(np.median(work_map))

    if golden_spatial_ref is not None and golden_tol_weight > 0:
        if golden_spatial_ref.shape[:2] != (h, w):
            g_ref = cv2.resize(golden_spatial_ref, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            g_ref = golden_spatial_ref
        g_med = float(np.median(g_ref))
        structural_excess = np.maximum(0.0, g_ref - g_med)
        work_map = img_median + np.maximum(0.0, (work_map - img_median) - golden_tol_weight * structural_excess)

    if speckle_filter_ratio > 0:
        k_speckle = max(3, int(round(max(h, w) * speckle_filter_ratio)) | 1)
        if k_speckle >= 3:
            disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_speckle, k_speckle))
            opened_map = cv2.morphologyEx(work_map, cv2.MORPH_OPEN, disk_kernel)
            work_map = 0.75 * opened_map + 0.25 * cv2.GaussianBlur(work_map, (k_speckle, k_speckle), 0)

    if border_ratio > 0:
        by = max(2, int(round(h * border_ratio)))
        bx = max(2, int(round(w * border_ratio)))
        work_map[:by, :] = img_median
        work_map[-by:, :] = img_median
        work_map[:, :bx] = img_median
        work_map[:, -bx:] = img_median

    use_circle, cy, cx, radius = detect_circular_wafer_geometry(
        img_path=img_path, h=h, w=w, wafer_edge_mode=wafer_edge_mode, category=category
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
    refined_amap: np.ndarray,
    img_threshold: float,
    core_ratio: float = 0.55,
    min_area_ratio: float = 0.00012,
    raw_score_mode: bool = False,
) -> Tuple[List[Tuple[int, int, int, int, float, int]], float]:
    """
    峰值自適應核心收斂 (Peak-Adaptive Core Thresholding) 演算法：
    - 針對通過門檻的異常熱力圖切出核心缺陷紅框。
    - 若開啟 raw_score_mode，區域分數直接取單點最大值 np.max(region_pixels)。
    """
    h, w = refined_amap.shape[:2]
    work_map = refined_amap.copy()

    img_median = float(np.median(work_map))
    max_score = float(np.max(work_map))
    if max_score < img_threshold:
        return [], img_threshold

    core_thresh = img_median + core_ratio * (max_score - img_median)
    pixel_box_thresh = max(float(img_threshold * 0.92), float(core_thresh))

    binary_mask = (work_map >= pixel_box_thresh).astype(np.uint8) * 255
    if not raw_score_mode:
        k_size = max(3, int(round(max(h, w) / 500)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 1 if raw_score_mode else max(16, int(round((h * w) * min_area_ratio)))
    vis_scale = max(1.0, max(h, w) / 900.0)
    pad = max(3, int(round(4 * vis_scale)))

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
    """
    計算最終異常判定門檻，支援 5 種模式：
    - fixed      : 直接使用 user_threshold。
    - mean_std   : 直接使用非 Golden 正常圖之 mean + sigma_k * std 作為門檻 (不受 min_threshold 強制墊高影響)。
    - percentile : 自動對齊盲池 P95/P98 分位數，並以 min_threshold 保底。
    - top_k      : 自動對齊盲池第 K 高分 (最多輸出 Top-K 張)，並以 min_threshold 保底。
    - hybrid     : 結合 P95 分位數 + 抗極端值之 MAD 穩健上界 + 去極端值 Trimmed Mean+k*Std 上界 + min_threshold 保底。
    """
    mode = thresh_mode.lower().strip()
    scores_arr = np.array(non_golden_scores if non_golden_scores else golden_scores, dtype=np.float32)

    if scores_arr.size == 0:
        return max(user_threshold, min_threshold), "無樣本保底門檻"

    mean_val = float(np.mean(scores_arr))
    std_val = float(np.std(scores_arr))
    mean_std_bound = mean_val + sigma_k * std_val

    # 計算排除最高 5% 極端值後的穩健 Trimmed Mean + k*Std (兼顧抗 Escape 污染與抗反光右偏尾端)
    if scores_arr.size >= 10:
        p95_cut = float(np.percentile(scores_arr, 95.0))
        trimmed_arr = scores_arr[scores_arr <= p95_cut]
        if trimmed_arr.size >= 5:
            trim_mean = float(np.mean(trimmed_arr))
            trim_std = float(np.std(trimmed_arr))
        else:
            trim_mean, trim_std = mean_val, std_val
    else:
        trim_mean, trim_std = mean_val, std_val
    trimmed_stat_bound = trim_mean + sigma_k * max(trim_std, 0.012)

    med = float(np.median(scores_arr))
    q75, q25 = np.percentile(scores_arr, [75.0, 25.0])
    iqr_sigma = float((q75 - q25) / 1.349)
    mad_sigma = float(1.4826 * np.median(np.abs(scores_arr - med)))
    robust_sigma = max(iqr_sigma, mad_sigma, trim_std, 0.012)

    p_val = float(np.percentile(scores_arr, np.clip(auto_percentile, 50.0, 99.9)))
    mad_bound = max(med + sigma_k * robust_sigma, trimmed_stat_bound)

    topk_bound = 0.0
    if top_k > 0 and scores_arr.size > top_k:
        sorted_desc = np.sort(scores_arr)[::-1]
        topk_bound = float(0.5 * (sorted_desc[top_k - 1] + sorted_desc[top_k]))

    if mode == "fixed":
        eff = float(user_threshold)
        desc = f"Fixed 固定門檻 = {eff:.4f}"
    elif mode == "mean_std":
        eff = max(mean_std_bound, topk_bound) if top_k > 0 else mean_std_bound
        desc = (
            f"Mean+Std 統計門檻 (mean+{sigma_k:g}*std) = {eff:.4f} "
            f"[Mean={mean_val:.4f}, Std={std_val:.4f} | "
            f"mean+1*std={mean_val + 1.0 * std_val:.4f}, "
            f"mean+2*std={mean_val + 2.0 * std_val:.4f}, "
            f"mean+3*std={mean_val + 3.0 * std_val:.4f}"
            f"{f', TopK({top_k})={topk_bound:.4f}' if top_k > 0 else ''}]"
        )
    elif mode == "percentile":
        eff = max(float(min_threshold), p_val, topk_bound)
        desc = (
            f"Auto-P{auto_percentile:g} 動態分位數門檻 = {eff:.4f} "
            f"(P{auto_percentile:g}={p_val:.4f}, MinFloor={min_threshold:.4f})"
        )
    elif mode == "top_k":
        eff = max(float(min_threshold), topk_bound if top_k > 0 else p_val)
        desc = f"Auto-TopK (K={top_k}) 動態門檻 = {eff:.4f} (TopK_Bound={topk_bound:.4f}, MinFloor={min_threshold:.4f})"
    else:
        eff = max(float(min_threshold), float(user_threshold), p_val, mad_bound, topk_bound)
        desc = (
            f"Hybrid 智慧動態門檻 = {eff:.4f} "
            f"[P{auto_percentile:g}={p_val:.4f}, 穩健統計上界(med+{sigma_k:g}σ)={mad_bound:.4f}, "
            f"MinFloor={max(float(min_threshold), float(user_threshold)):.4f}"
            f"{f', TopK({top_k})={topk_bound:.4f}' if top_k > 0 else ''}]"
        )

    return eff, desc


def evaluate_overfitting_diagnostics(
    golden_scores: List[float],
    non_golden_normal_scores: List[float],
) -> Tuple[float, float, float, str, str]:
    """
    無監督過擬合自檢指標 (Unsupervised Overfitting Self-Diagnostic)：
    計算「非 Golden 正常圖中位數分數 (Median)」與「Golden 圖平均分數 (Mean)」的比值與差距：
    - Overfit_Ratio = Non_Golden_Normal_Median / Golden_Mean
    - Golden_Gap    = Non_Golden_Normal_Median - Golden_Mean
    回傳: (golden_mean, overfit_ratio, golden_gap, status_badge, advice_msg)
    """
    if not golden_scores or not non_golden_normal_scores:
        return 0.0, 1.0, 0.0, "🟢 正常 (無對照組)", ""

    g_mean = float(np.mean(golden_scores))
    ng_med = float(np.median(non_golden_normal_scores))
    overfit_ratio = ng_med / max(g_mean, 1e-6)
    golden_gap = ng_med - g_mean

    if overfit_ratio >= 1.45 and golden_gap >= 0.040:
        status_badge = "🔴 警告: 偵測到明顯過擬合 (Overfit Warning)"
        advice_msg = (
            f"  [⚠️ 過擬合自檢建議] 非 Golden 正常晶圓中位數 ({ng_med:.4f}) 顯著高於 Golden 訓練圖平均 ({g_mean:.4f}) "
            f"[Overfit_Ratio={overfit_ratio:.2f}x, Gap=+{golden_gap:.4f}]！\n"
            f"                     代表模型可能過度記憶 Golden 圖塊微結構。若發現正常晶圓誤報偏多，\n"
            f"                     可直接調高門檻倍數 (例如設定 --skip_train --thresh_mode mean_std --sigma_k 3.5) 或調降 --ckpt_step。"
        )
    elif overfit_ratio >= 1.25 and golden_gap >= 0.025:
        status_badge = "🟡 注意: 輕微過擬合傾向 (Mild Overfit — 已由動態門檻自動補償)"
        advice_msg = ""
    else:
        status_badge = "🟢 良好: 無過擬合現象 (Healthy Generalization — Golden與盲池正常分佈一致)"
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
    log_files: List[str],
) -> Tuple[int, int, float, str, str, str]:
    """
    三階段串流縫合與篩選（支援 --raw_score_mode 未濾波原生計分與四合一純化計分雙軌對照）：
    - Pass 1: 縫合 4 張 Golden 圖建立空間結構容差基準 `golden_spatial_ref`，算完立即釋放 Golden 陣列。
    - Pass 2: 逐張串流縫合，同時計算 Raw_Max_Score (未濾波) 與 Filtered_Max_Score (四合一純化後)，
              並依 raw_score_mode 決定主用熱力圖與分數，將 `float16` 暫存於磁碟 `_refined_cache_tmp`。
    - Pass 3: 僅針對超過有效門檻的異常圖片按需載入陣列畫紅框；正常圖片直接清理暫存檔略過。
    """
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
    total_imgs = len(manifest["images"])
    score_mode_label = (
        "RAW_UNFILTERED (未經四合一濾波 + 單點最大值 np.max)"
        if raw_score_mode
        else "FILTERED_ROBUST (四合一防誤報純化 + Top-0.015% 抗噪點峰值)"
    )

    log_and_print(
        f"\n[Step 4.1] 啟動低記憶體串流縫合模式 (共 {total_imgs} 張原圖) | 計分模式: {score_mode_label}",
        log_files,
    )

    # -----------------------------------------------------------------
    # Pass 1: 僅先縫合 Golden 圖 (通常僅 4 張) 計算跨片空間結構容差基準
    # -----------------------------------------------------------------
    golden_maps: List[np.ndarray] = []
    if not raw_score_mode:
        for fname, meta in manifest["images"].items():
            if fname in golden_set or os.path.basename(fname) in golden_set:
                g_amap = stitch_single_image_amap(meta, tile_amap_dir, tile_overlap)
                golden_maps.append(g_amap)

    golden_spatial_ref: Optional[np.ndarray] = None
    if golden_maps:
        ref_h, ref_w = golden_maps[0].shape[:2]
        aligned_g = [
            cv2.resize(gm, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR) if gm.shape[:2] != (ref_h, ref_w) else gm
            for gm in golden_maps
        ]
        g_max_map = np.max(np.stack(aligned_g, axis=0), axis=0)
        blur_k = max(15, int(round(max(ref_h, ref_w) * 0.035)) | 1)
        golden_spatial_ref = cv2.GaussianBlur(g_max_map, (blur_k, blur_k), 0)
        del aligned_g, g_max_map
    del golden_maps
    gc.collect()

    # -----------------------------------------------------------------
    # Pass 2: 單張串流縫合，同步計算「未濾波 Raw 分數」與「四合一濾波後分數」
    # -----------------------------------------------------------------
    stitched_info = []
    golden_max_scores = []
    non_golden_all_scores = []
    non_golden_normal_scores = []

    for idx, (fname, meta) in enumerate(manifest["images"].items(), 1):
        is_golden_img = (fname in golden_set) or (os.path.basename(fname) in golden_set)
        raw_amap = stitch_single_image_amap(meta, tile_amap_dir, tile_overlap)
        raw_flat = raw_amap.reshape(-1)
        raw_single_max = float(np.max(raw_flat))
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
            golden_tol_weight=0.35,
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
            if not os.path.basename(fname).startswith("escape_"):
                non_golden_normal_scores.append(active_max_score)

        cache_npy_path = os.path.join(refined_cache_dir, f"{idx:05d}_{meta['stem']}.npy")
        try:
            np.save(cache_npy_path, active_amap.astype(np.float16))
        except Exception:
            cache_npy_path = ""
        del raw_amap, refined_amap, active_amap

        stitched_info.append({
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
        })

        if idx % 25 == 0:
            gc.collect()

    ref_normal_scores = non_golden_normal_scores if non_golden_normal_scores else non_golden_all_scores

    effective_threshold, thresh_mode_str = compute_dynamic_effective_threshold(
        non_golden_scores=ref_normal_scores,
        golden_scores=golden_max_scores,
        thresh_mode=thresh_mode,
        user_threshold=user_threshold,
        min_threshold=min_threshold,
        auto_percentile=auto_percentile,
        sigma_k=sigma_k,
        top_k=top_k,
    )

    g_mean_score, overfit_ratio, golden_gap, overfit_badge, overfit_advice = evaluate_overfitting_diagnostics(
        golden_scores=golden_max_scores,
        non_golden_normal_scores=ref_normal_scores,
    )

    log_and_print(
        f"[Step 4.2] 異常判定門檻設定: {thresh_mode_str}\n"
        f"           (計分模式={'RAW未濾波' if raw_score_mode else '四合一純化濾波'}, "
        f"圓周邊緣抑制={'OFF(Raw)' if raw_score_mode else wafer_edge_mode}, "
        f"邊界縮減={0.0 if raw_score_mode else wafer_edge_margin*100:.1f}%, "
        f"微塵開運算比例={0.0 if raw_score_mode else speckle_filter_ratio:.4f})\n"
        f"[Step 4.2b] 過擬合自檢診斷 : {overfit_badge} "
        f"[Golden平均={g_mean_score:.4f} | 非Golden正常中位數={float(np.median(ref_normal_scores)) if ref_normal_scores else 0.0:.4f} | "
        f"Overfit_Ratio={overfit_ratio:.2f}x (Gap={golden_gap:+.4f})]",
        log_files,
    )
    if overfit_advice:
        log_and_print(overfit_advice, log_files)

    # -----------------------------------------------------------------
    # Pass 3: 依分數高低排序，僅針對超過門檻的圖片載入陣列並輸出紅框
    # -----------------------------------------------------------------
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
            ranking_records.append({
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
            })
            continue

        if cache_npy_path and os.path.exists(cache_npy_path):
            active_amap = np.load(cache_npy_path).astype(np.float32)
            try:
                os.remove(cache_npy_path)
            except OSError:
                pass
        else:
            raw_amap = stitch_single_image_amap(meta, tile_amap_dir, tile_overlap)
            if raw_score_mode:
                active_amap = raw_amap
            else:
                active_amap = refine_amap_for_wafer_and_noise(
                    full_amap=raw_amap,
                    img_path=meta["path"],
                    category=category,
                    border_ratio=border_ratio,
                    wafer_edge_mode=wafer_edge_mode,
                    wafer_edge_margin=wafer_edge_margin,
                    speckle_filter_ratio=speckle_filter_ratio,
                    golden_spatial_ref=golden_spatial_ref if not is_golden else None,
                    golden_tol_weight=0.35,
                )
                del raw_amap

        bboxes, pixel_box_thresh = extract_tight_bboxes_for_anomaly_image(
            refined_amap=active_amap,
            img_threshold=effective_threshold,
            core_ratio=core_ratio,
            min_area_ratio=min_area_ratio,
            raw_score_mode=raw_score_mode,
        )

        if not bboxes:
            del active_amap
            skipped_normal_count += 1
            status_tag = "GOLDEN_PASS" if is_golden else "PASS_NOISE_FILTERED"
            ranking_records.append({
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
            })
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
                bbox_canvas, label_txt, (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
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
        mode_tag = "RAW" if raw_score_mode else "FILT"
        banner_txt = (
            f"[{mode_tag}] Orig: {fname} | Score: {max_score:.4f} (Raw:{raw_max_score:.4f}, Filt:{filtered_max_score:.4f}, "
            f"EffThresh:{effective_threshold:.4f}) | Status: ANOMALY (Boxes:{len(bboxes)})"
        )
        cv2.putText(
            banner, banner_txt, (12, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, banner_h / 55.0), (0, 220, 255), 2, cv2.LINE_AA
        )
        final_vis = np.vstack([banner, tri_panel])
        vis_out_path = os.path.join(anomaly_dir, prefixed_img_name)
        cv2.imwrite(vis_out_path, final_vis)

        del active_amap, orig_bgr, bbox_canvas, hm_color, overlay_bgr, tri_panel, final_vis
        gc.collect()

        status_tag = "GOLDEN_ANOMALY" if is_golden else "ANOMALY"
        ranking_records.append({
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
        })

    safe_rmtree(refined_cache_dir)
    del golden_spatial_ref
    gc.collect()

    # 計算非 Golden 正常圖之「濾波後 vs 未濾波 Raw」分數分佈對照
    non_golden_recs = [r for r in ranking_records if not r["status"].startswith("GOLDEN")]
    non_golden_normal_recs = [
        r for r in non_golden_recs if not os.path.basename(r["filename"]).startswith("escape_")
    ]
    target_recs = non_golden_normal_recs if non_golden_normal_recs else non_golden_recs
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
        f"      📊 非 Golden 正常圖分數分佈 ({'RAW未濾波' if raw_score_mode else '四合一濾波後'}) : "
        f"Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
        f"      📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
        f"      🔬 濾波前後正常圖對照 (Ablation): "
        f"[未濾波 Raw] Max={raw_max_s:.4f}, Mean={raw_mean_s:.4f}, Std={raw_std_s:.4f}  vs  "
        f"[四合一濾波] Max={filt_max_s:.4f}, Mean={filt_mean_s:.4f}, Std={filt_std_s:.4f}\n"
        f"      🩺 模型過擬合自檢 (Overfit)    : {overfit_badge}\n"
        f"                                       [Golden平均={g_mean_score:.4f} vs 非Golden中位數={median_s:.4f} | Overfit_Ratio={overfit_ratio:.2f}x | Gap={golden_gap:+.4f}]"
    )
    if overfit_advice:
        dist_summary_lines += f"\n{overfit_advice}"

    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== FoundAD High-Res Anomaly Detection Report ===\n")
        f.write(
            f"Score Mode: {'RAW_UNFILTERED' if raw_score_mode else 'FILTERED_ROBUST'} | "
            f"Threshold Mode: {thresh_mode} | Effective Threshold: {effective_threshold:.4f} | "
            f"Wafer Edge Mode: {'off(raw)' if raw_score_mode else wafer_edge_mode} (Margin: {wafer_edge_margin:.3f})\n"
            f"Core Box Ratio: {core_ratio:.2f} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_normal_count}\n"
            f"Non-Golden Normal Score Dist: Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | "
            f"Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
            f"Ablation Compare (Normal)   : Raw_Max={raw_max_s:.4f}, Raw_Mean={raw_mean_s:.4f} vs "
            f"Filtered_Max={filt_max_s:.4f}, Filtered_Mean={filt_mean_s:.4f}\n"
            f"Threshold Ref (Mean+k*Std)  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
            f"Overfit Diagnostic          : {overfit_badge} "
            f"(Golden_Mean={g_mean_score:.4f}, NonGolden_Med={median_s:.4f}, Ratio={overfit_ratio:.2f}x, Gap={golden_gap:+.4f})\n"
        )
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
        writer.writerow([
            "Rank", "Original_Filename", "Status", "Max_Score",
            "Raw_Max_Score", "Filtered_Max_Score", "Filter_Delta",
            "TopK_Mean_Score", "Num_BBoxes", "Saved_Anomaly_Filename", "Resolution"
        ])
        for r in ranking_records:
            writer.writerow([
                r["rank"], r["filename"], r["status"],
                f"{r['max_score']:.6f}",
                f"{r['raw_max_score']:.6f}",
                f"{r['filtered_max_score']:.6f}",
                f"{r['filter_delta']:+.6f}",
                f"{r['top_k_score']:.6f}",
                r["num_bboxes"], r["saved_file"], r["resolution"],
            ])

    log_and_print(
        f"[Step 4.3] 篩選與畫框完成：共 {anomaly_count} 張超過有效門檻 ({effective_threshold:.4f}) 已畫框輸出；"
        f"其餘 {skipped_normal_count} 張正常圖已自動略過不輸出。",
        log_files,
    )
    return anomaly_count, skipped_normal_count, effective_threshold, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines


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

    # 門檻與動態對齊參數
    parser.add_argument(
        "--thresh_mode", type=str, default="hybrid",
        choices=["hybrid", "mean_std", "percentile", "top_k", "fixed"],
        help="門檻決定模式：hybrid(結合P95+MAD穩健統計+保底門檻)、mean_std(直接以 mean + sigma_k * std 作為門檻)、percentile(對齊分位數)、top_k(只抓前K高分)、fixed(固定threshold)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="當 thresh_mode=fixed 時作為固定門檻；當 thresh_mode=hybrid 時作為保底門檻下限之一"
    )
    parser.add_argument(
        "--min_threshold", type=float, default=0.27,
        help="動態門檻的絕對最低安全底線 (適用於 hybrid / percentile / top_k 模式)"
    )
    parser.add_argument(
        "--auto_percentile", type=float, default=95.0,
        help="動態門檻自動對齊之盲池分位數 (例如 95.0 代表 P95，98.0 代表 P98)"
    )
    parser.add_argument(
        "--top_k", type=int, default=0,
        help="若設為 > 0 (例如 5 或 8)，則限制最多只輸出分數最高的前 K 張可疑圖片"
    )
    parser.add_argument(
        "--sigma_k", type=float, default=3.5,
        help="當 thresh_mode=mean_std 時代表 mean + k*std 的倍數 k (如 1.0, 2.0, 3.0)；當為 hybrid 時代表 MAD 標準差倍數"
    )

    # 晶圓圓周邊緣抑制與微塵濾波參數 + Raw 未濾波對照開關
    parser.add_argument(
        "--raw_score_mode", action="store_true",
        help="若設定此旗標，將完全關閉四合一防誤報濾波，直接以縫合後之原始熱力圖與單點最大值 np.max(amap) 進行計分與畫框 (供消融實驗對照用)"
    )
    parser.add_argument(
        "--core_ratio", type=float, default=0.55,
        help="異常熱區峰值核心收斂比例 (0.45~0.65，越高紅框越緊密貼合最高分缺陷核心)"
    )
    parser.add_argument(
        "--border_ratio", type=float, default=0.02,
        help="抑制影像四週矩形邊框 Padding 偽影的比例 (預設 0.02 即 2%%)"
    )
    parser.add_argument(
        "--wafer_edge_mode", type=str, default="auto",
        choices=["auto", "on", "off"],
        help="晶圓圓周邊緣 (Bevel/Notch) 抑制模式：auto(自動偵測圓形工件)、on(強制開啟)、off(關閉)"
    )
    parser.add_argument(
        "--wafer_edge_margin", type=float, default=0.04,
        help="晶圓圓周外圈倒角抑制寬度佔半徑比例 (預設 0.04 即外圈 4%% 半徑平滑衰減)"
    )
    parser.add_argument(
        "--speckle_filter_ratio", type=float, default=0.008,
        help="灰階形態學開運算濾除單點微塵尖峰之核大小比例 (預設 0.008，設 0 則關閉)"
    )
    parser.add_argument(
        "--min_area_ratio", type=float, default=0.00012,
        help="有效異常紅框的最小面積佔全圖比例 (低於此面積之微小雜訊點不畫框、不誤報)"
    )

    parser.add_argument("--skip_train", action="store_true", help="跳過訓練，並於快取命中時自動跳過切塊與 GPU 推論，3秒極速重調門檻")
    parser.add_argument("--force_infer", action="store_true", help="即使設定 --skip_train 仍強制重新切塊並重跑 GPU 推論")
    parser.add_argument("--mock_from_mvtec", type=str, default="")

    parser.add_argument("--tile_size", type=int, default=512, help="子圖塊像素大小 (建議 512 或 1024，避免大圖壓縮失真)")
    parser.add_argument("--tile_overlap", type=int, default=64, help="相鄰子圖塊重疊像素寬度 (用於無縫平滑拼接)")
    parser.add_argument("--max_train_tiles", type=int, default=48, help="從 Golden 大圖抽出的代表性高解析度訓練圖塊上限")
    parser.add_argument("--train_batch_size", type=int, default=8, help="訓練時的最大 Batch Size，防止高解析度圖塊 OOM")
    parser.add_argument("--infer_batch_size", type=int, default=16, help="推論子圖塊時的 Mini-Batch 大小，大幅加速 5000x5000 推論")
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
        existing_pool_imgs = scan_pool_images(args.pool_dir) if os.path.isdir(args.pool_dir) else []
        if args.skip_train and existing_pool_imgs and not args.force_infer:
            log_and_print(
                f"[⚡ 模擬無GT盲池快取命中] 已設定 --skip_train，直接沿用既有 {len(existing_pool_imgs)} 張盲池圖片 "
                f"-> {os.path.abspath(args.pool_dir)}",
                log_files,
            )
        else:
            prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, log_files, defect_limit=8)

    pool_signature = compute_pool_signature(args.pool_dir)
    tile_config_key = (
        f"t{args.tile_size}_ov{args.tile_overlap}_bg{args.bg_std_thresh:g}_dt{int(args.disable_tiling)}"
    )
    golden_meta_path = os.path.join(shot_dir, "golden_cache_meta.json")

    # =========================================================================
    # [階段一：整體訓練階段] (包含 Golden 自動挑圖 + 高解析度切塊 + 模型訓練)
    # =========================================================================
    used_golden_cache = False
    golden_filenames: List[str] = []
    num_train_samples = args.shots
    golden_select_sec = 0.0
    train_tile_sec = 0.0

    if args.skip_train and not args.force_infer and os.path.isfile(golden_meta_path):
        try:
            with open(golden_meta_path, "r", encoding="utf-8") as f:
                g_meta = json.load(f)
            if (
                g_meta.get("pool_signature") == pool_signature
                and g_meta.get("category") == args.category
                and int(g_meta.get("shots", -1)) == int(args.shots)
                and int(g_meta.get("tile_size", -1)) == int(args.tile_size)
                and int(g_meta.get("max_train_tiles", -1)) == int(args.max_train_tiles)
                and g_meta.get("golden_filenames")
            ):
                golden_filenames = list(g_meta["golden_filenames"])
                num_train_samples = int(g_meta.get("num_train_samples", args.shots))
                rec_src = os.path.join(shot_dir, f"selected_{args.shots}_golden_images.txt")
                if os.path.exists(rec_src):
                    safe_copy(rec_src, os.path.join(run_out_dir, f"selected_{args.shots}_golden_images.txt"))
                used_golden_cache = True
                log_and_print(
                    f"[⚡ Golden 快取命中] 已設定 --skip_train，直接沿用已挑選之 {len(golden_filenames)} 張 Golden Image "
                    f"({', '.join(golden_filenames)}) (0.00 秒)",
                    log_files,
                )
        except Exception:
            used_golden_cache = False

    if not used_golden_cache:
        safe_rmtree(shot_dir)
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
        golden_filenames = [
            os.path.relpath(p, args.pool_dir).replace("\\", "/") for p in golden_paths
        ]
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

        try:
            with open(golden_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pool_signature": pool_signature,
                        "category": args.category,
                        "shots": args.shots,
                        "tile_size": args.tile_size,
                        "max_train_tiles": args.max_train_tiles,
                        "golden_filenames": golden_filenames,
                        "num_train_samples": num_train_samples,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

    effective_batch_size = min(max(1, num_train_samples), args.train_batch_size)
    iters_per_epoch = max(1, int(np.ceil(num_train_samples / float(effective_batch_size))))
    # 動態折算訓練 Epochs：當切塊數較多 (iters_per_epoch > 1) 時，防止總梯度更新步數暴增 6 倍導致嚴重過擬合與耗時過長
    effective_epochs = (
        max(400, int(np.ceil(args.ckpt_step / float(iters_per_epoch))))
        if iters_per_epoch > 1 else args.ckpt_step
    )
    total_grad_updates = effective_epochs * iters_per_epoch

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
            f"\n[Step 2] 執行 FoundAD 高解析度圖塊訓練 "
            f"(train_tiles={num_train_samples}, batch_size={effective_batch_size}, "
            f"iters_per_epoch={iters_per_epoch}, epochs={effective_epochs}, total_updates={total_grad_updates}):",
            log_files,
        )
        run_cmd_and_log(train_cmd, log_files)
        model_train_sec = time.perf_counter() - t_model_train_start
    else:
        log_and_print("[⏱️ 計時] 已設定 --skip_train，跳過 Step 2 模型訓練階段 (0.00 秒)", log_files)

    total_train_time = golden_select_sec + train_tile_sec + model_train_sec
    actual_step, ckpt_dirs, ckpt_files = prepare_ckpt_for_demo(
        args.data_name, args.diy_name, effective_epochs if not args.skip_train else args.ckpt_step, log_files
    )
    ckpt_signature = compute_ckpt_signature(ckpt_files)

    train_tiles_dir = os.path.abspath(os.path.join(shot_dir, "train", args.category))
    golden_backup_dir = os.path.abspath(os.path.join(shot_dir, "golden_full_res_backup", args.category))

    train_done_lines = [
        f"\n{'='*75}",
        f" [✅ 訓練階段完成 — 時間統計與權重檔儲存位置]",
        f"{'-'*75}",
        f"  ⏱️ Step 1 自動挑選 Golden 圖耗時 : {format_duration(golden_select_sec)}{' (Cached)' if used_golden_cache else ''}",
        f"  ⏱️ Step 1.5 高解析度訓練切塊耗時 : {format_duration(train_tile_sec)}{' (Cached)' if used_golden_cache else ''}",
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
    # [階段二：整體推論階段] (支援子圖塊熱力圖持久化快取，--skip_train 時 3 秒秒速完成)
    # =========================================================================
    amap_dump_dir = os.path.join(run_out_dir, "amap_npy")
    report_txt_path = os.path.join(run_out_dir, "escape_ranking_report.txt")
    report_csv_path = os.path.join(run_out_dir, "escape_ranking_report.csv")

    tiled_pool_dir = os.path.join(abs_datasets_dir, f"{args.data_name}_tiled_pool")
    tile_cache_root = os.path.join(abs_datasets_dir, f"{args.data_name}_tile_amap_cache")
    cached_manifest_path = os.path.join(tile_cache_root, "cached_tile_manifest.json")
    tile_amap_cache_dir = os.path.join(
        tile_cache_root, f"{args.diy_name}_step{actual_step}_{tile_config_key}"
    )

    test_tile_sec = 0.0
    model_infer_sec = 0.0
    used_amap_cache = False
    used_png_tile_cache = False

    # 檢查是否可直接命中 Step 3.1 + Step 3.2 的子圖塊 .npy 推論結果快取
    manifest = None
    if args.skip_train and not args.force_infer:
        manifest = is_tile_amap_cache_valid(
            tile_amap_cache_dir=tile_amap_cache_dir,
            cached_manifest_path=cached_manifest_path,
            pool_signature=pool_signature,
            tile_config_key=tile_config_key,
            ckpt_signature=ckpt_signature,
            actual_step=actual_step,
            diy_name=args.diy_name,
        )

    if manifest is not None:
        used_amap_cache = True
        used_png_tile_cache = True
        safe_copy(cached_manifest_path, os.path.join(run_out_dir, "tile_manifest.json"))
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)
        log_and_print(
            f"\n[⚡ 子圖塊熱力圖快取命中] 偵測到完整推論快取 ({total_test_tiles} 塊 .npy, step={actual_step})！\n"
            f"   -> 自動跳過 Step 3.1 測試池切塊與 Step 3.2 GPU 推論 (0.00 秒)，直接進入 Step 4 極速重調門檻！\n"
            f"   -> 快取目錄: {os.path.abspath(tile_amap_cache_dir)}",
            log_files,
        )
    else:
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
                f"\n[⚡ 測試池切塊快取命中] 測試圖片池與切塊參數未變動，直接沿用既有 {total_test_tiles} 張子圖塊 PNG (0.00 秒)",
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
                bg_std_thresh=args.bg_std_thresh,
                disable_tiling=args.disable_tiling,
                pool_signature=pool_signature,
                tile_config_key=tile_config_key,
                cached_manifest_path=cached_manifest_path,
                log_files=log_files,
            )
            test_tile_sec = time.perf_counter() - t_test_tile_start
            num_test_images = manifest.get("num_images", 1)
            total_test_tiles = manifest.get("total_tiles", num_test_images)

        safe_rmtree(tile_amap_cache_dir)
        os.makedirs(tile_amap_cache_dir, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FAB_AMAP_DUMP_DIR"] = os.path.abspath(tile_amap_cache_dir)
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
            f"\n[Step 3] 執行 FoundAD 子圖塊高解析度批次推論 "
            f"(BatchSize={args.infer_batch_size}, 關閉冗餘子圖塊 PNG 輸出 -> {os.path.abspath(tile_amap_cache_dir)}):",
            log_files,
        )
        run_cmd_and_log(demo_cmd, log_files, env=env)
        model_infer_sec = time.perf_counter() - t_infer_start

        for old_viz in glob.glob(f"./logs/{args.data_name}/**/no_gt_results", recursive=True):
            safe_rmtree(old_viz)
        for old_hm in glob.glob(f"./logs/{args.data_name}/**/demo/heatmaps", recursive=True):
            safe_rmtree(old_hm)

        try:
            with open(os.path.join(tile_amap_cache_dir, "cache_info.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pool_signature": pool_signature,
                        "tile_config_key": tile_config_key,
                        "ckpt_signature": ckpt_signature,
                        "actual_step": actual_step,
                        "diy_name": args.diy_name,
                        "created_at": timestamp,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

    t_stitch_start = time.perf_counter()
    anomaly_count, skipped_count, eff_thresh, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines = stitch_and_filter_anomalies(
        manifest=manifest,
        tile_amap_dir=tile_amap_cache_dir,
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
        log_files=log_files,
    )
    stitch_vis_sec = time.perf_counter() - t_stitch_start

    total_infer_time = test_tile_sec + model_infer_sec + stitch_vis_sec
    avg_infer_per_img = total_infer_time / max(1, num_test_images)
    total_wall_time = time.perf_counter() - pipeline_start_t

    infer_done_summary = (
        f"\n{'='*75}\n"
        f" [✅ 推論階段完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-'*75}\n"
        f"  [1] 整體訓練階段總時間 (Total Training Time)  : {format_duration(total_train_time)}\n"
        f"      ├─ Step 1 自動挑選 Golden Image            : {format_duration(golden_select_sec)}"
        f"{' (Cached)' if used_golden_cache else ''}\n"
        f"      ├─ Step 1.5 Golden 高解析度切塊            : {format_duration(train_tile_sec)}"
        f"{' (Cached)' if used_golden_cache else ''}\n"
        f"      └─ Step 2 FoundAD 模型訓練 (step={actual_step})     : {format_duration(model_train_sec)}"
        f"{' (Skipped)' if args.skip_train else ''}\n"
        f"{'-'*75}\n"
        f"  [2] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ Step 3.1 測試池高解析度切塊 ({total_test_tiles} 塊)   : {format_duration(test_tile_sec)}"
        f"{' (Cached)' if used_png_tile_cache else ''}\n"
        f"      ├─ Step 3.2 FoundAD 模型推論               : {format_duration(model_infer_sec)}"
        f"{' (Cached - 0秒極速載入)' if used_amap_cache else ''}\n"
        f"      ├─ Step 4 全解析度串流縫合與{'未濾波' if args.raw_score_mode else '四合一濾波'}畫框 : {format_duration(stitch_vis_sec)}\n"
        f"      └─ 平均每張原圖推論耗時 ({num_test_images} 張原圖)         : {avg_infer_per_img:.2f} 秒/張\n"
        f"{'-'*75}\n"
        f"  [3] 任務全程總執行時間 (Total Wall Time)      : {format_duration(total_wall_time)}\n"
        f"{'-'*75}\n"
        f"  [4] 門檻篩選統計、分數分佈與過擬合自檢 (Score={'RAW' if args.raw_score_mode else 'FILTERED'}, Mode={args.thresh_mode}, Eff_Thresh={eff_thresh:.4f}):\n"
        f"      🚨 超過門檻判定異常並畫框輸出 : {anomaly_count} 張\n"
        f"      ✅ 未達門檻或微塵濾除直接略過 : {skipped_count} 張 (不輸出圖片)\n"
        f"{dist_summary_lines}\n"
        f"{'-'*75}\n"
        f"  [5] 權重檔與本次推論結果儲存資料夾清單 (Saved Directories & Files):\n"
        f"      📦 訓練權重檔資料夾           : {ckpt_dirs[0]}\n"
        f"      ⚡ 子圖塊熱力圖快取資料夾     : {os.path.abspath(tile_amap_cache_dir)}\n"
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