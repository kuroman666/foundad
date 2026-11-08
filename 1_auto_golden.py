# 檔案位置: FoundAD/1_auto_golden.py (全選覆蓋此檔；亦可直接複製覆蓋至 fab_tools/auto_golden.py)
import os
import re
import gc
import stat
import glob
import json
import time
import shutil
import hashlib
import argparse
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def format_duration(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS.xx 與總秒數。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f} ({seconds:.2f} 秒)"


def log_and_print(msg: str, log_files: Optional[List[str]] = None):
    """同時在 Terminal 顯示並附加寫入 log.txt 檔案。"""
    print(msg, flush=True)
    if log_files:
        for lf in log_files:
            if lf:
                with open(lf, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")


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


def scan_pool_images(pool_dir: str) -> List[str]:
    """遞迴搜尋資料夾與所有子資料夾內的圖片，回傳排序後的完整路徑清單。"""
    if not os.path.isdir(pool_dir):
        return []
    img_paths = []
    for root, dirs, files in os.walk(pool_dir):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTENSIONS) and not f.startswith("."):
                img_paths.append(os.path.join(root, f))
    return sorted(img_paths)


def compute_pool_signature(pool_dir: str) -> str:
    """計算測試盲池所有有效影像之 (相對路徑, 檔案大小) MD5 指紋（不含絕對路徑，避免目錄搬移失效）。"""
    if not os.path.isdir(pool_dir):
        return ""
    img_paths = scan_pool_images(pool_dir)
    h = hashlib.md5()
    for fpath in img_paths:
        rel_path = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        try:
            sz = os.path.getsize(fpath)
        except OSError:
            sz = 0
        h.update(f"{rel_path}:{sz};".encode("utf-8"))
    return h.hexdigest()


def _robust_normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """使用 P0.1 ~ P99.9 穩健百分位數將 16-bit / float 陣列轉為 uint8。"""
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
    """安全讀取影像並統一轉為 (H, W, 3) uint8 RGB 格式。"""
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


def load_pil_rgb_uint8(img_path: str) -> Image.Image:
    """安全讀取 8-bit / 16-bit TIFF 或單通道灰階圖，統一正規化為 8-bit RGB PIL Image。"""
    return Image.fromarray(load_image_rgb_uint8(img_path), mode="RGB")


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


# =====================================================================
# 48 個代表性訓練圖塊與同心圓環帶視覺化輸出模組 (供報告與簡報使用)
# =====================================================================
def _draw_dashed_circle(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_deg: int = 6,
):
    """在 OpenCV 畫布上繪製虛線同心圓，用於標示徑向分層環帶 (Radial Bins)。"""
    if radius <= 4:
        return
    for angle in range(0, 360, dash_deg * 2):
        cv2.ellipse(canvas, center, (radius, radius), 0, angle, angle + dash_deg, color, thickness, cv2.LINE_AA)


def save_golden_tile_layout_visualization(
    base_img_rgb: np.ndarray,
    all_valid_coords: List[Tuple[int, int]],
    selected_tile_records: List[Tuple[int, int, str, float]],
    tile_size: int,
    tile_overlap: int,
    num_cands: int,
    out_vis_path: str,
    report_dir: Optional[str] = None,
    log_files: Optional[List[str]] = None,
):
    """
    將挑出的 48 個代表性訓練圖塊座標、候選切塊網格與同心圓環帶繪製於晶圓縮圖上並輸出高解析度 PNG：
    - 左圖：顯示所有非背景有效切塊網格 (灰色細框) + 4 道同心圓徑向分層環帶 (青色虛線圓)。
    - 右圖：顯示最終挑出的 48 個訓練圖塊 (依圓心->外環 #01~#48 以彩虹漸層上色，並標註編號與來源)。
    """
    h0, w0 = base_img_rgb.shape[:2]
    panel_h = 840
    scale = panel_h / float(h0)
    panel_w = max(320, int(round(w0 * scale)))

    base_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
    resized_bgr = cv2.resize(base_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    # 稍微壓暗底圖 25%，讓彩色切塊框、同心圓與編號文字清晰突顯
    dark_bg = (resized_bgr.astype(np.float32) * 0.72).astype(np.uint8)
    left_panel = dark_bg.copy()
    right_panel = dark_bg.copy()
    right_fill_layer = dark_bg.copy()

    cy_s, cx_s = int(round(h0 * 0.5 * scale)), int(round(w0 * 0.5 * scale))
    ts_y = max(4, int(round(tile_size * scale)))
    ts_x = max(4, int(round(tile_size * scale)))

    # 1. 左面板：繪製所有有效非背景候選切塊網格 + 同心圓半徑環帶
    max_radius = 0.0
    for (y, x) in all_valid_coords:
        y1, x1 = int(round(y * scale)), int(round(x * scale))
        y2, x2 = min(panel_h - 1, y1 + ts_y), min(panel_w - 1, x1 + ts_x)
        cv2.rectangle(left_panel, (x1, y1), (x2, y2), (115, 115, 115), 1, cv2.LINE_AA)
        r = float(np.hypot((y + tile_size * 0.5) - h0 * 0.5, (x + tile_size * 0.5) - w0 * 0.5))
        if r > max_radius:
            max_radius = r

    if max_radius < 1.0:
        max_radius = min(h0, w0) * 0.45

    # 繪製 25%, 50%, 75%, 100% 四道同心圓虛線環帶與圓心十字
    for frac, label_str in [(0.25, "R25% (Center)"), (0.50, "R50% (Mid-Inner)"), (0.75, "R75% (Mid-Outer)"), (1.00, "R100% (Edge)")]:
        r_px = int(round(max_radius * frac * scale))
        _draw_dashed_circle(left_panel, (cx_s, cy_s), r_px, (255, 230, 0), thickness=1, dash_deg=5)
        ly = max(18, cy_s - r_px - 4)
        cv2.putText(
            left_panel,
            label_str,
            (max(8, cx_s - 45), ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 240, 120),
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(left_panel, (cx_s, cy_s), (0, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # 2. 右面板：繪製最終挑出的 48 個代表性訓練圖塊 (依半徑 #01 -> #48 彩虹漸層上色)
    n_sel = max(1, len(selected_tile_records))
    for idx, (y, x, g_tag, _) in enumerate(selected_tile_records):
        # 計算 HSV 彩虹色 (Hue: 115 藍青色 [圓心] -> 60 綠色 [中環] -> 0 紅色 [外環])
        ratio = idx / float(max(1, n_sel - 1))
        hue = int(round(115 * (1.0 - ratio)))
        hsv_pixel = np.uint8([[[hue, 235, 255]]])
        bgr_color = tuple(int(c) for c in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])

        y1, x1 = int(round(y * scale)), int(round(x * scale))
        y2, x2 = min(panel_h - 1, y1 + ts_y), min(panel_w - 1, x1 + ts_x)
        cv2.rectangle(right_fill_layer, (x1, y1), (x2, y2), bgr_color, -1)

    # 半透明色塊疊加，讓重疊區 (64px overlap) 自然呈現交集加深效果
    cv2.addWeighted(right_fill_layer, 0.22, right_panel, 0.78, 0, right_panel)

    # 同步在右面板畫上淡虛線同心圓作為參考，再畫出 48 個圖塊的銳利邊框與編號標籤
    for frac in (0.25, 0.50, 0.75, 1.00):
        r_px = int(round(max_radius * frac * scale))
        _draw_dashed_circle(right_panel, (cx_s, cy_s), r_px, (180, 180, 180), thickness=1, dash_deg=4)

    for idx, (y, x, g_tag, _) in enumerate(selected_tile_records):
        ratio = idx / float(max(1, n_sel - 1))
        hue = int(round(115 * (1.0 - ratio)))
        hsv_pixel = np.uint8([[[hue, 235, 255]]])
        bgr_color = tuple(int(c) for c in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])

        y1, x1 = int(round(y * scale)), int(round(x * scale))
        y2, x2 = min(panel_h - 1, y1 + ts_y), min(panel_w - 1, x1 + ts_x)
        cv2.rectangle(right_panel, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

        # 在圖塊中心繪製黑底白字編號標籤 (例如 #01 g0)
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2
        label = f"#{idx + 1:02d}"
        (tw, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bx1 = max(0, cx_box - tw // 2 - 3)
        by1 = max(0, cy_box - th_txt // 2 - 3)
        bx2 = min(panel_w - 1, cx_box + tw // 2 + 3)
        by2 = min(panel_h - 1, cy_box + th_txt // 2 + 3)
        cv2.rectangle(right_panel, (bx1, by1), (bx2, by2), (18, 18, 18), -1)
        cv2.rectangle(right_panel, (bx1, by1), (bx2, by2), bgr_color, 1)
        cv2.putText(
            right_panel,
            label,
            (bx1 + 3, by2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. 組合左右雙面板與頂部/底部圖例資訊列
    sep = np.full((panel_h, 8, 3), 40, dtype=np.uint8)
    body = np.hstack([left_panel, sep, right_panel])
    total_w = body.shape[1]

    header_h = 82
    footer_h = 46
    header = np.full((header_h, total_w, 3), 24, dtype=np.uint8)
    footer = np.full((footer_h, total_w, 3), 30, dtype=np.uint8)

    title_main = (
        f"Stage 2 Radial Stratified Golden Tile Selection | "
        f"Orig: {w0}x{h0} | Tile: {tile_size}x{tile_size} (Overlap={tile_overlap}px, Stride={max(1, tile_size - tile_overlap)}px)"
    )
    sub_left = f"[Left] All {len(all_valid_coords)} Non-Background Grid Coords & Concentric Radial Rings (Pool={num_cands} Wafers)"
    sub_right = f"[Right] Final {len(selected_tile_records)} Purified Training Tiles (#01 Center [Blue] -> #{len(selected_tile_records):02d} Edge [Red])"

    cv2.putText(header, title_main, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(header, sub_left, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        sub_right,
        (panel_w + 20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 255, 160),
        1,
        cv2.LINE_AA,
    )

    footer_txt = (
        "Legend:  #01~#12 Center Die (Blue/Cyan)   |   #13~#32 Mid-Ring (Green/Yellow)   |   "
        "#33~#48 Wafer Edge/Bevel (Orange/Red)   |   Semi-transparent shading shows 64px tile overlap"
    )
    cv2.putText(footer, footer_txt, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

    final_canvas = np.vstack([header, body, footer])
    os.makedirs(os.path.dirname(os.path.abspath(out_vis_path)), exist_ok=True)
    cv2.imwrite(out_vis_path, final_canvas)

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        safe_copy(out_vis_path, os.path.join(report_dir, os.path.basename(out_vis_path)))

    log_and_print(
        f"[🎨 切塊空間分佈視覺化已輸出] 已將 {len(selected_tile_records)} 個代表性訓練圖塊與同心圓環帶繪製於 -> {os.path.abspath(out_vis_path)}",
        log_files,
    )


def _ensure_layout_vis_from_existing_dataset(
    shot_dir: str,
    category: str,
    tile_size: int,
    tile_overlap: int,
    bg_std_thresh: float,
    report_dir: Optional[str] = None,
    log_files: Optional[List[str]] = None,
):
    """
    若觸發防呆直接沿用既有 Golden 資料集，但目錄下尚無 golden_tile_layout_vis.png，
    自動從 backup 原圖與 train/<category> 檔名解析座標並補畫視覺化分佈圖 (不重挑 Golden)。
    """
    vis_path = os.path.join(shot_dir, "golden_tile_layout_vis.png")
    if os.path.isfile(vis_path):
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
            safe_copy(vis_path, os.path.join(report_dir, "golden_tile_layout_vis.png"))
        return

    backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    cat_train_dir = os.path.join(shot_dir, "train", category)
    if not os.path.isdir(cat_train_dir):
        return

    base_candidates = []
    if os.path.isdir(backup_dir):
        base_candidates = sorted(
            [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        )
    if not base_candidates:
        base_candidates = sorted(
            [os.path.join(cat_train_dir, f) for f in os.listdir(cat_train_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        )
    if not base_candidates:
        return

    try:
        base_rgb = load_image_rgb_uint8(base_candidates[0])
        h0, w0 = base_rgb.shape[:2]
        train_files = sorted([f for f in os.listdir(cat_train_dir) if f.lower().endswith(IMG_EXTENSIONS)])

        selected_records: List[Tuple[int, int, str, float]] = []
        for f in train_files:
            m = re.search(r"_([gc]\d+)_tile_y(\d+)_x(\d+)", f)
            if m:
                selected_records.append((int(m.group(2)), int(m.group(3)), m.group(1), 0.0))

        if not selected_records:
            # 整張原圖模式 (未切塊)
            selected_records = [(0, 0, "full", 0.0)]
            valid_coords = [(0, 0)]
            eff_ts = max(h0, w0)
        else:
            eff_ts = tile_size
            stride = max(1, tile_size - tile_overlap)
            ys = get_axis_positions(h0, tile_size, stride)
            xs = get_axis_positions(w0, tile_size, stride)
            valid_coords = []
            for y in ys:
                for x in xs:
                    crop = base_rgb[y:min(h0, y + tile_size), x:min(w0, x + tile_size)]
                    if not is_background_tile(crop, bg_std_thresh):
                        valid_coords.append((y, x))
            if not valid_coords:
                valid_coords = [(r[0], r[1]) for r in selected_records]

        save_golden_tile_layout_visualization(
            base_img_rgb=base_rgb,
            all_valid_coords=valid_coords,
            selected_tile_records=selected_records,
            tile_size=eff_ts,
            tile_overlap=tile_overlap,
            num_cands=len(base_candidates),
            out_vis_path=vis_path,
            report_dir=report_dir,
            log_files=log_files,
        )
    except Exception:
        pass


def prepare_mock_unlabeled_pool_from_mvtec(
    mvtec_cat_dir: str,
    mock_pool_dir: str,
    log_files: Optional[List[str]] = None,
    defect_limit: int = 8,
    rebuild: bool = False,
):
    """從 MVTec 建立模擬無 GT 盲池（若已存在且未指定 rebuild 則直接沿用）。"""
    if not rebuild and os.path.isdir(mock_pool_dir):
        existing = scan_pool_images(mock_pool_dir)
        if existing:
            log_and_print(
                f"[⚡ 模擬無GT盲池已存在] 直接沿用既有 {len(existing)} 張圖片 -> {os.path.abspath(mock_pool_dir)}",
                log_files,
            )
            return

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
        f"[模擬無GT盲池建立完成] 共 {total_count} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵) "
        f"-> {os.path.abspath(mock_pool_dir)}",
        log_files,
    )


def resolve_dino_ckpt_path(ckpt_path: str) -> str:
    """若指定的 ckpt_path 不存在，依序優先搜尋 dinov3 -> dinov2 -> dino 權重檔。"""
    if ckpt_path and ckpt_path.lower() != "auto" and os.path.exists(ckpt_path):
        return ckpt_path

    search_dirs = [
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        os.path.expanduser("~/.cache/torch/hub"),
        "./hub_cache",
        "./",
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            for ext in ("*.pth", "*.pt", "*.bin"):
                candidates.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    for keyword in ("dinov3", "dinov2", "dino"):
        matched = [p for p in candidates if keyword in os.path.basename(p).lower()]
        if matched:
            chosen = sorted(matched)[0]
            print(f"[Auto-Golden] 已自動在快取中找到 {keyword.upper()} 權重: {chosen}")
            return chosen

    if candidates:
        chosen = sorted(candidates)[0]
        print(f"[Auto-Golden] 已自動在快取中找到權重檔: {chosen}")
        return chosen

    raise FileNotFoundError(f"找不到權重檔: {ckpt_path}，請確認 ~/.cache/torch/hub/checkpoints/ 是否有 .pth 檔！")


def inspect_and_extract_patch_proj(ckpt_path: str, device: torch.device):
    real_ckpt_path = resolve_dino_ckpt_path(ckpt_path)
    print(f"[Auto-Golden] 正在讀取本地權重檔: {real_ckpt_path}")
    ckpt = torch.load(real_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("teacher", ckpt.get("state_dict", ckpt)))

    proj_weight, proj_bias = None, None
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            proj_weight = v.float()
        elif "patch_embed.proj.bias" in k:
            proj_bias = v.float()

    block_ids = [
        int(k.split(".")[1]) for k in state_dict.keys()
        if "blocks." in k and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
    ]
    num_layers = max(block_ids) + 1 if block_ids else 0
    has_reg = any("register_tokens" in k for k in state_dict.keys())

    if proj_weight is not None:
        embed_dim, _, patch_h, patch_w = proj_weight.shape
        print(f"[DINO 規格自動偵測] Embed_Dim={embed_dim} | Patch={patch_h}x{patch_w} | Layers={num_layers} | Registers={has_reg}")
    else:
        raise ValueError("無法在權重檔中找到 patch_embed.proj.weight，請確認是否為 ViT 權重！")

    qkv_weights = []
    for i in range(min(6, num_layers)):
        for k, v in state_dict.items():
            if f"blocks.{i}.attn.qkv.weight" in k:
                qkv_weights.append(v.float().to(device))
                break

    proj_conv = nn.Conv2d(3, embed_dim, kernel_size=patch_h, stride=patch_h, bias=(proj_bias is not None))
    proj_conv.weight.data.copy_(proj_weight)
    if proj_bias is not None:
        proj_conv.bias.data.copy_(proj_bias)
    proj_conv = proj_conv.to(device).eval()

    return proj_conv, qkv_weights, patch_h


def _robust_mad_z(arr: np.ndarray, directional: str = "both") -> np.ndarray:
    """計算穩健 MAD Z-score：directional='both' 雙邊偏離，'high' 僅懲罰過高，'low' 僅懲罰過低。"""
    med = float(np.median(arr))
    mad = float(1.4826 * np.median(np.abs(arr - med)))
    std_fallback = float(np.std(arr)) * 0.5
    scale = max(mad, std_fallback, 1e-4)
    if directional == "high":
        return np.maximum(0.0, (arr - med) / scale)
    if directional == "low":
        return np.maximum(0.0, (med - arr) / scale)
    return np.abs(arr - med) / scale


def _select_diverse_top_indices(
    scores: np.ndarray,
    global_feats: np.ndarray,
    spatial_feats: np.ndarray,
    target_k: int,
    sim_thresh_global: float = 0.992,
    sim_thresh_spatial: float = 0.975,
) -> List[int]:
    """
    多樣性非極大值抑制 (Near-Duplicate Diversity NMS)：
    防止同一顆工件的多重曝光/微位移變體同時霸佔 Top-16 候選池並在 Stage 2 切塊投票中串通得票。
    """
    order = [int(i) for i in np.argsort(scores)[::-1]]
    if len(order) <= target_k:
        return order

    selected: List[int] = []
    deferred: List[int] = []

    for idx in order:
        if not selected:
            selected.append(idx)
            if len(selected) >= target_k:
                break
            continue

        g_sims = np.dot(global_feats[selected], global_feats[idx])
        s_sims = np.dot(spatial_feats[selected], spatial_feats[idx])

        is_near_dup = bool(
            np.any((g_sims >= sim_thresh_global) & (s_sims >= sim_thresh_spatial))
            or np.any(s_sims >= 0.985)
            or np.any(g_sims >= 0.9975)
        )
        if is_near_dup:
            deferred.append(idx)
        else:
            selected.append(idx)
            if len(selected) >= target_k:
                break

    for idx in deferred:
        if len(selected) >= target_k:
            break
        selected.append(idx)

    return selected


# =====================================================================
# Stage 1: 支援上千張大資料集子抽樣 + 四合一特徵過濾鎖定 Top-M (預設 16 張) 候選池
# =====================================================================
@torch.no_grad()
def select_golden_images(
    pool_dir: str,
    ckpt_path: str,
    out_dir: str,
    category: str = "wafer",
    shots: int = 4,
    report_dir: Optional[str] = None,
    log_files: Optional[List[str]] = None,
    candidate_pool_size: int = 16,
    max_coarse_scan: int = 250,
    sample_seed: int = 42,
) -> List[str]:
    """
    Stage 1 候選池初篩 (支援上千張大資料集快速抽樣 + 光度/空間位移/跨片局部異常/去重四合一過濾)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_imgs = scan_pool_images(pool_dir)
    total_pool_count = len(all_imgs)
    if total_pool_count < shots:
        raise ValueError(f"資料夾 (含子資料夾) 內圖片數量 ({total_pool_count}) 少於所需的 shots ({shots})！")

    eff_cand_size = min(total_pool_count, max(shots, int(candidate_pool_size)))

    if max_coarse_scan > 0 and total_pool_count > max(max_coarse_scan, eff_cand_size):
        rng = np.random.RandomState(sample_seed)
        sampled_indices = np.sort(rng.choice(total_pool_count, size=max_coarse_scan, replace=False))
        scan_imgs = [all_imgs[int(i)] for i in sampled_indices]
        log_and_print(
            f"[Auto-Golden Stage 1 大資料集抽樣啟動] 盲池共 {total_pool_count} 張圖片 (> 上限 {max_coarse_scan} 張)，"
            f"已固定 Seed={sample_seed} 均勻抽樣 {len(scan_imgs)} 張進行快速全域篩選 (目標鎖定 Top-{eff_cand_size} 候選池)...",
            log_files,
        )
    else:
        scan_imgs = all_imgs
        log_and_print(
            f"[Auto-Golden Stage 1] 遞迴掃描共找到 {total_pool_count} 張圖片，正在提取全域特徵以鎖定 Top-{eff_cand_size} 候選池...",
            log_files,
        )

    proj_conv, qkv_weights, patch_size = inspect_and_extract_patch_proj(ckpt_path, device)
    img_size = 518 if patch_size == 14 else 512
    grid_side = img_size // patch_size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    spatial_align_vecs = []
    grid16_maps = []
    token_spike_scores = []
    luma_medians, luma_stds, luma_spans = [], [], []
    valid_paths = []

    for path in scan_imgs:
        try:
            rgb_u8 = load_image_rgb_uint8(path)
            small_rgb = cv2.resize(rgb_u8, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
            gray = 0.299 * small_rgb[:, :, 0] + 0.587 * small_rgb[:, :, 1] + 0.114 * small_rgb[:, :, 2]
            luma_medians.append(float(np.median(gray)))
            luma_stds.append(float(np.std(gray)))
            p5, p95 = np.percentile(gray, [5.0, 95.0])
            luma_spans.append(float(p95 - p5))

            pil_img = Image.fromarray(rgb_u8, mode="RGB")
            x = transform(pil_img).unsqueeze(0).to(device)
            tokens_raw = proj_conv(x).flatten(2).transpose(1, 2)
            tokens = F.layer_norm(tokens_raw, (tokens_raw.shape[-1],))

            token_mean = tokens.mean(dim=1, keepdim=True)
            token_dev = torch.norm(tokens - token_mean, dim=-1).squeeze(0)
            top_k_tok = max(1, int(token_dev.numel() * 0.02))
            top_vals, _ = torch.topk(token_dev, top_k_tok)
            med_val = torch.median(token_dev).clamp(min=1e-6)
            spike_ratio = float((top_vals.mean() / med_val).item())
            token_spike_scores.append(spike_ratio)

            tok_2d = tokens.transpose(1, 2).reshape(1, -1, grid_side, grid_side)
            sp8 = F.adaptive_avg_pool2d(tok_2d, (8, 8)).flatten(1)
            sp8 = F.normalize(sp8, dim=-1)
            spatial_align_vecs.append(sp8.cpu())

            sp16 = F.adaptive_avg_pool2d(tok_2d, (16, 16)).squeeze(0).flatten(1).transpose(0, 1)
            sp16 = F.normalize(sp16, dim=-1)
            grid16_maps.append(sp16.cpu())

            feat_list = [tokens.mean(dim=1), tokens.std(dim=1)]
            for w in qkv_weights[:3]:
                qkv = F.linear(tokens, w)
                feat_list.append(qkv.std(dim=1))

            feat = F.normalize(torch.cat(feat_list, dim=-1), dim=-1)
            features.append(feat.cpu())
            valid_paths.append(path)
        except Exception as e:
            log_and_print(f"  [略過損毀圖片] {path}: {e}", log_files)

    del proj_conv, qkv_weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_valid = len(valid_paths)
    if num_valid < shots:
        raise ValueError(f"有效可讀取圖片數量 ({num_valid}) 少於所需的 shots ({shots})！")

    eff_cand_size = min(num_valid, max(shots, int(candidate_pool_size)))

    feats_tensor = torch.cat(features, dim=0)
    sim_matrix = torch.mm(feats_tensor, feats_tensor.t()).numpy()
    np.fill_diagonal(sim_matrix, 0.0)
    k_trim = max(1, int(round((num_valid - 1) * 0.60)))
    sorted_sims = np.sort(sim_matrix, axis=1)[:, ::-1]
    raw_centrality = sorted_sims[:, :k_trim].mean(axis=1)

    sp8_tensor = torch.cat(spatial_align_vecs, dim=0).numpy()
    sp8_med = np.median(sp8_tensor, axis=0, keepdims=True)
    sp8_med = sp8_med / (np.linalg.norm(sp8_med, axis=-1, keepdims=True) + 1e-8)
    spatial_alignment = np.sum(sp8_tensor * sp8_med, axis=-1)
    shift_z = _robust_mad_z(spatial_alignment, directional="low")

    sp16_stack = torch.stack(grid16_maps, dim=0).numpy()
    sp16_med = np.median(sp16_stack, axis=0, keepdims=True)
    sp16_med = sp16_med / (np.linalg.norm(sp16_med, axis=-1, keepdims=True) + 1e-8)
    patch_cos_dist = np.maximum(0.0, 1.0 - np.sum(sp16_stack * sp16_med, axis=-1))
    coord_med_dist = np.median(patch_cos_dist, axis=0, keepdims=True) + 1e-4
    rel_patch_dist = patch_cos_dist / coord_med_dist
    top3_patch_anomaly = np.sort(rel_patch_dist, axis=1)[:, -3:].mean(axis=1)
    patch_anom_z = _robust_mad_z(top3_patch_anomaly, directional="high")

    spike_arr = np.array(token_spike_scores, dtype=np.float32)
    spike_z = _robust_mad_z(spike_arr, directional="high")

    luma_med_z = _robust_mad_z(np.array(luma_medians, dtype=np.float32), directional="both")
    luma_std_z = _robust_mad_z(np.array(luma_stds, dtype=np.float32), directional="low")
    luma_span_z = _robust_mad_z(np.array(luma_spans, dtype=np.float32), directional="low")
    photo_z = np.maximum(luma_med_z, 0.75 * np.maximum(luma_std_z, luma_span_z))

    photo_penalty = 0.015 * np.maximum(0.0, photo_z - 1.0) + 0.045 * np.maximum(0.0, photo_z - 1.8)
    shift_penalty = 0.012 * np.maximum(0.0, shift_z - 0.8) + 0.035 * np.maximum(0.0, shift_z - 1.6)
    patch_penalty = 0.008 * np.clip(patch_anom_z, 0.0, 6.0)
    spike_penalty = 0.003 * np.clip(spike_z, 0.0, 6.0)

    hybrid_centrality = (
        0.55 * raw_centrality
        + 0.45 * spatial_alignment
        - photo_penalty
        - shift_penalty
        - patch_penalty
        - spike_penalty
    )

    feats_np = feats_tensor.numpy()
    cand_indices = _select_diverse_top_indices(
        scores=hybrid_centrality,
        global_feats=feats_np,
        spatial_feats=sp8_tensor,
        target_k=eff_cand_size,
    )
    selected_indices = cand_indices[:shots]
    selected_paths = [valid_paths[idx] for idx in selected_indices]

    cand_records = []
    for rank_i, idx in enumerate(cand_indices):
        src_p = valid_paths[idx]
        rel_n = os.path.relpath(src_p, pool_dir).replace("\\", "/")
        cand_records.append(
            {
                "cand_rank": int(rank_i + 1),
                "abs_path": os.path.abspath(src_p),
                "rel_path": rel_n,
                "hybrid_score": float(hybrid_centrality[idx]),
                "raw_centrality": float(raw_centrality[idx]),
                "spatial_align": float(spatial_alignment[idx]),
                "photo_z": float(photo_z[idx]),
                "patch_anom_z": float(patch_anom_z[idx]),
                "token_spike": float(spike_arr[idx]),
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    cand_json_path = os.path.join(out_dir, f"stage1_candidates_{category}.json")
    with open(cand_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pool_dir": os.path.abspath(pool_dir),
                "total_pool_images": int(total_pool_count),
                "scanned_images": int(num_valid),
                "candidate_pool_size": int(eff_cand_size),
                "shots": int(shots),
                "candidates": cand_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    dest_dir = os.path.join(out_dir, "train", category)
    safe_rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    record_txt = os.path.join(out_dir, f"selected_{shots}_golden_images.txt")
    with open(record_txt, "w", encoding="utf-8") as f:
        f.write(
            f"=== Stage 1 Coarse Top-{shots} Golden Images "
            f"(Scanned {num_valid}/{total_pool_count} imgs, Candidate Pool={eff_cand_size}) ===\n"
        )
        for i, idx in enumerate(selected_indices):
            src_path = valid_paths[idx]
            h_score = hybrid_centrality[idx]
            r_score = raw_centrality[idx]
            s_align = spatial_alignment[idx]
            p_z = photo_z[idx]
            s_ratio = spike_arr[idx]
            ext = os.path.splitext(src_path)[1].lower()
            dst_path = os.path.join(dest_dir, f"{i:03d}{ext}")
            safe_copy(src_path, dst_path)
            rel_name = os.path.relpath(src_path, pool_dir).replace("\\", "/")
            log_line = (
                f"Golden #{i+1} (-> {i:03d}{ext}): {rel_name} | "
                f"Hybrid_Score: {h_score:.6f} (Raw_Cent: {r_score:.6f}, Spatial_Align: {s_align:.4f}, "
                f"Photo_Z: {p_z:.2f}, Token_Spike: {s_ratio:.3f})"
            )
            log_and_print("  " + log_line, log_files)
            f.write(log_line + "\n")

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        safe_copy(record_txt, os.path.join(report_dir, f"selected_{shots}_golden_images.txt"))

    log_and_print(
        f"[Auto-Golden Stage 1] 已鎖定 {eff_cand_size} 張候選圖片池，並建立初始目錄: {dest_dir}",
        log_files,
    )
    return selected_paths


# =====================================================================
# Stage 2: 對 16 張候選池於 RAM 內切塊 -> 同座標跨片對決投票選出最終 4 張 Golden 原圖與 48 張純淨訓練圖塊
# =====================================================================
@torch.no_grad()
def _extract_lightweight_dino_tile_embeddings(
    tiles_rgb: List[np.ndarray],
    ckpt_path: str,
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    if not tiles_rgb:
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj_conv, qkv_weights, patch_size = inspect_and_extract_patch_proj(ckpt_path, device)
        eval_size = 252 if patch_size == 14 else 256
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        all_feats = []
        for start_i in range(0, len(tiles_rgb), batch_size):
            batch_np = []
            for arr in tiles_rgb[start_i:start_i + batch_size]:
                if arr.shape[0] != eval_size or arr.shape[1] != eval_size:
                    arr_resized = cv2.resize(arr, (eval_size, eval_size), interpolation=cv2.INTER_AREA)
                else:
                    arr_resized = arr
                batch_np.append(arr_resized)

            x = torch.from_numpy(np.stack(batch_np, axis=0)).permute(0, 3, 1, 2).float().to(device) / 255.0
            x = (x - mean_t) / std_t
            tokens = proj_conv(x).flatten(2).transpose(1, 2)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))

            feat_list = [tokens.mean(dim=1), tokens.std(dim=1), tokens.amax(dim=1) * 0.25]
            for w in qkv_weights[:3]:
                qkv = F.linear(tokens, w)
                feat_list.append(qkv.std(dim=1))

            feat = F.normalize(torch.cat(feat_list, dim=-1), dim=-1)
            all_feats.append(feat.cpu().numpy())

        del proj_conv, qkv_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(all_feats, axis=0)
    except Exception as e:
        print(f"[Stage-2 Hybrid Filter] DINO 圖塊特徵提取退回純像素殘差模式 ({e})")
        return None


def _compute_local_patch_peak_residuals(
    crops: List[np.ndarray],
    eval_res: int = 256,
    patch_grid: int = 16,
) -> np.ndarray:
    n = len(crops)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    blurred_stack = []
    norm_stack = []
    tile_means = []
    tile_stds = []

    for c in crops:
        small = cv2.resize(c, (eval_res, eval_res), interpolation=cv2.INTER_AREA).astype(np.float32)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        blurred_stack.append(blur)

        m_val = float(np.mean(blur))
        s_val = float(np.std(blur))
        tile_means.append(m_val)
        tile_stds.append(s_val)
        norm_stack.append((blur - m_val) / max(s_val, 4.0))

    arr = np.stack(blurred_stack, axis=0)
    med_tile = np.median(arr, axis=0, keepdims=True)
    abs_diff = np.mean(np.abs(arr - med_tile), axis=-1)

    norm_arr = np.stack(norm_stack, axis=0)
    med_norm_tile = np.median(norm_arr, axis=0, keepdims=True)
    struct_diff = np.mean(np.abs(norm_arr - med_norm_tile), axis=-1) * 18.0

    combined_diff = 0.50 * abs_diff + 0.50 * struct_diff

    cell_h = eval_res // patch_grid
    cell_w = eval_res // patch_grid
    patched = combined_diff.reshape(n, patch_grid, cell_h, patch_grid, cell_w).mean(axis=(2, 4)).reshape(n, -1)

    top2_patch_peak = np.sort(patched, axis=1)[:, -2:].mean(axis=1)
    global_l1 = combined_diff.mean(axis=(1, 2))

    means_arr = np.array(tile_means, dtype=np.float32)
    stds_arr = np.array(tile_stds, dtype=np.float32)
    mean_z = _robust_mad_z(means_arr, directional="both")
    std_low_z = _robust_mad_z(stds_arr, directional="low")
    exposure_penalty = 2.5 * np.maximum(0.0, mean_z - 1.2) + 3.0 * np.maximum(0.0, std_low_z - 1.2)

    return (0.65 * top2_patch_peak + 0.35 * global_l1 + exposure_penalty).astype(np.float32)


def purify_golden_tiles_two_stage(
    coord_Execution_items: List[Tuple[int, int, float, List[Tuple[int, np.ndarray]]]],
    max_train_tiles: int,
    ckpt_path: str = "auto",
    num_candidates: int = 4,
    shots: int = 4,
    stage1_priors: Optional[List[float]] = None,
) -> Tuple[List[Tuple[int, int, int, float]], Dict]:
    flat_tiles: List[np.ndarray] = []
    coord_slices: List[Tuple[int, int]] = []
    for (_, _, _, non_bg_items) in coord_Execution_items:
        start_idx = len(flat_tiles)
        for (_, crop) in non_bg_items:
            flat_tiles.append(crop)
        coord_slices.append((start_idx, len(flat_tiles)))

    total_evaluated_tiles = len(flat_tiles)
    dino_feats = _extract_lightweight_dino_tile_embeddings(flat_tiles, ckpt_path=ckpt_path, batch_size=32)

    purified_candidates = []
    rejected_outlier_tiles = 0

    cand_win_votes = np.zeros(max(1, num_candidates), dtype=np.int32)
    cand_cost_sums = np.zeros(max(1, num_candidates), dtype=np.float64)
    cand_eval_counts = np.zeros(max(1, num_candidates), dtype=np.int32)

    prior_arr = np.zeros(max(1, num_candidates), dtype=np.float32)
    if stage1_priors and len(stage1_priors) == num_candidates:
        s1_arr = np.array(stage1_priors, dtype=np.float32)
        s1_max = float(np.max(s1_arr))
        s1_mad = max(float(1.4826 * np.median(np.abs(s1_arr - np.median(s1_arr)))), 1e-4)
        prior_arr = np.clip((s1_max - s1_arr) / s1_mad, 0.0, 4.0) * 0.08

    for idx, (y, x, radius, non_bg_items) in enumerate(coord_Execution_items):
        s_idx, e_idx = coord_slices[idx]
        n_items = len(non_bg_items)
        if n_items == 1:
            g_idx_0 = int(non_bg_items[0][0])
            if g_idx_0 < len(cand_win_votes):
                cand_win_votes[g_idx_0] += 2
                cand_eval_counts[g_idx_0] += 1
            purified_candidates.append(
                {
                    "radius": float(radius),
                    "y": int(y),
                    "x": int(x),
                    "best_g_idx": g_idx_0,
                    "best_cost": 0.0,
                    "safe_extras": [],
                }
            )
            continue

        crops = [item[1] for item in non_bg_items]
        peak_res = _compute_local_patch_peak_residuals(crops)
        norm_peak = peak_res / (float(np.median(peak_res)) + 1e-4)

        if dino_feats is not None:
            sub_f = dino_feats[s_idx:e_idx]
            sim_mat = np.matmul(sub_f, sub_f.T)
            dist_mat = np.maximum(0.0, 1.0 - sim_mat)
            dino_dist = dist_mat.sum(axis=1) / max(1, n_items - 1)
            norm_dino = dino_dist / (float(np.median(dino_dist)) + 1e-5)
            hybrid_cost = 0.55 * norm_peak + 0.45 * norm_dino
            raw_abs_cost = 0.5 * peak_res + 10.0 * dino_dist
        else:
            hybrid_cost = norm_peak
            raw_abs_cost = peak_res

        for loc_i, (c_idx, _) in enumerate(non_bg_items):
            if int(c_idx) < len(prior_arr):
                hybrid_cost[loc_i] += float(prior_arr[int(c_idx)])

        sorted_order = np.argsort(hybrid_cost)
        best_local_i = int(sorted_order[0])
        best_g_idx = int(non_bg_items[best_local_i][0])
        best_abs_cost = float(raw_abs_cost[best_local_i])

        for rank_pos, loc_i in enumerate(sorted_order):
            c_idx = int(non_bg_items[int(loc_i)][0])
            if c_idx < len(cand_win_votes):
                cand_cost_sums[c_idx] += float(hybrid_cost[int(loc_i)])
                cand_eval_counts[c_idx] += 1
                if rank_pos == 0:
                    cand_win_votes[c_idx] += 2
                elif rank_pos == 1 and n_items >= 4:
                    cand_win_votes[c_idx] += 1

        med_cost = float(np.median(hybrid_cost))
        mad_cost = float(1.4826 * np.median(np.abs(hybrid_cost - med_cost))) + 1e-4
        adaptive_gate = min(float(hybrid_cost[best_local_i]) * 1.65 + 0.15, med_cost + 2.0 * mad_cost)

        safe_extras: List[Tuple[int, float]] = []
        if n_items >= 3:
            drop_worst_k = max(1, n_items // 4)
            candidate_extra_order = sorted_order[1:-drop_worst_k]
            rejected_outlier_tiles += drop_worst_k
        else:
            candidate_extra_order = sorted_order[1:]

        for loc_i in candidate_extra_order:
            loc_i_int = int(loc_i)
            if float(hybrid_cost[loc_i_int]) <= adaptive_gate:
                safe_extras.append((int(non_bg_items[loc_i_int][0]), float(raw_abs_cost[loc_i_int])))
            else:
                rejected_outlier_tiles += 1

        purified_candidates.append(
            {
                "radius": float(radius),
                "y": int(y),
                "x": int(x),
                "best_g_idx": best_g_idx,
                "best_cost": best_abs_cost,
                "safe_extras": safe_extras,
            }
        )

    cand_mean_costs = cand_cost_sums / np.maximum(1, cand_eval_counts)
    valid_cands = [i for i in range(len(cand_win_votes)) if cand_eval_counts[i] > 0]
    if not valid_cands:
        valid_cands = list(range(min(shots, num_candidates)))
    valid_cands.sort(key=lambda i: (-int(cand_win_votes[i]), float(cand_mean_costs[i]), i))
    winning_cand_indices = valid_cands[:min(shots, len(valid_cands))]

    purified_candidates.sort(key=lambda item: item["radius"])
    all_best_costs = np.array([c["best_cost"] for c in purified_candidates], dtype=np.float32)
    global_cost_p90 = float(np.percentile(all_best_costs, 90.0)) if all_best_costs.size > 0 else 999.0

    selected_specs: List[Tuple[int, int, int, float]] = []
    radial_purified_swaps = 0

    if len(purified_candidates) > max_train_tiles:
        bins = np.array_split(np.arange(len(purified_candidates)), max_train_tiles)
        for b_indices in bins:
            if len(b_indices) == 0:
                continue
            mid_idx = int(b_indices[len(b_indices) // 2])
            best_bin_idx = min(b_indices, key=lambda idx: purified_candidates[int(idx)]["best_cost"])
            best_bin_idx = int(best_bin_idx)
            if best_bin_idx != mid_idx:
                radial_purified_swaps += 1
            chosen = purified_candidates[best_bin_idx]
            selected_specs.append((chosen["y"], chosen["x"], chosen["best_g_idx"], chosen["best_cost"]))
    else:
        for c in purified_candidates:
            selected_specs.append((c["y"], c["x"], c["best_g_idx"], c["best_cost"]))

        if len(selected_specs) < max_train_tiles:
            extra_needed = max_train_tiles - len(selected_specs)
            extra_pool = []
            for c in purified_candidates:
                for (ex_g_idx, ex_cost) in c["safe_extras"]:
                    if ex_cost <= global_cost_p90 * 1.25:
                        extra_pool.append((ex_cost, c["y"], c["x"], ex_g_idx))
            extra_pool.sort(key=lambda item: item[0])
            for (ex_cost, ey, ex, eg_idx) in extra_pool[:extra_needed]:
                selected_specs.append((ey, ex, eg_idx, ex_cost))

    stats = {
        "total_evaluated_tiles": int(total_evaluated_tiles),
        "valid_grid_coords": int(len(purified_candidates)),
        "rejected_outlier_tiles": int(rejected_outlier_tiles),
        "radial_purified_swaps": int(radial_purified_swaps),
        "used_dino_stage2": int(dino_feats is not None),
        "winning_cand_indices": [int(i) for i in winning_cand_indices],
        "cand_win_votes": [int(v) for v in cand_win_votes],
        "cand_mean_costs": [float(c) for c in cand_mean_costs],
    }
    return selected_specs, stats


def prepare_tiled_golden_trainset(
    shot_dir: str,
    category: str,
    tile_size: int,
    tile_overlap: int,
    max_train_tiles: int,
    bg_std_thresh: float,
    log_files: Optional[List[str]] = None,
    ckpt_path: str = "auto",
    report_dir: Optional[str] = None,
    min_tile_trigger: int = 1280,
    shots: int = 4,
) -> int:
    """
    Stage 2 先切塊再投票選 Golden 流程 + 自動輸出 48 個訓練圖塊空間分佈視覺化圖 (`golden_tile_layout_vis.png`)。
    """
    cat_train_dir = os.path.join(shot_dir, "train", category)
    initial_files = sorted(
        [
            os.path.join(cat_train_dir, f)
            for f in os.listdir(cat_train_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]
    )
    if not initial_files:
        raise FileNotFoundError(f"在 {cat_train_dir} 中找不到初始 Golden 圖片！")

    cand_json_path = os.path.join(shot_dir, f"stage1_candidates_{category}.json")
    cand_records = []
    if os.path.isfile(cand_json_path):
        try:
            with open(cand_json_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            for item in c_data.get("candidates", []):
                if os.path.isfile(item.get("abs_path", "")):
                    cand_records.append(item)
        except Exception:
            cand_records = []

    sample_check_path = cand_records[0]["abs_path"] if cand_records else initial_files[0]
    sample_img = load_image_rgb_uint8(sample_check_path)
    h0, w0 = sample_img.shape[:2]

    full_backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    safe_rmtree(full_backup_dir)
    os.makedirs(full_backup_dir, exist_ok=True)

    eff_trigger = max(tile_size, min_tile_trigger)
    if max(h0, w0) <= eff_trigger:
        for p in initial_files[:shots]:
            safe_copy(p, os.path.join(full_backup_dir, os.path.basename(p)))
        log_and_print(
            f"[Full-Image Alignment Mode] 原始圖片尺寸 ({w0}x{h0}) <= 切塊啟動門檻 ({eff_trigger}px)，"
            f"自動保持整張原圖訓練以保留完整空間結構對齊 (共 {len(initial_files)} 張 Golden 原圖)。",
            log_files,
        )
        purity_txt_path = os.path.join(shot_dir, "golden_tile_purity_report.txt")
        lines = [
            "=== Stage 2: Full-Image Spatial Alignment Mode (No Sub-Tile Fragmentation) ===",
            f"Original Image Resolution : {w0}x{h0} (<= {eff_trigger}px trigger)",
            f"Retained Full Golden Imgs : {len(initial_files)}",
        ]
        with open(purity_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
            safe_copy(purity_txt_path, os.path.join(report_dir, "golden_tile_purity_report.txt"))

        save_golden_tile_layout_visualization(
            base_img_rgb=sample_img,
            all_valid_coords=[(0, 0)],
            selected_tile_records=[(0, 0, "full", 0.0)],
            tile_size=max(h0, w0),
            tile_overlap=0,
            num_cands=len(initial_files),
            out_vis_path=os.path.join(shot_dir, "golden_tile_layout_vis.png"),
            report_dir=report_dir,
            log_files=log_files,
        )
        del sample_img
        return len(initial_files)

    del sample_img

    if not cand_records:
        cand_records = [
            {
                "cand_rank": i + 1,
                "abs_path": os.path.abspath(p),
                "rel_path": os.path.basename(p),
                "hybrid_score": 1.0,
                "raw_centrality": 1.0,
                "token_spike": 1.0,
            }
            for i, p in enumerate(initial_files)
        ]

    candidate_imgs = []
    valid_cand_records = []
    for rec in cand_records:
        try:
            img = load_image_rgb_uint8(rec["abs_path"])
            candidate_imgs.append(img)
            valid_cand_records.append(rec)
        except Exception:
            pass

    num_cands = len(candidate_imgs)
    stride = max(1, tile_size - tile_overlap)
    ys = get_axis_positions(h0, tile_size, stride)
    xs = get_axis_positions(w0, tile_size, stride)
    grid_coords = [(y, x) for y in ys for x in xs]

    log_and_print(
        f"[Stage 2 候選池切塊投票] 正在將 {num_cands} 張 ({w0}x{h0}) 候選晶圓於 RAM 內切塊 "
        f"(共 {len(grid_coords)} 個網格座標)，進行跨片同座標純淨度對決...",
        log_files,
    )

    coord_Execution_items: List[Tuple[int, int, float, List[Tuple[int, np.ndarray]]]] = []
    for (y, x) in grid_coords:
        non_bg_items: List[Tuple[int, np.ndarray]] = []
        for c_idx, c_img in enumerate(candidate_imgs):
            gh, gw = c_img.shape[:2]
            py = min(y, max(0, gh - tile_size))
            px = min(x, max(0, gw - tile_size))
            ph = min(tile_size, gh - py)
            pw = min(tile_size, gw - px)
            crop = c_img[py:py + ph, px:px + pw]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            if not is_background_tile(crop, bg_std_thresh):
                non_bg_items.append((c_idx, crop))

        if not non_bg_items:
            continue

        cy, cx = y + tile_size * 0.5, x + tile_size * 0.5
        radius = float(np.hypot(cy - h0 * 0.5, cx - w0 * 0.5))
        coord_Execution_items.append((y, x, radius, non_bg_items))

    if not coord_Execution_items:
        log_and_print("[Tiling] 警告：所有切塊皆低於背景門檻，退回保留全部切塊。", log_files)
        for (y, x) in grid_coords:
            c0 = candidate_imgs[0]
            crop0 = c0[y:y + tile_size, x:x + tile_size]
            coord_Execution_items.append((y, x, 0.0, [(0, crop0)]))

    s1_priors = [float(r.get("hybrid_score", 1.0)) for r in valid_cand_records]
    selected_specs, purity_stats = purify_golden_tiles_two_stage(
        coord_Execution_items=coord_Execution_items,
        max_train_tiles=max_train_tiles,
        ckpt_path=ckpt_path,
        num_candidates=num_cands,
        shots=shots,
        stage1_priors=s1_priors,
    )

    winning_cand_indices = purity_stats["winning_cand_indices"]
    cand_win_votes = purity_stats["cand_win_votes"]
    cand_mean_costs = purity_stats["cand_mean_costs"]

    cand_to_final_g_map: Dict[int, int] = {}
    record_txt = os.path.join(shot_dir, f"selected_{shots}_golden_images.txt")
    with open(record_txt, "w", encoding="utf-8") as f:
        f.write(
            f"=== Auto-Selected Golden Images (Stage 2 Tile-First Voting from {num_cands} Candidates) ===\n"
        )
        for final_g_i, c_idx in enumerate(winning_cand_indices):
            cand_to_final_g_map[c_idx] = final_g_i
            rec = valid_cand_records[c_idx]
            src_path = rec["abs_path"]
            ext = os.path.splitext(src_path)[1].lower()
            backup_dst = os.path.join(full_backup_dir, f"{final_g_i:03d}{ext}")
            safe_copy(src_path, backup_dst)
            log_line = (
                f"Golden #{final_g_i + 1} (-> {final_g_i:03d}{ext}): {rec['rel_path']} | "
                f"Tile_Win_Votes: {cand_win_votes[c_idx]} | Mean_Tile_Cost: {cand_mean_costs[c_idx]:.4f} | "
                f"Stage1_Score: {rec['hybrid_score']:.6f}"
            )
            log_and_print("  [切塊投票勝出] " + log_line, log_files)
            f.write(log_line + "\n")

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        safe_copy(record_txt, os.path.join(report_dir, f"selected_{shots}_golden_images.txt"))

    for f in os.listdir(cat_train_dir):
        if f.lower().endswith(IMG_EXTENSIONS):
            p = os.path.join(cat_train_dir, f)
            try:
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            os.remove(p)

    purity_report_lines = [
        "=== Stage 2: Tile-First Candidate Pool Voting & Purification Report ===",
        f"Candidate Pool Size (Stage 1->2): {num_cands} full-res images",
        f"Total Evaluated Candidate Tiles : {purity_stats['total_evaluated_tiles']}",
        f"Valid Non-Background Coords     : {purity_stats['valid_grid_coords']}",
        f"Rejected Outlier Tiles (Stage2) : {purity_stats['rejected_outlier_tiles']}",
        f"Radial Bin Purity Swaps         : {purity_stats['radial_purified_swaps']}",
        f"DINO Feature Guidance Active    : {'YES' if purity_stats['used_dino_stage2'] else 'FALLBACK_PIXEL_PEAK'}",
        "-" * 78,
        "--- Candidate Pool Tile-Level Voting Leaderboard ---",
    ]
    for rank_i, c_idx in enumerate(
        sorted(range(num_cands), key=lambda i: (-cand_win_votes[i], cand_mean_costs[i]))
    ):
        rec = valid_cand_records[c_idx]
        badge = f"🏆 SELECTED GOLDEN #{cand_to_final_g_map[c_idx] + 1}" if c_idx in cand_to_final_g_map else "   Candidate"
        purity_report_lines.append(
            f"Rank #{rank_i + 1:02d} [{badge}]: {rec['rel_path']} | "
            f"Win_Votes={cand_win_votes[c_idx]:03d} | Mean_Tile_Cost={cand_mean_costs[c_idx]:.4f}"
        )
    purity_report_lines.append("-" * 78)

    vis_selected_records: List[Tuple[int, int, str, float]] = []
    for idx, (y, x, c_idx, p_cost) in enumerate(selected_specs):
        c_img = candidate_imgs[c_idx]
        gh, gw = c_img.shape[:2]
        py = min(y, max(0, gh - tile_size))
        px = min(x, max(0, gw - tile_size))
        ph = min(tile_size, gh - py)
        pw = min(tile_size, gw - px)
        patch = c_img[py:py + ph, px:px + pw]
        if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
            patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        g_tag = f"g{cand_to_final_g_map[c_idx]}" if c_idx in cand_to_final_g_map else f"c{c_idx:02d}"
        tile_fname = f"{idx:03d}_{g_tag}_tile_y{y:04d}_x{x:04d}.png"
        out_path = os.path.join(cat_train_dir, tile_fname)
        Image.fromarray(patch).save(out_path)
        vis_selected_records.append((y, x, g_tag, p_cost))
        purity_report_lines.append(
            f"Tile #{idx + 1:02d}: {tile_fname} | Source={valid_cand_records[c_idx]['rel_path']} | "
            f"Coord=(y={y:04d}, x={x:04d}) | Hybrid_Residual={p_cost:.4f}"
        )

    purity_txt_path = os.path.join(shot_dir, "golden_tile_purity_report.txt")
    with open(purity_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(purity_report_lines) + "\n")
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        safe_copy(purity_txt_path, os.path.join(report_dir, "golden_tile_purity_report.txt"))

    # 輸出 48 個代表性訓練圖塊與同心圓環帶視覺化對照圖
    top1_cand_idx = winning_cand_indices[0] if winning_cand_indices else 0
    valid_xy_coords = [(item[0], item[1]) for item in coord_Execution_items]
    save_golden_tile_layout_visualization(
        base_img_rgb=candidate_imgs[top1_cand_idx],
        all_valid_coords=valid_xy_coords,
        selected_tile_records=vis_selected_records,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_cands=num_cands,
        out_vis_path=os.path.join(shot_dir, "golden_tile_layout_vis.png"),
        report_dir=report_dir,
        log_files=log_files,
    )

    num_saved = len(selected_specs)
    del candidate_imgs, coord_Execution_items
    gc.collect()

    log_and_print(
        f"[Tiling-Train 切塊投票與二次純化完成] 已從 {num_cands} 張 ({w0}x{h0}) 候選晶圓評估 "
        f"{purity_stats['total_evaluated_tiles']} 張子圖塊：\n"
        f"  ├── 切塊投票選出最終 Golden 原圖 : 已從 {num_cands} 張候選池正式選出 {len(winning_cand_indices)} 張最純淨 Golden 原圖\n"
        f"  ├── 跨片局部峰值與 DINO 離群剔除 : 自動攔截剔除 {purity_stats['rejected_outlier_tiles']} 張可疑離群圖塊\n"
        f"  ├── 徑向分層純淨度優化替換       : {purity_stats['radial_purified_swaps']} 個徑向區間已自動替換為更純淨座標\n"
        f"  ├── 切塊分佈視覺化圖檔           : {os.path.abspath(os.path.join(shot_dir, 'golden_tile_layout_vis.png'))}\n"
        f"  └── 最終輸出純淨代表性訓練圖塊   : {num_saved} 張 ({tile_size}x{tile_size}) -> {os.path.abspath(cat_train_dir)}",
        log_files,
    )
    return num_saved


def _recover_golden_filenames_from_disk(shot_dir: str, shots: int, category: str) -> List[str]:
    """從既有 selected_*_golden_images.txt 或備份資料夾推斷已挑選的 Golden 檔名清單。"""
    txt_candidates = [os.path.join(shot_dir, f"selected_{shots}_golden_images.txt")] + glob.glob(
        os.path.join(shot_dir, "selected_*_golden_images.txt")
    )
    for rec_txt in txt_candidates:
        if os.path.isfile(rec_txt):
            names = []
            try:
                with open(rec_txt, "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.search(r":\s*([^|]+?)\s*(?:\||$)", line)
                        if m and "Golden #" in line:
                            names.append(m.group(1).strip())
                if names:
                    return names
            except Exception:
                pass

    backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    if os.path.isdir(backup_dir):
        b_files = sorted([f for f in os.listdir(backup_dir) if f.lower().endswith(IMG_EXTENSIONS)])
        if b_files:
            return b_files

    cat_train_dir = os.path.join(shot_dir, "train", category)
    if os.path.isdir(cat_train_dir):
        t_files = sorted([f for f in os.listdir(cat_train_dir) if f.lower().endswith(IMG_EXTENSIONS)])
        if t_files:
            return t_files[:shots]
    return []


def run_auto_golden_pipeline(
    pool_dir: str,
    shot_dir: str,
    category: str,
    ckpt_path: str = "auto",
    shots: int = 4,
    tile_size: int = 512,
    tile_overlap: int = 64,
    max_train_tiles: int = 48,
    bg_std_thresh: float = 1.5,
    disable_tiling: bool = False,
    report_dir: Optional[str] = None,
    log_files: Optional[List[str]] = None,
    min_tile_trigger: int = 1280,
    candidate_pool_size: int = 16,
    max_coarse_scan: int = 250,
) -> Dict:
    """執行完整 Stage 1 (大資料集抽樣 + 16 張候選池鎖定) + Stage 2 (RAM 切塊投票選出最終 4 張 Golden 與 48 張訓練圖塊)。"""
    safe_rmtree(shot_dir)
    os.makedirs(shot_dir, exist_ok=True)
    pool_signature = compute_pool_signature(pool_dir)

    t_s1 = time.perf_counter()
    golden_paths = select_golden_images(
        pool_dir=pool_dir,
        ckpt_path=ckpt_path,
        out_dir=shot_dir,
        category=category,
        shots=shots,
        report_dir=report_dir,
        log_files=log_files,
        candidate_pool_size=candidate_pool_size,
        max_coarse_scan=max_coarse_scan,
    )
    golden_select_sec = time.perf_counter() - t_s1
    golden_filenames = [
        os.path.relpath(p, pool_dir).replace("\\", "/") for p in golden_paths
    ]
    log_and_print(
        f"[⏱️ 計時] Stage 1 候選池鎖定耗時: {format_duration(golden_select_sec)}",
        log_files,
    )

    t_s2 = time.perf_counter()
    if not disable_tiling:
        num_train_samples = prepare_tiled_golden_trainset(
            shot_dir=shot_dir,
            category=category,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_train_tiles=max_train_tiles,
            bg_std_thresh=bg_std_thresh,
            log_files=log_files,
            ckpt_path=ckpt_path,
            report_dir=report_dir,
            min_tile_trigger=min_tile_trigger,
            shots=shots,
        )
        voted_names = _recover_golden_filenames_from_disk(shot_dir, shots, category)
        if voted_names:
            golden_filenames = voted_names
    else:
        num_train_samples = len(golden_filenames)
        _ensure_layout_vis_from_existing_dataset(
            shot_dir=shot_dir,
            category=category,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            bg_std_thresh=bg_std_thresh,
            report_dir=report_dir,
            log_files=log_files,
        )
    train_tile_sec = time.perf_counter() - t_s2

    meta = {
        "pool_signature": pool_signature,
        "category": category,
        "shots": int(shots),
        "candidate_pool_size": int(candidate_pool_size),
        "max_coarse_scan": int(max_coarse_scan),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "max_train_tiles": int(max_train_tiles),
        "disable_tiling": bool(disable_tiling),
        "min_tile_trigger": int(min_tile_trigger),
        "golden_filenames": golden_filenames,
        "num_train_samples": int(num_train_samples),
        "golden_select_sec": float(golden_select_sec),
        "train_tile_sec": float(train_tile_sec),
    }
    golden_meta_path = os.path.join(shot_dir, "golden_cache_meta.json")
    with open(golden_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def load_or_prepare_golden_dataset(
    pool_dir: str,
    shot_dir: str,
    category: str,
    ckpt_path: str = "auto",
    shots: int = 4,
    tile_size: int = 512,
    tile_overlap: int = 64,
    max_train_tiles: int = 48,
    bg_std_thresh: float = 1.5,
    disable_tiling: bool = False,
    rebuild_golden: bool = False,
    report_dir: Optional[str] = None,
    log_files: Optional[List[str]] = None,
    min_tile_trigger: int = 1280,
    candidate_pool_size: int = 16,
    max_coarse_scan: int = 250,
) -> Tuple[List[str], int, bool, float, float]:
    """
    防呆機制：
    1. 只要未指定 rebuild_golden=True，且已存在挑好的 Golden Image，絕不重新執行挑選！
    2. 若既有資料夾尚無 `golden_tile_layout_vis.png`，自動從既有訓練切塊補畫視覺化圖。
    """
    cat_train_dir = os.path.join(shot_dir, "train", category)
    backup_dir = os.path.join(shot_dir, "golden_full_res_backup", category)
    golden_meta_path = os.path.join(shot_dir, "golden_cache_meta.json")

    if not rebuild_golden and os.path.isdir(cat_train_dir):
        train_files = sorted([
            f for f in os.listdir(cat_train_dir) if f.lower().endswith(IMG_EXTENSIONS)
        ])
        if train_files:
            has_fragmented_tiles = any("_tile_y" in f for f in train_files)
            if has_fragmented_tiles and os.path.isdir(backup_dir):
                backup_files = sorted([
                    os.path.join(backup_dir, f)
                    for f in os.listdir(backup_dir)
                    if f.lower().endswith(IMG_EXTENSIONS)
                ])
                if backup_files:
                    sample_img = load_image_rgb_uint8(backup_files[0])
                    bh, bw = sample_img.shape[:2]
                    eff_trigger = max(tile_size, min_tile_trigger)
                    if disable_tiling or max(bh, bw) <= eff_trigger:
                        for f in train_files:
                            p = os.path.join(cat_train_dir, f)
                            try:
                                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
                                os.remove(p)
                            except OSError:
                                pass
                        for bp in backup_files:
                            safe_copy(bp, os.path.join(cat_train_dir, os.path.basename(bp)))
                        train_files = sorted([
                            f for f in os.listdir(cat_train_dir) if f.lower().endswith(IMG_EXTENSIONS)
                        ])
                        log_and_print(
                            f"[🔧 自動修復碎片化 Golden] 偵測到原圖尺寸 ({bw}x{bh}) <= {eff_trigger}px，"
                            f"已直接從備份還原既有 {len(train_files)} 張完整 Golden 原圖 (保留原挑選名單，不重挑 Golden)！",
                            log_files,
                        )

            golden_filenames = []
            if os.path.isfile(golden_meta_path):
                try:
                    with open(golden_meta_path, "r", encoding="utf-8") as f:
                        g_meta = json.load(f)
                    golden_filenames = list(g_meta.get("golden_filenames", []))
                    if int(g_meta.get("num_train_samples", -1)) != len(train_files):
                        g_meta["num_train_samples"] = len(train_files)
                        with open(golden_meta_path, "w", encoding="utf-8") as fw:
                            json.dump(g_meta, fw, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            if not golden_filenames:
                golden_filenames = _recover_golden_filenames_from_disk(shot_dir, shots, category)
                try:
                    meta_patch = {
                        "pool_signature": compute_pool_signature(pool_dir),
                        "category": category,
                        "shots": int(shots),
                        "candidate_pool_size": int(candidate_pool_size),
                        "max_coarse_scan": int(max_coarse_scan),
                        "tile_size": int(tile_size),
                        "tile_overlap": int(tile_overlap),
                        "max_train_tiles": int(max_train_tiles),
                        "disable_tiling": bool(disable_tiling),
                        "min_tile_trigger": int(min_tile_trigger),
                        "golden_filenames": golden_filenames,
                        "num_train_samples": len(train_files),
                        "golden_select_sec": 0.0,
                        "train_tile_sec": 0.0,
                    }
                    with open(golden_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta_patch, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            _ensure_layout_vis_from_existing_dataset(
                shot_dir=shot_dir,
                category=category,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                bg_std_thresh=bg_std_thresh,
                report_dir=report_dir,
                log_files=log_files,
            )

            num_train_samples = len(train_files)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
                rec_src = os.path.join(shot_dir, f"selected_{shots}_golden_images.txt")
                if os.path.exists(rec_src):
                    safe_copy(rec_src, os.path.join(report_dir, f"selected_{shots}_golden_images.txt"))
                purity_src = os.path.join(shot_dir, "golden_tile_purity_report.txt")
                if os.path.exists(purity_src):
                    safe_copy(purity_src, os.path.join(report_dir, "golden_tile_purity_report.txt"))

            log_and_print(
                f"[⚡ 防呆略過 Golden 重挑] 偵測到已存在 Golden 訓練集 ({cat_train_dir})，"
                f"直接沿用既有 {len(golden_filenames)} 張 Golden 原圖 ({', '.join(golden_filenames)}) "
                f"與 {num_train_samples} 張訓練樣本 (若需強制重挑請加上 --rebuild_golden)",
                log_files,
            )
            return golden_filenames, num_train_samples, True, 0.0, 0.0

    meta = run_auto_golden_pipeline(
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
        report_dir=report_dir,
        log_files=log_files,
        min_tile_trigger=min_tile_trigger,
        candidate_pool_size=candidate_pool_size,
        max_coarse_scan=max_coarse_scan,
    )
    return (
        meta["golden_filenames"],
        meta["num_train_samples"],
        False,
        meta["golden_select_sec"],
        meta["train_tile_sec"],
    )


def main():
    parser = argparse.ArgumentParser(description="FoundAD Stage 1 (大資料集抽樣候選池) + Stage 2 (先切塊再投票選 Golden 與視覺化輸出) 工具")
    parser.add_argument("--pool_dir", type=str, required=True, help="無標籤測試盲池圖片資料夾路徑")
    parser.add_argument("--ckpt", type=str, default="auto", help="DINO 預訓練權重路徑")
    parser.add_argument("--datasets_dir", type=str, default="./datasets", help="資料集根目錄")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot", help="資料集名稱")
    parser.add_argument("--category", type=str, default="auto", help="產品/晶圓類別名稱")
    parser.add_argument("--shots", type=int, default=4, help="最終選出的 Golden 原圖張數")
    parser.add_argument("--candidate_pool_size", type=int, default=16, help="Stage 1 初篩進入切塊投票的候選晶圓張數 (預設 16 張)")
    parser.add_argument("--max_coarse_scan", type=int, default=250, help="面對上千張大資料集時，Stage 1 抽樣評估的最大圖片上限 (預設 250 張)")
    parser.add_argument("--tile_size", type=int, default=512, help="Stage 2 子圖塊像素大小")
    parser.add_argument("--tile_overlap", type=int, default=64, help="相鄰子圖塊重疊像素寬度")
    parser.add_argument("--max_train_tiles", type=int, default=48, help="Stage 2 最終保留的代表性訓練圖塊上限")
    parser.add_argument("--min_tile_trigger", type=int, default=1280, help="原圖長或寬大於此像素值才啟動切塊 (<=此值保持整張原圖)")
    parser.add_argument("--bg_std_thresh", type=float, default=1.5, help="過濾純背景角落塊的標準差門檻")
    parser.add_argument("--disable_tiling", action="store_true", help="關閉切塊，直接使用整張 Golden 原圖")
    parser.add_argument("--rebuild_golden", action="store_true", help="強制刪除既有 Golden 並重新挑選")
    parser.add_argument("--mock_from_mvtec", type=str, default="", help="從 MVTec 類別目錄建立模擬無 GT 盲池")
    parser.add_argument("--report_dir", type=str, default="", help="額外複製挑選與純化報表的輸出目錄")
    args = parser.parse_args()

    if not args.category or args.category.lower() == "auto":
        if args.mock_from_mvtec:
            args.category = os.path.basename(os.path.normpath(args.mock_from_mvtec))
        else:
            args.category = os.path.basename(os.path.normpath(args.pool_dir))

    if args.mock_from_mvtec:
        prepare_mock_unlabeled_pool_from_mvtec(
            args.mock_from_mvtec,
            args.pool_dir,
            defect_limit=8,
            rebuild=args.rebuild_golden,
        )

    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    shot_dir = os.path.join(abs_datasets_dir, args.data_name)
    report_dir = os.path.abspath(args.report_dir) if args.report_dir else None

    if not args.rebuild_golden and not os.path.isdir(os.path.join(shot_dir, "train", args.category)):
        legacy_shot_dir = os.path.join(abs_datasets_dir, f"mvtec_sim_{args.shots}shot")
        if os.path.isdir(os.path.join(legacy_shot_dir, "train", args.category)):
            print(f"[🔄 向下相容命中] 自動沿用舊版命名 Golden 目錄: {legacy_shot_dir}")
            shot_dir = legacy_shot_dir

    golden_filenames, num_train_samples, used_existing, _, _ = load_or_prepare_golden_dataset(
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
        report_dir=report_dir,
        min_tile_trigger=args.min_tile_trigger,
        candidate_pool_size=args.candidate_pool_size,
        max_coarse_scan=args.max_coarse_scan,
    )
    status_str = "⚡ 直接沿用既有 Golden (未重挑)" if used_existing else "✅ 新建完成 (16張候選池切塊投票)"
    print(
        f"\n[{status_str}]\n"
        f"  ├── 最終勝出 Golden 原圖 : {', '.join(golden_filenames)}\n"
        f"  ├── 最終訓練樣本數量     : {num_train_samples} 張\n"
        f"  ├── 切塊分佈視覺化圖檔   : {os.path.join(shot_dir, 'golden_tile_layout_vis.png')}\n"
        f"  └── 輸出資料集目錄       : {shot_dir}"
    )


if __name__ == "__main__":
    main()