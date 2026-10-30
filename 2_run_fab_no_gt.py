# 檔案位置: FoundAD/2_run_fab_no_gt.py (全選覆蓋並存檔)
import os
import re
import csv
import stat
import glob
import json
import shutil
import subprocess
import argparse
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from fab_tools.auto_golden import select_golden_images, IMG_EXTENSIONS


def safe_rmtree(target_dir: str):
    """安全刪除可能含有唯讀檔案的資料夾。"""
    if not os.path.exists(target_dir):
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


def prepare_ckpt_for_demo(data_name: str, diy_name: str, req_step: int) -> int:
    log_dirs = glob.glob(f"./logs/{data_name}/*{diy_name}*") or glob.glob(f"./logs/{data_name}/*")
    chosen_step = req_step
    found_any = False
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
            print(f"[Auto-Ckpt] 已將訓練權重 {target_ckpt} 對接至 {pretrained_alias}")
            found_any = True

    if not found_any:
        raise FileNotFoundError(
            f"在 ./logs/{data_name}/ 下找不到任何 train-step*.pth.tar 權重檔！代表 Step 2 訓練未成功完成。"
        )
    return chosen_step


def prepare_mock_unlabeled_pool_from_mvtec(mvtec_cat_dir: str, mock_pool_dir: str, defect_limit: int = 8):
    safe_rmtree(mock_pool_dir)
    os.makedirs(mock_pool_dir, exist_ok=True)
    test_dir = os.path.join(mvtec_cat_dir, "test")

    idx = 1
    good_dir = os.path.join(test_dir, "good")
    for f in sorted(os.listdir(good_dir)):
        if f.lower().endswith(IMG_EXTENSIONS):
            safe_copy(os.path.join(good_dir, f), os.path.join(mock_pool_dir, f"wafer_{idx:04d}_clean.png"))
            idx += 1

    defect_count = 0
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good":
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for f in sorted(os.listdir(sub_dir)):
            if f.lower().endswith(IMG_EXTENSIONS) and defect_count < defect_limit:
                safe_copy(os.path.join(sub_dir, f), os.path.join(mock_pool_dir, f"wafer_{idx:04d}_escape_{sub}.png"))
                defect_count += 1
                idx += 1

    print(f"[模擬無GT盲池建立完成] 共 {idx-1} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵) -> {mock_pool_dir}")


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
    """判斷是否為晶圓圓周外部的純黑/純白無資訊背景塊。"""
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
) -> int:
    """
    將 Step 1 挑出的 4 張高解析度 Golden Wafer 切分為原生解析度子圖塊，
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
        print(f"[Tiling] 原始圖片尺寸 ({w0}x{h0}) <= tile_size ({tile_size})，保持原圖訓練。")
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
        print("[Tiling] 警告：所有切塊皆低於背景門檻，退回保留全部切塊。")
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

    print(
        f"[Tiling-Train] 已將 {len(golden_files)} 張 ({w0}x{h0}) Golden 大圖無損切分為 "
        f"{len(selected_tiles)} 張 ({tile_size}x{tile_size}) 高解析度代表性訓練圖塊！"
    )
    return len(selected_tiles)


def prepare_tiled_test_pool(
    pool_dir: str,
    tiled_pool_dir: str,
    tile_size: int,
    tile_overlap: int,
    bg_std_thresh: float,
) -> Dict:
    """
    將測試盲池中的所有高解析度晶圓圖片切分為重疊子圖塊 (tile_size x tile_size)，
    並回傳完整的座標映射表 (manifest) 供推論後無縫縫合回 5000x5000。
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

        ys = get_axis_positions(h, tile_size, stride)
        xs = get_axis_positions(w, tile_size, stride)

        tiles_info = []
        for y in ys:
            for x in xs:
                ch = min(tile_size, h - y)
                cw = min(tile_size, w - x)
                crop = img[y:y + ch, x:x + cw]

                if is_background_tile(crop, bg_std_thresh) and (len(ys) * len(xs) > 1):
                    skipped_bg_tiles += 1
                    continue

                if ch != tile_size or cw != tile_size:
                    crop_save = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
                else:
                    crop_save = crop

                tile_stem = f"{stem}__tile_y{y:04d}_x{x:04d}_h{ch:04d}_w{cw:04d}"
                tile_fname = f"{tile_stem}.png"
                Image.fromarray(crop_save).save(os.path.join(tiled_pool_dir, tile_fname))

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
            cy, cx = max(0, (h - tile_size) // 2), max(0, (w - tile_size) // 2)
            ch, cw = min(tile_size, h - cy), min(tile_size, w - cx)
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

    manifest_path = os.path.join(tiled_pool_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"[Tiling-Test] 已將 {len(img_files)} 張原圖切分為 {total_tiles} 張 ({tile_size}x{tile_size}) 子圖塊 "
        f"(自動跳過 {skipped_bg_tiles} 張純背景角落塊) -> {tiled_pool_dir}"
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
    """將單張 5000x5000 晶圓的所有子圖塊 .npy 無縫縫合回原圖完整解析度 (H, W)。"""
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


def stitch_and_visualize_full_res(
    manifest: Dict,
    tile_amap_dir: str,
    final_amap_dir: str,
    out_dir: str,
    golden_filenames: List[str],
    sigma_k: float,
):
    os.makedirs(final_amap_dir, exist_ok=True)
    bbox_dir = os.path.join(out_dir, "bbox_images")
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    tile_overlap = manifest["tile_overlap"]
    golden_set = set(golden_filenames)

    golden_sampled_pixels = []
    all_sampled_pixels = []
    stitched_paths: Dict[str, str] = {}

    print("\n[Step 4.1] 正在將子圖塊無縫拼接回原圖完整解析度...")
    for fname, meta in manifest["images"].items():
        full_amap = stitch_single_image_amap(meta, tile_amap_dir, tile_overlap)
        stem = meta["stem"]

        save_dtype = np.float16 if max(full_amap.shape) > 2048 else np.float32
        npy_out_path = os.path.join(final_amap_dir, f"{stem}.npy")
        np.save(npy_out_path, full_amap.astype(save_dtype))
        stitched_paths[fname] = npy_out_path

        step = max(1, int(max(full_amap.shape) // 512))
        sub_pixels = full_amap[::step, ::step].reshape(-1).astype(np.float32)
        all_sampled_pixels.append(sub_pixels)
        if fname in golden_set:
            golden_sampled_pixels.append(sub_pixels)

    ref_pixels = (
        np.concatenate(golden_sampled_pixels, axis=0)
        if golden_sampled_pixels
        else np.concatenate(all_sampled_pixels, axis=0)
    )
    mu = float(np.mean(ref_pixels))
    sigma = float(np.std(ref_pixels)) + 1e-8
    threshold = mu + sigma_k * sigma
    g_min, g_max = float(np.min(ref_pixels)), float(max(np.max(ref_pixels), threshold * 1.2))

    print(
        f"[Step 4.2] 全解析度統計門檻計算完成: μ={mu:.6f}, σ={sigma:.6f}, "
        f"k={sigma_k} -> 動態門檻 Threshold={threshold:.6f}"
    )

    ranking_records = []
    for fname, meta in manifest["images"].items():
        stem = meta["stem"]
        h, w = meta["height"], meta["width"]
        full_amap = np.load(stitched_paths[fname]).astype(np.float32)
        if full_amap.shape[0] != h or full_amap.shape[1] != w:
            full_amap = cv2.resize(full_amap, (w, h), interpolation=cv2.INTER_CUBIC)

        orig_bgr = cv2.imread(meta["path"])
        if orig_bgr is None:
            orig_rgb = np.array(Image.open(meta["path"]).convert("RGB"))
            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

        max_score = float(np.max(full_amap))
        top_k_count = max(10, int(full_amap.size * 0.0005))
        top_k_score = float(np.mean(np.partition(full_amap.reshape(-1), -top_k_count)[-top_k_count:]))

        binary_mask = (full_amap >= threshold).astype(np.uint8) * 255
        k_size = max(3, int(round(max(h, w) / 1000)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        min_area = max(4, int((max(h, w) / 1000.0) ** 2 * 4))

        vis_scale = max(1.0, max(h, w) / 900.0)
        box_thickness = max(2, int(round(2.5 * vis_scale)))
        font_scale = max(0.5, 0.55 * vis_scale)
        font_thickness = max(1, int(round(1.5 * vis_scale)))

        bbox_canvas = orig_bgr.copy()
        bboxes = []
        for lbl_idx in range(1, num_labels):
            bx, by, bw, bh, area = stats[lbl_idx]
            if area < min_area:
                continue
            region_scores = full_amap[by:by + bh, bx:bx + bw]
            region_max = float(np.max(region_scores))

            pad = max(3, int(round(4 * vis_scale)))
            x1, y1 = max(0, int(bx - pad)), max(0, int(by - pad))
            x2, y2 = min(w - 1, int(bx + bw + pad)), min(h - 1, int(by + bh + pad))
            bboxes.append((x1, y1, x2, y2, region_max, int(area)))

            cv2.rectangle(bbox_canvas, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            label_txt = f"{region_max:.3f}"
            (tw, th_txt), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            ty1 = max(0, y1 - th_txt - 8)
            cv2.rectangle(bbox_canvas, (x1, ty1), (x1 + tw + 6, ty1 + th_txt + 6), (0, 0, 255), -1)
            cv2.putText(
                bbox_canvas, label_txt, (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        is_golden = fname in golden_set
        status_tag = "GOLDEN" if is_golden else ("ESCAPE_SUSPECT" if bboxes else "CLEAN")
        bbox_out_path = os.path.join(bbox_dir, f"{stem}_bbox.png")
        cv2.imwrite(bbox_out_path, bbox_canvas)
        if bboxes:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_bbox.png"), bbox_canvas)

        norm_map = np.clip((full_amap - g_min) / max(g_max - g_min, 1e-6), 0.0, 1.0)
        hm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_bgr, 0.6, hm_color, 0.4, 0)
        for (x1, y1, x2, y2, _, _) in bboxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
        cv2.imwrite(os.path.join(heatmap_dir, f"{stem}_heatmap.png"), overlay)

        ranking_records.append({
            "filename": fname,
            "status": status_tag,
            "max_score": max_score,
            "top_k_score": top_k_score,
            "num_bboxes": len(bboxes),
            "resolution": f"{w}x{h}",
        })

    ranking_records.sort(key=lambda r: r["max_score"], reverse=True)
    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== FoundAD High-Res Tiling Escape Ranking Report ===\n")
        f.write(
            f"Tile Size: {manifest['tile_size']} | Overlap: {tile_overlap} | "
            f"Threshold (μ + {sigma_k}σ): {threshold:.6f} (μ={mu:.6f}, σ={sigma:.6f})\n"
        )
        f.write("-" * 95 + "\n")
        f.write(f"{'Rank':<6} | {'Filename':<32} | {'Status':<16} | {'Max_Score':<11} | {'BBoxes':<6} | {'Resolution'}\n")
        f.write("-" * 95 + "\n")
        for idx, r in enumerate(ranking_records, 1):
            f.write(
                f"{idx:<6} | {r['filename']:<32} | {r['status']:<16} | "
                f"{r['max_score']:<11.6f} | {r['num_bboxes']:<6} | {r['resolution']}\n"
            )

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Filename", "Status", "Max_Score", "TopK_Mean_Score", "Num_BBoxes", "Resolution"])
        for idx, r in enumerate(ranking_records, 1):
            writer.writerow([
                idx, r["filename"], r["status"],
                f"{r['max_score']:.6f}", f"{r['top_k_score']:.6f}",
                r["num_bboxes"], r["resolution"],
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="./output/2_fab_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot")
    parser.add_argument("--category", type=str, default="wafer")
    parser.add_argument("--app_train", type=str, default="train_dinov3")
    parser.add_argument("--app_test", type=str, default="test")
    parser.add_argument("--diy_name", type=str, default="sim_escape")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--sigma_k", type=float, default=3.2)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--mock_from_mvtec", type=str, default="")

    parser.add_argument("--tile_size", type=int, default=512, help="子圖塊像素大小 (建議 512 或 1024，避免大圖壓縮失真)")
    parser.add_argument("--tile_overlap", type=int, default=64, help="相鄰子圖塊重疊像素寬度 (用於無縫平滑拼接)")
    parser.add_argument("--max_train_tiles", type=int, default=16, help="從 Golden 大圖抽出的代表性高解析度訓練圖塊上限")
    parser.add_argument("--train_batch_size", type=int, default=8, help="訓練時的最大 Batch Size，防止高解析度圖塊 OOM")
    parser.add_argument("--bg_std_thresh", type=float, default=1.5, help="過濾晶圓圓周外純背景角落塊的標準差門檻")
    parser.add_argument("--disable_tiling", action="store_true", help="若設定此旗標則關閉切塊，退回整張縮圖模式")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)

    if args.mock_from_mvtec:
        prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, defect_limit=8)

    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    shot_dir = os.path.join(abs_datasets_dir, args.data_name)
    safe_rmtree(shot_dir)

    # Step 1: 自動挑 4 張純淨 Golden Image 並存放在 ./datasets/<data_name>/train/<category>/
    golden_paths = select_golden_images(
        pool_dir=args.pool_dir,
        ckpt_path=args.ckpt,
        out_dir=shot_dir,
        category=args.category,
        shots=args.shots,
        report_dir=args.out_dir,
    )
    golden_filenames = [os.path.basename(p) for p in golden_paths]

    # Step 1.5: 將高解析度 Golden 大圖切分為無失真子圖塊供訓練
    if not args.disable_tiling:
        num_train_samples = prepare_tiled_golden_trainset(
            shot_dir=shot_dir,
            category=args.category,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_train_tiles=args.max_train_tiles,
            bg_std_thresh=args.bg_std_thresh,
        )
    else:
        num_train_samples = args.shots

    effective_batch_size = min(max(1, num_train_samples), args.train_batch_size)

    # Step 2: 執行 FoundAD 少樣本訓練
    if not args.skip_train:
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={effective_batch_size} "
            f"data.dataset=mvtec data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={shot_dir} "
            f"data.mvtec_classnames='[{args.category}]' "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        print(f"\n[Step 2] 執行 FoundAD 高解析度圖塊訓練 (batch_size={effective_batch_size}):\n{train_cmd}")
        subprocess.run(train_cmd, shell=True, check=True)

    actual_step = prepare_ckpt_for_demo(args.data_name, args.diy_name, args.ckpt_step)

    # Step 3 & Step 4: 高解析度切塊推論 + 5000x5000 全解析度無縫縫合與畫紅框
    amap_dump_dir = os.path.join(args.out_dir, "amap_npy")

    if not args.disable_tiling:
        tiled_pool_dir = os.path.join(abs_datasets_dir, f"{args.data_name}_tiled_pool")
        tile_amap_dir = os.path.join(args.out_dir, "_tile_amap_tmp")
        tile_bbox_tmp = os.path.join(args.out_dir, "_tile_bbox_tmp")
        safe_rmtree(tile_amap_dir)
        safe_rmtree(tile_bbox_tmp)
        os.makedirs(tile_amap_dir, exist_ok=True)
        os.makedirs(tile_bbox_tmp, exist_ok=True)

        manifest = prepare_tiled_test_pool(
            pool_dir=args.pool_dir,
            tiled_pool_dir=tiled_pool_dir,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            bg_std_thresh=args.bg_std_thresh,
        )

        env = os.environ.copy()
        env["FAB_AMAP_DUMP_DIR"] = os.path.abspath(tile_amap_dir)
        env["FAB_BBOX_OUT_DIR"] = os.path.abspath(tile_bbox_tmp)
        env["FAB_SIGMA_K"] = str(args.sigma_k)
        env["FAB_GOLDEN_NAMES"] = ""

        demo_cmd = (
            f"python foundad/main.py mode=demo data.dataset=mvtec "
            f"data.data_name={args.data_name} diy_name={args.diy_name} "
            f"data.test_root={os.path.abspath(tiled_pool_dir)} app={args.app_test} app.ckpt_step={actual_step}"
        )
        print(f"\n[Step 3] 執行 FoundAD 子圖塊高解析度推論:\n{demo_cmd}")
        subprocess.run(demo_cmd, shell=True, env=env, check=True)

        stitch_and_visualize_full_res(
            manifest=manifest,
            tile_amap_dir=tile_amap_dir,
            final_amap_dir=amap_dump_dir,
            out_dir=os.path.abspath(args.out_dir),
            golden_filenames=golden_filenames,
            sigma_k=args.sigma_k,
        )

        safe_rmtree(tile_amap_dir)
        safe_rmtree(tile_bbox_tmp)
    else:
        env = os.environ.copy()
        env["FAB_AMAP_DUMP_DIR"] = amap_dump_dir
        env["FAB_BBOX_OUT_DIR"] = os.path.abspath(args.out_dir)
        env["FAB_SIGMA_K"] = str(args.sigma_k)
        env["FAB_GOLDEN_NAMES"] = ",".join(golden_filenames)

        demo_cmd = (
            f"python foundad/main.py mode=demo data.dataset=mvtec "
            f"data.data_name={args.data_name} diy_name={args.diy_name} "
            f"data.test_root={os.path.abspath(args.pool_dir)} app={args.app_test} app.ckpt_step={actual_step}"
        )
        print(f"\n[Step 3] 執行 FoundAD 無 GT 熱力圖推論與瑕疵畫框:\n{demo_cmd}")
        subprocess.run(demo_cmd, shell=True, env=env, check=True)

        for hm_src in glob.glob(f"./logs/{args.data_name}/**/heatmaps", recursive=True):
            if os.path.isdir(hm_src):
                hm_dst = os.path.join(args.out_dir, "heatmaps")
                safe_rmtree(hm_dst)
                shutil.copytree(hm_src, hm_dst, dirs_exist_ok=True)

    print(f"\n[✓] 無 GT 高解析度抓漏所有文字報表與紅框圖片已統一整合至: {args.out_dir}")


if __name__ == "__main__":
    main()