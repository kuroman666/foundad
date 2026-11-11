# 檔案位置: FoundAD/2_train_fab_no_gt_old.py (僅保留 Training 流程精簡版，原始檔案為 2_run_fab_no_gt.py(1105))
import shlex
import os
import re
import gc
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
            f.write(f"\n{'=' * 60}\n[CMD] {cmd} (PORT={run_env['FAB_MASTER_PORT']})\n{'=' * 60}\n")
            f.flush()

        process = subprocess.Popen(
            shlex.split(cmd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=run_env
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
    """遞迴計算影像池所有有效影像之 (相對路徑, 檔案大小) MD5 指紋，用於判斷資料池是否變動。"""
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


def verify_and_export_ckpt(
    data_name: str, diy_name: str, req_step: int, log_files: List[str]
) -> Tuple[int, List[str], List[str]]:
    """確認訓練權重已生成並複製為 pretrained.pth.tar，回傳 (步數, 權重資料夾清單, 權重檔案清單)。"""
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
    從 MVTec 建立模擬無 GT 盲池（供挑選 Golden Image 訓練使用）：
    - 正常樣本: clean_<原始檔名>
    - 瑕疵樣本: escape_<瑕疵類別>_<原始檔名>
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
        f"[模擬無GT盲池建立完成] 共 {total_count} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵) "
        f"-> {os.path.abspath(mock_pool_dir)}",
        log_files,
    )


# =====================================================================
# 高解析度訓練切塊 (Tiling) 模組
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
    將 Step 1 挑出的 Golden Image 切分為原生解析度子圖塊：
    - 對每個空間座標 (y, x) 計算跨片 L1 距離矩陣，優先選出最純淨的 Medoid 圖塊。
    - 當需要補充跨片公差圖塊時，自動排除距離最遠的離群圖塊 (Outlier Rejection)，防止局部微瑕疵混入訓練集。
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


def main():
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--skip_train", action="store_true", help="跳過模型訓練步驟")
    parser.add_argument("--mock_from_mvtec", type=str, default="")

    parser.add_argument("--tile_size", type=int, default=512, help="子圖塊像素大小 (建議 512 或 1024，避免大圖壓縮失真)")
    parser.add_argument("--tile_overlap", type=int, default=64, help="相鄰子圖塊重疊像素寬度")
    parser.add_argument("--max_train_tiles", type=int, default=48, help="從 Golden 大圖抽出的代表性高解析度訓練圖塊上限")
    parser.add_argument("--train_batch_size", type=int, default=8, help="訓練時的最大 Batch Size，防止高解析度圖塊 OOM")
    parser.add_argument("--bg_std_thresh", type=float, default=1.5, help="過濾純背景角落塊的標準差門檻")
    parser.add_argument("--disable_tiling", action="store_true", help="若設定此旗標則關閉切塊，退回整張訓練模式")
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

    start_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_and_print(
        f"\n{'=' * 75}\n"
        f"[Task 2 無GT訓練啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_out_dir}\n"
        f"{'=' * 75}",
        log_files,
    )

    if args.mock_from_mvtec:
        existing_pool_imgs = scan_pool_images(args.pool_dir) if os.path.isdir(args.pool_dir) else []
        if args.skip_train and existing_pool_imgs:
            log_and_print(
                f"[⚡ 模擬無GT盲池快取命中] 已設定 --skip_train，直接沿用既有 {len(existing_pool_imgs)} 張盲池圖片 "
                f"-> {os.path.abspath(args.pool_dir)}",
                log_files,
            )
        else:
            prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, log_files, defect_limit=8)

    pool_signature = compute_pool_signature(args.pool_dir)
    golden_meta_path = os.path.join(shot_dir, "golden_cache_meta.json")

    # =========================================================================
    # [整體訓練階段] (包含 Golden 自動挑圖 + 高解析度切塊 + 模型訓練)
    # =========================================================================
    used_golden_cache = False
    golden_filenames: List[str] = []
    num_train_samples = args.shots
    golden_select_sec = 0.0
    train_tile_sec = 0.0

    if args.skip_train and os.path.isfile(golden_meta_path):
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
    actual_step, ckpt_dirs, ckpt_files = verify_and_export_ckpt(
        args.data_name, args.diy_name, effective_epochs if not args.skip_train else args.ckpt_step, log_files
    )

    train_tiles_dir = os.path.abspath(os.path.join(shot_dir, "train", args.category))
    golden_backup_dir = os.path.abspath(os.path.join(shot_dir, "golden_full_res_backup", args.category))

    train_done_lines = [
        f"\n{'=' * 75}",
        f" [✅ 訓練階段完成 — 時間統計與權重檔儲存位置]",
        f"{'-' * 75}",
        f"  ⏱️ Step 1 自動挑選 Golden 圖耗時 : {format_duration(golden_select_sec)}{' (Cached)' if used_golden_cache else ''}",
        f"  ⏱️ Step 1.5 高解析度訓練切塊耗時 : {format_duration(train_tile_sec)}{' (Cached)' if used_golden_cache else ''}",
        f"  ⏱️ Step 2 FoundAD 模型訓練耗時   : {format_duration(model_train_sec)}{' (Skipped)' if args.skip_train else ''}",
        f"  ⏱️ 整體訓練階段總時間            : {format_duration(total_train_time)}",
        f"{'-' * 75}",
        f"  📁 Golden 訓練圖塊資料夾         : {train_tiles_dir}",
        f"  📁 Golden 原圖備份資料夾         : {golden_backup_dir}",
        f"  📦 訓練權重檔儲存資料夾          : {ckpt_dirs[0]}",
    ]
    for cf in ckpt_files:
        train_done_lines.append(f"      └── {cf}")
    train_done_lines.append(f"{'=' * 75}")
    log_and_print("\n".join(train_done_lines), log_files)


if __name__ == "__main__":
    main()