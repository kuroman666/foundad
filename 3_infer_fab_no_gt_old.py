# 檔案位置: FoundAD/3_infer_fab_no_gt_old.py (無損加速優化版：保留完整舊版評估與報表流程，加入 GPU 高Batch/TF32/AMP/多執行緒、1024px降維縫合濾波、ProcessPoolExecutor多核平行、無壓縮PNG與圓心快取)
# 背景執行全部 YDFID-2（19 個類別）： nohup ./run_outside_new_datasets.sh ./data/YDFID-2 > foundad_YDFID2.log 2>&1 &
#                                 tail -f foundad_YDFID2.log
# 執行 Mura / mini_AVMap2： nohup ./run_outside_new_datasets.sh ./data/mini_AVMap2 ./data/Mura2_wo_bg_w_pb ./data/Mura2_bg_mask > foundad_mura.log 2>&1 &
#                          tail -f foundad_mura.log
import importlib
import shlex
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# 避免超大解析度晶圓圖觸發 PIL DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

TZ_TAIPEI = timezone(timedelta(hours=8))

_ag = importlib.import_module("1_auto_golden_old")
IMG_EXTENSIONS = _ag.IMG_EXTENSIONS
scan_pool_images = _ag.scan_pool_images


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


def _ensure_offline_hub_patch_dir() -> str:
    """
    建立 sitecustomize.py 自動攔截：
    1. GPU 加速：自動啟用 TF32 與 cuDNN benchmark。
    2. torch.hub.load：優先從本地快取離線載入 DINOv3/DINOv2。
    3. DataLoader 多執行緒注入：讀取 FAB_NUM_WORKERS 自動啟用 num_workers 與 pin_memory=True。
    """
    patch_dir = os.path.abspath("./.offline_hub_patch")
    os.makedirs(patch_dir, exist_ok=True)
    sc_path = os.path.join(patch_dir, "sitecustomize.py")
    project_root = os.path.abspath(os.path.dirname(__file__))

    code_sc = (
        "import os\n"
        "import glob\n"
        "import inspect\n"
        f"PROJECT_ROOT = {repr(project_root)}\n"
        "try:\n"
        "    import torch\n"
        "    import torch.hub\n"
        "\n"
        "    if torch.cuda.is_available():\n"
        "        torch.backends.cuda.matmul.allow_tf32 = True\n"
        "        torch.backends.cudnn.allow_tf32 = True\n"
        "        torch.backends.cudnn.benchmark = True\n"
        "        try:\n"
        "            torch.set_float32_matmul_precision('high')\n"
        "        except Exception:\n"
        "            pass\n"
        "\n"
        "    _orig_hub_load = torch.hub.load\n"
        "\n"
        "    def _find_local_repo(repo_or_dir):\n"
        "        search_roots = [\n"
        "            os.path.join(PROJECT_ROOT, 'hub_cache'),\n"
        "            os.path.abspath('./hub_cache'),\n"
        "            torch.hub.get_dir(),\n"
        "            os.path.expanduser('~/.cache/torch/hub'),\n"
        "        ]\n"
        "        owner, repo = ('', '')\n"
        "        if isinstance(repo_or_dir, str) and '/' in repo_or_dir:\n"
        "            parts = repo_or_dir.split('/')\n"
        "            owner, repo = parts[0], parts[1].split(':')[0]\n"
        "        for root in search_roots:\n"
        "            if not os.path.isdir(root):\n"
        "                continue\n"
        "            if owner and repo:\n"
        "                for pat in [f'{owner}_{repo}_*', f'*{repo}*']:\n"
        "                    for c in sorted(glob.glob(os.path.join(root, pat))):\n"
        "                        if os.path.isdir(c) and os.path.isfile(os.path.join(c, 'hubconf.py')):\n"
        "                            return c\n"
        "            for c in sorted(glob.glob(os.path.join(root, '*dino*'))):\n"
        "                if os.path.isdir(c) and os.path.isfile(os.path.join(c, 'hubconf.py')):\n"
        "                    return c\n"
        "        return None\n"
        "\n"
        "    def _safe_hub_load(repo_or_dir, model, *args, **kwargs):\n"
        "        if kwargs.get('source', 'github') == 'github':\n"
        "            local_dir = _find_local_repo(repo_or_dir)\n"
        "            if local_dir is not None:\n"
        "                kw_local = dict(kwargs)\n"
        "                kw_local['source'] = 'local'\n"
        "                return _orig_hub_load(local_dir, model, *args, **kw_local)\n"
        "        try:\n"
        "            return _orig_hub_load(repo_or_dir, model, *args, **kwargs)\n"
        "        except Exception:\n"
        "            local_dir = _find_local_repo(repo_or_dir)\n"
        "            if local_dir is not None:\n"
        "                kw_local = dict(kwargs)\n"
        "                kw_local['source'] = 'local'\n"
        "                return _orig_hub_load(local_dir, model, *args, **kw_local)\n"
        "            raise\n"
        "\n"
        "    torch.hub.load = _safe_hub_load\n"
        "\n"
        "    _fab_nw = int(os.environ.get('FAB_NUM_WORKERS', '0') or '0')\n"
        "    if _fab_nw > 0:\n"
        "        import torch.utils.data\n"
        "        _orig_dl_init = torch.utils.data.DataLoader.__init__\n"
        "        _dl_sig = inspect.signature(_orig_dl_init)\n"
        "\n"
        "        def _patched_dl_init(self, *args, **kwargs):\n"
        "            try:\n"
        "                bound = _dl_sig.bind_partial(self, *args, **kwargs)\n"
        "                if bound.arguments.get('num_workers', 0) == 0:\n"
        "                    bound.arguments['num_workers'] = _fab_nw\n"
        "                if torch.cuda.is_available() and 'pin_memory' not in bound.arguments:\n"
        "                    bound.arguments['pin_memory'] = True\n"
        "                if bound.arguments.get('num_workers', 0) > 0 and 'persistent_workers' not in bound.arguments:\n"
        "                    bound.arguments['persistent_workers'] = True\n"
        "                return _orig_dl_init(*bound.args, **bound.kwargs)\n"
        "            except Exception:\n"
        "                return _orig_dl_init(self, *args, **kwargs)\n"
        "\n"
        "        torch.utils.data.DataLoader.__init__ = _patched_dl_init\n"
        "except Exception:\n"
        "    pass\n"
    )
    with open(sc_path, "w", encoding="utf-8") as f:
        f.write(code_sc)
    return patch_dir


def run_cmd_and_log(cmd: str, log_files: List[str], env: Optional[Dict] = None) -> str:
    """執行子程序指令，自動注入空閒 FAB_MASTER_PORT 與離線/加速 Patch，即時在 Terminal 顯示並同步寫入 log.txt。"""
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
    """計算實際載入之模型權重檔指紋 (路徑 + 檔案大小 + 修改時間)，確保權重更新後自動更新推論快取。"""
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


def _print_inline_progress(
    stage_title: str,
    current: int,
    total: int,
    start_t: float,
    extra_info: str = "",
    bar_len: int = 24,
) -> None:
    """在終端機單行即時更新視覺化進度條 (含百分比、處理速度與預估剩餘時間 ETA)。"""
    total = max(1, total)
    elapsed = max(1e-4, time.perf_counter() - start_t)
    rate = current / elapsed
    eta_sec = max(0.0, (total - current) / max(rate, 1e-6))
    pct = min(100.0, (current / total) * 100.0)
    filled = int(round(bar_len * current / total))
    bar = "█" * filled + "░" * (bar_len - filled)
    eta_str = format_duration(eta_sec) if current < total else "0.00秒"
    el_str = format_duration(elapsed)
    extra_str = f" | {extra_info}" if extra_info else ""
    msg = (
        f"\r⏳ [{stage_title}] |{bar}| {current}/{total} ({pct:5.1f}%) "
        f"[{el_str}<{eta_str}, {rate:5.2f}張/s]{extra_str}   "
    )
    print(msg, end="" if current < total else "\n", flush=True)


def prepare_ckpt_for_demo(
    data_name: str, diy_name: str, req_step: int, log_files: List[str]
) -> Tuple[int, List[str], List[str]]:
    """確認權重已存在並複製為 pretrained.pth.tar，支援遞迴搜尋與直接傳入路徑。"""
    search_roots = [
        f"./logs/{data_name}",
        data_name,
    ]
    step_files = []
    for root in search_roots:
        if os.path.isdir(root):
            step_files.extend(glob.glob(os.path.join(root, "**", "train-step*.pth.tar"), recursive=True))

    if diy_name and any(diy_name in f for f in step_files):
        step_files = [f for f in step_files if diy_name in f]

    ckpt_dirs = []
    ckpt_files = []
    chosen_step = req_step

    dir_map = {}
    for f in step_files:
        d = os.path.dirname(f)
        dir_map.setdefault(d, []).append(f)

    for d, files in dir_map.items():
        target_ckpt = os.path.join(d, f"train-step{req_step}.pth.tar")
        if not os.path.exists(target_ckpt):
            steps = [int(re.search(r"train-step(\d+)", f).group(1)) for f in files if re.search(r"train-step(\d+)", f)]
            if steps:
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
            f"在 ./logs/{data_name}/ (當前工作目錄: {os.getcwd()}) 下找不到任何 train-step*.pth.tar 權重檔！"
        )
    return chosen_step, ckpt_dirs, ckpt_files


def load_existing_golden_filenames(shot_dir: str, run_out_dir: str, log_files: List[str]) -> List[str]:
    """從既有快取或紀錄檔讀取 Golden Image 清單（供推論階段空間結構容差與過擬合診斷參考，無則回傳空陣列）。"""
    golden_meta_path = os.path.join(shot_dir, "golden_cache_meta.json")
    golden_filenames: List[str] = []

    if os.path.isfile(golden_meta_path):
        try:
            with open(golden_meta_path, "r", encoding="utf-8") as f:
                g_meta = json.load(f)
            if g_meta.get("golden_filenames"):
                golden_filenames = list(g_meta["golden_filenames"])
        except Exception:
            pass

    for rec_src in sorted(glob.glob(os.path.join(shot_dir, "selected_*_golden_images.txt"))):
        safe_copy(rec_src, os.path.join(run_out_dir, os.path.basename(rec_src)))
        if not golden_filenames:
            try:
                with open(rec_src, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("="):
                            golden_filenames.append(line.split()[0])
            except Exception:
                pass

    if golden_filenames:
        log_and_print(
            f"[Golden-Ref] 已載入既有 {len(golden_filenames)} 張 Golden Image 參考清單: {', '.join(golden_filenames)}",
            log_files,
        )
    else:
        log_and_print("[Golden-Ref] 未偵測到既有 Golden Image 紀錄，將以全盲池無參考模式進行推論。", log_files)
    return golden_filenames


def prepare_mock_unlabeled_pool_from_mvtec(
    mvtec_cat_dir: str, mock_pool_dir: str, log_files: List[str], defect_limit: int = 8
):
    """從 MVTec 建立模擬無 GT 盲池。"""
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
# 圓心偵測模組 (支援直接從記憶體 RGB 陣列偵測並快取，免去 Step 4 重複讀大圖)
# =====================================================================

def detect_circular_wafer_geometry_from_rgb(
    full_rgb: np.ndarray, wafer_edge_mode: str, category: str
) -> Tuple[bool, float, float, float]:
    """直接對已載入記憶體的 RGB 原圖偵測圓形晶圓幾何資訊，回傳 (是否啟用, cy, cx, radius)。"""
    h, w = full_rgb.shape[:2]
    mode = wafer_edge_mode.lower().strip()
    if mode == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5

    cat_lower = category.lower()
    is_circular_cat = any(k in cat_lower for k in ["wafer", "bottle", "cable", "coin", "lens", "circle"])

    scale = 256.0 / max(h, w)
    sh, sw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
    try:
        small_rgb = cv2.resize(full_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        corner_sz = max(4, min(sh, sw) // 10)
        corners = np.concatenate([
            blur[:corner_sz, :corner_sz].reshape(-1),
            blur[:corner_sz, -corner_sz:].reshape(-1),
            blur[-corner_sz:, :corner_sz].reshape(-1),
            blur[-corner_sz:, -corner_sz:].reshape(-1),
        ])
        center_patch = blur[sh // 3: 2 * sh // 3, sw // 3: 2 * sw // 3].reshape(-1)
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


def detect_circular_wafer_geometry(
    img_path: str, h: int, w: int, wafer_edge_mode: str, category: str
) -> Tuple[bool, float, float, float]:
    """容錯版：若 manifest 未快取 wafer_geom 時，才從磁碟讀取原圖偵測圓心。"""
    if wafer_edge_mode.lower().strip() == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5
    try:
        full_rgb = load_image_rgb_uint8(img_path)
        res = detect_circular_wafer_geometry_from_rgb(full_rgb, wafer_edge_mode, category)
        del full_rgb
        return res
    except Exception:
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5


# =====================================================================
# 高解析度 5000x5000 平行切塊 (無壓縮 PNG + 同步快取圓心) 與降維縫合模組
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


def _tile_single_image_worker(task_args: Tuple) -> Tuple[str, Dict, int, int]:
    (
        fpath,
        rel_fname,
        stem,
        tiled_pool_dir,
        tile_size,
        stride,
        bg_std_thresh,
        disable_tiling,
        wafer_edge_mode,
        category,
    ) = task_args
    cv2.setNumThreads(1)
    img = load_image_rgb_uint8(fpath)
    h, w = img.shape[:2]

    # 在原圖已載入記憶體時順便計算圓心座標並快取
    use_circle, cy_f, cx_f, r_f = detect_circular_wafer_geometry_from_rgb(
        img, wafer_edge_mode=wafer_edge_mode, category=category
    )

    if disable_tiling or (h <= tile_size and w <= tile_size):
        ys, xs = [0], [0]
        eff_tile_h, eff_tile_w = h, w
    else:
        ys = get_axis_positions(h, tile_size, stride)
        xs = get_axis_positions(w, tile_size, stride)
        eff_tile_h, eff_tile_w = tile_size, tile_size

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
            # 取消 zlib 壓縮 (compress_level=0)，大幅降低寫檔耗時
            Image.fromarray(crop).save(
                os.path.join(tiled_pool_dir, tile_fname), format="PNG", compress_level=0
            )
            tiles_info.append({
                "tile_stem": tile_stem,
                "tile_fname": tile_fname,
                "y": int(y),
                "x": int(x),
                "h": int(ch),
                "w": int(cw),
            })

    if not tiles_info:
        cy, cx = max(0, (h - eff_tile_h) // 2), max(0, (w - eff_tile_w) // 2)
        ch, cw = min(eff_tile_h, h - cy), min(eff_tile_w, w - cx)
        crop = img[cy:cy + ch, cx:cx + cw]
        tile_stem = f"{stem}__tile_y{cy:04d}_x{cx:04d}_h{ch:04d}_w{cw:04d}"
        tile_fname = f"{tile_stem}.png"
        Image.fromarray(crop).save(
            os.path.join(tiled_pool_dir, tile_fname), format="PNG", compress_level=0
        )
        tiles_info.append({
            "tile_stem": tile_stem,
            "tile_fname": tile_fname,
            "y": int(cy),
            "x": int(cx),
            "h": int(ch),
            "w": int(cw),
        })

    del img
    entry = {
        "stem": stem,
        "path": os.path.abspath(fpath),
        "height": int(h),
        "width": int(w),
        "wafer_geom": [bool(use_circle), float(cy_f), float(cx_f), float(r_f)],
        "tiles": tiles_info,
    }
    return rel_fname, entry, len(tiles_info), skipped_bg


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
    wafer_edge_mode: str = "auto",
    category: str = "wafer",
    num_workers: int = 8,
) -> Dict:
    """
    遞迴搜尋測試盲池 (含所有子資料夾) 中的所有圖片並以多核心平行切分為重疊子圖塊，
    使用無壓縮 PNG 存檔並同步快取晶圓圓心座標。
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

    tasks = []
    for fpath in img_paths:
        rel_fname = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_fname)[0]
        stem = re.sub(r"[\\/]+", "__", rel_no_ext)
        tasks.append((
            fpath,
            rel_fname,
            stem,
            tiled_pool_dir,
            tile_size,
            stride,
            bg_std_thresh,
            disable_tiling,
            wafer_edge_mode,
            category,
        ))

    num_pool_imgs = len(tasks)
    eff_workers = max(1, min(int(num_workers), num_pool_imgs))
    t_tile_loop_start = time.perf_counter()
    print(
        f"\n[Tiling-Test] 開始準備 {num_pool_imgs} 張測試原圖之推論圖塊 "
        f"(Workers={eff_workers}, 無壓縮 PNG + 同步快取圓心)...",
        flush=True,
    )

    ordered_results: Dict[str, Dict] = {}
    if eff_workers <= 1:
        for img_idx, t_args in enumerate(tasks, 1):
            rel_fname, entry, n_t, n_skip = _tile_single_image_worker(t_args)
            ordered_results[rel_fname] = entry
            total_tiles += n_t
            skipped_bg_tiles += n_skip
            if img_idx == 1 or img_idx % 5 == 0 or img_idx == num_pool_imgs:
                _print_inline_progress(
                    stage_title="Step 3.1 測試池切塊",
                    current=img_idx,
                    total=num_pool_imgs,
                    start_t=t_tile_loop_start,
                    extra_info=f"累計圖塊: {total_tiles} (略過背景: {skipped_bg_tiles})",
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=eff_workers) as executor:
            futures = [executor.submit(_tile_single_image_worker, t_args) for t_args in tasks]
            for fut in as_completed(futures):
                rel_fname, entry, n_t, n_skip = fut.result()
                ordered_results[rel_fname] = entry
                total_tiles += n_t
                skipped_bg_tiles += n_skip
                completed += 1
                if completed == 1 or completed % 5 == 0 or completed == num_pool_imgs:
                    _print_inline_progress(
                        stage_title=f"Step 3.1 平行切塊(x{eff_workers})",
                        current=completed,
                        total=num_pool_imgs,
                        start_t=t_tile_loop_start,
                        extra_info=f"累計圖塊: {total_tiles} (略過背景: {skipped_bg_tiles})",
                    )

    for t_args in tasks:
        rf = t_args[1]
        if rf in ordered_results:
            manifest["images"][rf] = ordered_results[rf]

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
        f"(自動跳過 {skipped_bg_tiles} 張純背景角落塊, Workers={eff_workers}) -> {os.path.abspath(tiled_pool_dir)}",
        log_files,
    )
    return manifest


def build_2d_feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """建立 2D 餘弦平滑權重遮罩 (Raised-Cosine Window)，消除重疊切塊拼接時的邊界接縫。"""
    def _1d_ramp(length: int, ov: int) -> np.ndarray:
        w1d = np.ones(length, dtype=np.float32)
        ramp = min(max(ov, 4), max(1, length // 3))
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


def compute_stitch_canvas_shape(orig_h: int, orig_w: int, stitch_max_dim: int) -> Tuple[int, int, float, float]:
    """計算降維縫合畫布尺寸 (sh, sw) 與縮放比例 (scale_y, scale_x)。"""
    if stitch_max_dim > 0 and max(orig_h, orig_w) > stitch_max_dim:
        s = stitch_max_dim / float(max(orig_h, orig_w))
        sh = max(1, int(round(orig_h * s)))
        sw = max(1, int(round(orig_w * s)))
        return sh, sw, sh / float(orig_h), sw / float(orig_w)
    return orig_h, orig_w, 1.0, 1.0


def stitch_single_image_amap(
    img_meta: Dict,
    tile_amap_dir: str,
    tile_overlap: int,
    stitch_max_dim: int = 1024,
    delete_tiles: bool = True,
) -> np.ndarray:
    """
    將單張大圖的所有子圖塊 .npy 無縫縫合至目標畫布 (預設最大邊 1024px，符合 ViT Patch 解析度，
    相比 5000x5000 運算量與記憶體直接降低 ~25 倍)，並在縫合後立即刪除子圖塊 .npy。
    """
    orig_h, orig_w = img_meta["height"], img_meta["width"]
    sh, sw, scale_y, scale_x = compute_stitch_canvas_shape(orig_h, orig_w, stitch_max_dim)
    tiles = img_meta["tiles"]

    canvas_sum = np.zeros((sh, sw), dtype=np.float32)
    weight_sum = np.zeros((sh, sw), dtype=np.float32)

    weight_cache: Dict[Tuple[int, int], np.ndarray] = {}
    consumed_tile_npys: List[str] = []
    is_single_full_tile = len(tiles) == 1 and tiles[0]["h"] == orig_h and tiles[0]["w"] == orig_w
    scaled_overlap = max(4, int(round(tile_overlap * min(scale_y, scale_x))))

    for t in tiles:
        npy_path = find_tile_npy(tile_amap_dir, t["tile_stem"], t["tile_fname"])
        if not npy_path:
            continue
        amap = np.load(npy_path).astype(np.float32)
        consumed_tile_npys.append(npy_path)
        amap = np.squeeze(amap)
        if amap.ndim != 2:
            continue

        y, x, th, tw = t["y"], t["x"], t["h"], t["w"]
        sy1 = int(round(y * scale_y))
        sx1 = int(round(x * scale_x))
        sy2 = min(sh, max(sy1 + 1, int(round((y + th) * scale_y))))
        sx2 = min(sw, max(sx1 + 1, int(round((x + tw) * scale_x))))
        sth, stw = sy2 - sy1, sx2 - sx1

        if amap.shape[0] != sth or amap.shape[1] != stw:
            amap = cv2.resize(amap, (stw, sth), interpolation=cv2.INTER_LINEAR)

        if is_single_full_tile:
            w2d = np.ones((sth, stw), dtype=np.float32)
        else:
            key = (sth, stw)
            if key not in weight_cache:
                weight_cache[key] = build_2d_feather_weight(sth, stw, scaled_overlap)
            w2d = weight_cache[key]

        canvas_sum[sy1:sy2, sx1:sx2] += amap * w2d
        weight_sum[sy1:sy2, sx1:sx2] += w2d

    valid_mask = weight_sum > 1e-6
    stitched_amap = np.zeros((sh, sw), dtype=np.float32)
    stitched_amap[valid_mask] = canvas_sum[valid_mask] / weight_sum[valid_mask]

    if np.any(valid_mask) and not np.all(valid_mask):
        baseline = float(np.percentile(stitched_amap[valid_mask], 5.0))
        stitched_amap[~valid_mask] = baseline

    if delete_tiles:
        for p in consumed_tile_npys:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    return stitched_amap


# =====================================================================
# 防誤報核心演算法：晶圓圓周邊緣抑制 + 微塵單點尖峰濾除 + 動態門檻對齊
# =====================================================================

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
    cached_wafer_geom: Optional[List] = None,
    orig_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    四合一防誤報熱力圖純化器 (支援於 1024x1024 降維畫布執行與圓心座標快取)：
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

    orig_h, orig_w = orig_hw if orig_hw is not None else (h, w)
    scale_y = h / float(max(1, orig_h))
    scale_x = w / float(max(1, orig_w))
    if wafer_edge_mode.lower().strip() == "off":
        use_circle, cy, cx, radius = False, h * 0.5, w * 0.5, min(h, w) * 0.5
    elif cached_wafer_geom is not None and len(cached_wafer_geom) == 4:
        use_circle = bool(cached_wafer_geom[0])
        cy = float(cached_wafer_geom[1]) * scale_y
        cx = float(cached_wafer_geom[2]) * scale_x
        radius = float(cached_wafer_geom[3]) * min(scale_y, scale_x)
    else:
        use_circle, cy_f, cx_f, r_f = detect_circular_wafer_geometry(
            img_path=img_path, h=orig_h, w=orig_w, wafer_edge_mode=wafer_edge_mode, category=category
        )
        cy = cy_f * scale_y
        cx = cx_f * scale_x
        radius = r_f * min(scale_y, scale_x)

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
    """峰值自適應核心收斂 (Peak-Adaptive Core Thresholding) 演算法。"""
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
    min_area = 1 if raw_score_mode else max(4, int(round((h * w) * min_area_ratio)))
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
    thresh_mode: str = "auto",
    user_threshold: float = 0.0,
    min_threshold: float = 0.0,
    auto_percentile: float = 95.0,
    sigma_k: float = 3.5,
    top_k: int = 0,
) -> Tuple[float, str]:
    """雙模式門檻判定：Fixed 固定門檻或 Min(3-Sigma, Modified Z-Score)。"""
    scores_arr = np.array(
        non_golden_scores if non_golden_scores else golden_scores,
        dtype=np.float32,
    )
    if scores_arr.size == 0:
        fallback = float(user_threshold) if user_threshold > 0 else float(min_threshold)
        return fallback, f"無樣本保底門檻 ({fallback:.4f})"

    total_n = int(scores_arr.size)

    mean_val = float(np.mean(scores_arr))
    std_val = float(np.std(scores_arr))
    thresh_3sigma = mean_val + 3.0 * std_val

    med_val = float(np.median(scores_arr))
    mad_val = float(np.median(np.abs(scores_arr - med_val)))
    thresh_mod_z = med_val + (3.5 / 0.6745) * mad_val

    if user_threshold is not None and float(user_threshold) > 0.0:
        eff_thresh = float(user_threshold)
        eff_count = int(np.sum(scores_arr >= eff_thresh))
        eff_pct = (eff_count / total_n) * 100.0
        desc = (
            f"Fixed 固定門檻 (沿用訓練集設定) = {eff_thresh:.4f} "
            f"(本時期切出 {eff_count}/{total_n} 張, {eff_pct:.2f}%)\n"
            f"           ├─ [本時期飄移參考] 3-Sigma (Shewhart 1931)         : {thresh_3sigma:.4f} (Mean={mean_val:.4f}, Std={std_val:.4f})\n"
            f"           └─ [本時期飄移參考] Modified Z-Score (Iglewicz 1993): {thresh_mod_z:.4f} (Median={med_val:.4f}, MAD={mad_val:.4f})"
        )
        return eff_thresh, desc

    if thresh_3sigma <= thresh_mod_z:
        eff_thresh, eff_name = thresh_3sigma, "3-Sigma (mean+3*std)"
        high_thresh, high_name = thresh_mod_z, "Modified Z-Score (3.5)"
    else:
        eff_thresh, eff_name = thresh_mod_z, "Modified Z-Score (3.5)"
        high_thresh, high_name = thresh_3sigma, "3-Sigma (mean+3*std)"

    eff_count = int(np.sum(scores_arr >= eff_thresh))
    eff_pct = (eff_count / total_n) * 100.0

    high_count = int(np.sum(scores_arr >= high_thresh))
    high_pct = (high_count / total_n) * 100.0
    high_percentile = float(np.mean(scores_arr < high_thresh) * 100.0)

    if high_count == 0:
        high_cut_info = f"高於全體最高分 ({float(np.max(scores_arr)):.4f})，切出 0 張 (0.00%)"
    else:
        high_cut_info = (
            f"切在 Rank {high_count}/{total_n} 之後 "
            f"(共 {high_count} 張 >= 該門檻, 佔 {high_pct:.2f}%, 約 P{high_percentile:.2f})"
        )

    desc = (
        f"Min(3-Sigma, Mod-Z) 採用 [{eff_name}] = {eff_thresh:.4f} "
        f"(切出 {eff_count}/{total_n} 張, {eff_pct:.2f}%)\n"
        f"           ├─ 3-Sigma (Shewhart 1931)         : {thresh_3sigma:.4f} (Mean={mean_val:.4f}, Std={std_val:.4f})\n"
        f"           ├─ Modified Z-Score (Iglewicz 1993): {thresh_mod_z:.4f} (Median={med_val:.4f}, MAD={mad_val:.4f})\n"
        f"           └─ 較高門檻 [{high_name}] = {high_thresh:.4f} -> {high_cut_info}"
    )
    return float(eff_thresh), desc


def evaluate_overfitting_diagnostics(
    golden_scores: List[float],
    non_golden_normal_scores: List[float],
) -> Tuple[float, float, float, str, str]:
    """無監督過擬合自檢指標。"""
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
            f"                     可直接調高門檻倍數 (例如設定 --thresh_mode mean_std --sigma_k 3.5) 或調降 --ckpt_step。"
        )
    elif overfit_ratio >= 1.25 and golden_gap >= 0.025:
        status_badge = "🟡 注意: 輕微過擬合傾向 (Mild Overfit — 已由動態門檻自動補償)"
        advice_msg = ""
    else:
        status_badge = "🟢 良好: 無過擬合現象 (Healthy Generalization — Golden與盲池正常分佈一致)"
        advice_msg = ""

    return g_mean, overfit_ratio, golden_gap, status_badge, advice_msg


def compute_blind_pool_eval_metrics(ranking_records: List[Dict]) -> Optional[Dict[str, float]]:
    """若盲池檔名包含模擬標籤 (escape_*)，自動計算 I-AUROC、I-AP、F1-max、Recall、FPR、Precision、F1。"""
    eval_recs = [r for r in ranking_records if not str(r.get("status", "")).startswith("GOLDEN")]
    if not eval_recs:
        eval_recs = ranking_records

    escape_recs = [r for r in eval_recs if os.path.basename(r["filename"]).startswith("escape_")]
    normal_recs = [r for r in eval_recs if not os.path.basename(r["filename"]).startswith("escape_")]
    if not escape_recs or not normal_recs:
        return None

    y_true = np.array(
        [1 if os.path.basename(r["filename"]).startswith("escape_") else 0 for r in eval_recs],
        dtype=np.int32,
    )
    y_score = np.array([float(r["max_score"]) for r in eval_recs], dtype=np.float64)
    y_pred = np.array([1 if "ANOMALY" in str(r["status"]) else 0 for r in eval_recs], dtype=np.int32)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    ranks = np.zeros(len(y_score_sorted), dtype=np.float64)
    i = 0
    n = len(y_score_sorted)
    while i < n:
        j = i + 1
        while j < n and y_score_sorted[j] == y_score_sorted[i]:
            j += 1
        avg_rank = 0.5 * ((n - j + 1) + (n - i))
        ranks[i:j] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[y_true_sorted == 1]))
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))

    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    thresh_idxs = np.r_[distinct_idx, n - 1]

    tps = tp_cum[thresh_idxs]
    fps = fp_cum[thresh_idxs]
    precisions = tps / np.maximum(1, tps + fps)
    recalls = tps / max(1, n_pos)

    rec_diff = np.diff(np.r_[0.0, recalls])
    ap = float(np.sum(rec_diff * precisions))

    f1_curve = (2.0 * precisions * recalls) / np.maximum(1e-12, precisions + recalls)
    best_idx = int(np.argmax(f1_curve))
    f1_max = float(f1_curve[best_idx])
    best_thresh = float(y_score_sorted[thresh_idxs[best_idx]])

    tp_cur = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_cur = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_cur = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall_cur = tp_cur / max(1, n_pos)
    fpr_cur = fp_cur / max(1, n_neg)
    prec_cur = tp_cur / max(1, tp_cur + fp_cur)
    f1_cur = (2.0 * prec_cur * recall_cur) / max(1e-12, prec_cur + recall_cur)

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": auroc,
        "ap": ap,
        "f1_max": f1_max,
        "best_thresh": best_thresh,
        "tp": tp_cur,
        "fp": fp_cur,
        "fn": fn_cur,
        "recall": recall_cur,
        "fpr": fpr_cur,
        "precision": prec_cur,
        "f1_cur": f1_cur,
    }


# =====================================================================
# Step 4.1 Pass 2 多核心平行縫合與濾波 Worker
# =====================================================================

_WORKER_GOLDEN_REF: Optional[np.ndarray] = None


def _init_stitch_worker(golden_ref_path: str):
    global _WORKER_GOLDEN_REF
    cv2.setNumThreads(1)
    if golden_ref_path and os.path.isfile(golden_ref_path):
        _WORKER_GOLDEN_REF = np.load(golden_ref_path).astype(np.float32)
    else:
        _WORKER_GOLDEN_REF = None


def _stitch_and_refine_single_worker(task: Dict) -> Dict:
    cv2.setNumThreads(1)
    idx = task["idx"]
    fname = task["fname"]
    meta = task["meta"]
    is_golden_img = task["is_golden_img"]
    pre_stitched_raw_path = task["pre_stitched_raw_path"]
    tile_amap_dir = task["tile_amap_dir"]
    tile_overlap = task["tile_overlap"]
    stitch_max_dim = task["stitch_max_dim"]
    refined_cache_dir = task["refined_cache_dir"]
    category = task["category"]
    border_ratio = task["border_ratio"]
    wafer_edge_mode = task["wafer_edge_mode"]
    wafer_edge_margin = task["wafer_edge_margin"]
    speckle_filter_ratio = task["speckle_filter_ratio"]
    raw_score_mode = task["raw_score_mode"]

    if pre_stitched_raw_path and os.path.isfile(pre_stitched_raw_path):
        raw_amap = np.load(pre_stitched_raw_path).astype(np.float32)
        try:
            os.remove(pre_stitched_raw_path)
        except OSError:
            pass
    else:
        raw_amap = stitch_single_image_amap(
            img_meta=meta,
            tile_amap_dir=tile_amap_dir,
            tile_overlap=tile_overlap,
            stitch_max_dim=stitch_max_dim,
            delete_tiles=True,
        )

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
        golden_spatial_ref=_WORKER_GOLDEN_REF if not is_golden_img else None,
        golden_tol_weight=0.35,
        cached_wafer_geom=meta.get("wafer_geom", None),
        orig_hw=(meta["height"], meta["width"]),
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

    cache_npy_path = os.path.join(refined_cache_dir, f"{idx:05d}_{meta['stem']}.npy")
    try:
        np.save(cache_npy_path, active_amap.astype(np.float16))
    except Exception:
        cache_npy_path = ""
    del raw_amap, refined_amap, active_amap

    return {
        "idx": idx,
        "filename": fname,
        "stem": meta["stem"],
        "meta": meta,
        "is_golden_img": is_golden_img,
        "cache_npy_path": cache_npy_path,
        "max_score": active_max_score,
        "raw_max_score": raw_single_max,
        "raw_top_peak_score": raw_top_peak,
        "filtered_max_score": filtered_robust_max,
        "filter_delta": filtered_robust_max - raw_single_max,
        "top_k_score": top_k_score,
    }


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
    stitch_max_dim: int = 1024,
    num_workers: int = 8,
) -> Tuple[int, int, float, str, str, str]:
    """三階段多核心平行縫合與篩選 (支援降維畫布 <= 1024px 與異常圖 1:1 座標映回)。"""
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
    canvas_desc = f"降維畫布 <= {stitch_max_dim}px" if stitch_max_dim > 0 else "原圖全解析度"

    log_and_print(
        f"\n[Step 4.1] 啟動 RAM 多核心平行縫合與碎片銷毀模式 (共 {total_imgs} 張原圖 | "
        f"畫布={canvas_desc} | Workers={num_workers}) | 計分模式: {score_mode_label}",
        log_files,
    )

    # Pass 1: 縫合 Golden 圖建立空間結構容差基準
    golden_pre_stitched_paths: Dict[str, str] = {}
    golden_maps: List[np.ndarray] = []
    if not raw_score_mode and golden_set:
        for g_idx, (fname, meta) in enumerate(manifest["images"].items()):
            if fname in golden_set or os.path.basename(fname) in golden_set:
                g_amap = stitch_single_image_amap(
                    meta, tile_amap_dir, tile_overlap, stitch_max_dim=stitch_max_dim, delete_tiles=True
                )
                tmp_g_path = os.path.join(refined_cache_dir, f"_golden_raw_{g_idx:03d}.npy")
                np.save(tmp_g_path, g_amap.astype(np.float32))
                golden_pre_stitched_paths[fname] = tmp_g_path
                golden_maps.append(g_amap)

    golden_ref_file = ""
    if golden_maps:
        ref_h, ref_w = golden_maps[0].shape[:2]
        aligned_g = [
            cv2.resize(gm, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR) if gm.shape[:2] != (ref_h, ref_w) else gm
            for gm in golden_maps
        ]
        g_max_map = np.max(np.stack(aligned_g, axis=0), axis=0)
        blur_k = max(15, int(round(max(ref_h, ref_w) * 0.035)) | 1)
        golden_spatial_ref = cv2.GaussianBlur(g_max_map, (blur_k, blur_k), 0)
        golden_ref_file = os.path.join(refined_cache_dir, "_golden_spatial_ref.npy")
        np.save(golden_ref_file, golden_spatial_ref.astype(np.float32))
        del aligned_g, g_max_map, golden_spatial_ref
    del golden_maps
    gc.collect()

    # Pass 2: 多核心平行縫合與四合一濾波
    tasks = []
    for idx, (fname, meta) in enumerate(manifest["images"].items(), 1):
        is_golden_img = (fname in golden_set) or (os.path.basename(fname) in golden_set)
        tasks.append({
            "idx": idx,
            "fname": fname,
            "meta": meta,
            "is_golden_img": is_golden_img,
            "pre_stitched_raw_path": golden_pre_stitched_paths.get(fname, ""),
            "tile_amap_dir": tile_amap_dir,
            "tile_overlap": tile_overlap,
            "stitch_max_dim": stitch_max_dim,
            "refined_cache_dir": refined_cache_dir,
            "category": category,
            "border_ratio": border_ratio,
            "wafer_edge_mode": wafer_edge_mode,
            "wafer_edge_margin": wafer_edge_margin,
            "speckle_filter_ratio": speckle_filter_ratio,
            "raw_score_mode": raw_score_mode,
        })

    stitched_info = []
    golden_max_scores = []
    non_golden_all_scores = []
    non_golden_normal_scores = []

    eff_workers = max(1, min(int(num_workers), len(tasks)))
    t_stitch_loop_start = time.perf_counter()

    if eff_workers <= 1:
        _init_stitch_worker(golden_ref_file)
        for completed, t_item in enumerate(tasks, 1):
            res = _stitch_and_refine_single_worker(t_item)
            if res["is_golden_img"]:
                golden_max_scores.append(res["max_score"])
            else:
                non_golden_all_scores.append(res["max_score"])
                if not os.path.basename(res["filename"]).startswith("escape_"):
                    non_golden_normal_scores.append(res["max_score"])
            stitched_info.append(res)
            if completed == 1 or completed % 5 == 0 or completed == total_imgs:
                _print_inline_progress(
                    stage_title="Step 4.1 RAM縫合與濾波",
                    current=completed,
                    total=total_imgs,
                    start_t=t_stitch_loop_start,
                    extra_info=f"最新分數={res['max_score']:.4f}",
                )
    else:
        completed = 0
        with ProcessPoolExecutor(
            max_workers=eff_workers,
            initializer=_init_stitch_worker,
            initargs=(golden_ref_file,),
        ) as executor:
            futures = [executor.submit(_stitch_and_refine_single_worker, t_item) for t_item in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res["is_golden_img"]:
                    golden_max_scores.append(res["max_score"])
                else:
                    non_golden_all_scores.append(res["max_score"])
                    if not os.path.basename(res["filename"]).startswith("escape_"):
                        non_golden_normal_scores.append(res["max_score"])
                stitched_info.append(res)
                completed += 1
                if completed == 1 or completed % 5 == 0 or completed == total_imgs:
                    _print_inline_progress(
                        stage_title=f"Step 4.1 平行縫合濾波(x{eff_workers})",
                        current=completed,
                        total=total_imgs,
                        start_t=t_stitch_loop_start,
                        extra_info=f"最新分數={res['max_score']:.4f}",
                    )

    if golden_ref_file and os.path.exists(golden_ref_file):
        try:
            os.remove(golden_ref_file)
        except OSError:
            pass

    safe_rmtree(tile_amap_dir)
    log_and_print(
        f"[🧹 碎片清理完成] 所有子圖塊已於 RAM 內拼回整張 Wafer，已自動銷毀全部碎片 Tile .npy -> {os.path.abspath(tile_amap_dir)}",
        log_files,
    )

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
        f"           (計分模式={'RAW未濾波' if raw_score_mode else '四合一純化濾波'}, 縫合畫布={canvas_desc}, "
        f"圓周邊緣抑制={'OFF(Raw)' if raw_score_mode else wafer_edge_mode}, "
        f"邊界縮減={0.0 if raw_score_mode else wafer_edge_margin * 100:.1f}%, "
        f"微塵開運算比例={0.0 if raw_score_mode else speckle_filter_ratio:.4f})\n"
        f"[Step 4.2b] 過擬合自檢診斷 : {overfit_badge} "
        f"[Golden平均={g_mean_score:.4f} | 非Golden正常中位數={float(np.median(ref_normal_scores)) if ref_normal_scores else 0.0:.4f} | "
        f"Overfit_Ratio={overfit_ratio:.2f}x (Gap={golden_gap:+.4f})]",
        log_files,
    )
    if overfit_advice:
        log_and_print(overfit_advice, log_files)

    # Pass 3: 依分數高低排序，僅針對超過門檻的圖片載入陣列並映回原圖畫紅框
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
            sh, sw, _, _ = compute_stitch_canvas_shape(h, w, stitch_max_dim)
            active_amap = np.zeros((sh, sw), dtype=np.float32)

        amap_h, amap_w = active_amap.shape[:2]
        bboxes_scaled, _ = extract_tight_bboxes_for_anomaly_image(
            refined_amap=active_amap,
            img_threshold=effective_threshold,
            core_ratio=core_ratio,
            min_area_ratio=min_area_ratio,
            raw_score_mode=raw_score_mode,
        )

        if not bboxes_scaled:
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

        # 將降維畫布紅框座標映回 1:1 原圖解析度
        inv_sx = w / float(max(1, amap_w))
        inv_sy = h / float(max(1, amap_h))
        bboxes_full = []
        for (x1, y1, x2, y2, region_max, area_s) in bboxes_scaled:
            fx1 = max(0, min(w - 1, int(round(x1 * inv_sx))))
            fy1 = max(0, min(h - 1, int(round(y1 * inv_sy))))
            fx2 = max(0, min(w - 1, int(round(x2 * inv_sx))))
            fy2 = max(0, min(h - 1, int(round(y2 * inv_sy))))
            farea = int(round(area_s * inv_sx * inv_sy))
            bboxes_full.append((fx1, fy1, fx2, fy2, region_max, farea))

        anomaly_count += 1

        save_dtype = np.float16 if max(active_amap.shape) > 1024 else np.float32
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
        for (x1, y1, x2, y2, region_max, _) in bboxes_full:
            cv2.rectangle(bbox_canvas, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            label_txt = f"{region_max:.3f}"
            (tw, th_txt), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            ty1 = max(0, y1 - th_txt - 8)
            cv2.rectangle(bbox_canvas, (x1, ty1), (x1 + tw + 6, ty1 + th_txt + 6), (0, 0, 255), -1)
            cv2.putText(
                bbox_canvas, label_txt, (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        safe_out_fname = re.sub(r"[\\/]+", "__", fname)
        prefixed_img_name = f"Rank{rank_idx:03d}_Score{max_score:.4f}_Box{len(bboxes_full):02d}_{safe_out_fname}"

        fullres_out_path = os.path.join(anomaly_fullres_bbox_dir, prefixed_img_name)
        cv2.imwrite(fullres_out_path, bbox_canvas)

        # 三合一對照圖直接於 panel_h 尺度合成熱力圖疊圖
        panel_h = min(900, h)
        panel_w = int(round(w * (panel_h / float(h))))
        p1 = cv2.resize(orig_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(bbox_canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        amap_panel = cv2.resize(active_amap, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        img_med = float(np.median(amap_panel))
        norm_map = np.clip((amap_panel - img_med) / max(float(np.max(amap_panel)) - img_med, 1e-6), 0.0, 1.0)
        hm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        p2 = cv2.addWeighted(p1, 0.55, hm_color, 0.45, 0)
        tri_panel = np.hstack([p1, p2, p3])

        banner_h = max(36, int(round(panel_h * 0.055)))
        banner = np.zeros((banner_h, tri_panel.shape[1], 3), dtype=np.uint8)
        mode_tag = "RAW" if raw_score_mode else "FILT"
        banner_txt = (
            f"[{mode_tag}] Orig: {fname} | Score: {max_score:.4f} (Raw: {raw_max_score:.4f}, Filt: {filtered_max_score:.4f}, "
            f"EffThresh: {effective_threshold:.4f}) | Status: ANOMALY (Boxes: {len(bboxes_full)})"
        )
        cv2.putText(
            banner, banner_txt, (12, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, banner_h / 55.0), (0, 220, 255), 2, cv2.LINE_AA
        )
        final_vis = np.vstack([banner, tri_panel])
        vis_out_path = os.path.join(anomaly_dir, prefixed_img_name)
        cv2.imwrite(vis_out_path, final_vis)

        del active_amap, orig_bgr, bbox_canvas, amap_panel, hm_color, p1, p2, p3, tri_panel, final_vis
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
            "num_bboxes": len(bboxes_full),
            "saved_file": prefixed_img_name,
            "resolution": f"{w}x{h}",
        })

    safe_rmtree(refined_cache_dir)
    gc.collect()

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

    eval_metrics = compute_blind_pool_eval_metrics(ranking_records)
    metrics_summary_line = ""
    metrics_report_header = ""
    if eval_metrics is not None:
        metrics_summary_line = (
            f"      🏆 模擬盲池量化指標 (Eval)     : "
            f"I-AUROC={eval_metrics['auroc']:.4f} | I-AP={eval_metrics['ap']:.4f} | "
            f"F1-max={eval_metrics['f1_max']:.4f} (@{eval_metrics['best_thresh']:.4f})\n"
            f"      🎯 當前門檻實測表現 (Current)    : "
            f"Recall={eval_metrics['recall']*100:.2f}% ({eval_metrics['tp']}/{eval_metrics['n_pos']}) | "
            f"FPR={eval_metrics['fpr']*100:.2f}% ({eval_metrics['fp']}/{eval_metrics['n_neg']}) | "
            f"Precision={eval_metrics['precision']*100:.2f}% | F1={eval_metrics['f1_cur']*100:.2f}%\n"
        )
        metrics_report_header = (
            f"Benchmark Metrics (No-GT Sim) : I-AUROC={eval_metrics['auroc']:.4f} | I-AP={eval_metrics['ap']:.4f} | "
            f"F1-max={eval_metrics['f1_max']:.4f} (Best_Th={eval_metrics['best_thresh']:.4f})\n"
            f"Current Threshold Performance : Recall={eval_metrics['recall']*100:.2f}% ({eval_metrics['tp']}/{eval_metrics['n_pos']}) | "
            f"FPR={eval_metrics['fpr']*100:.2f}% ({eval_metrics['fp']}/{eval_metrics['n_neg']}) | "
            f"Precision={eval_metrics['precision']*100:.2f}% | F1={eval_metrics['f1_cur']*100:.2f}%\n"
        )

    dist_summary_lines = (
        f"{metrics_summary_line}"
        f"      📊 非 Golden 正常圖分數分佈 ({'RAW未濾波' if raw_score_mode else '四合一濾波後'}) : "
        f"Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
        f"      📐 非 Golden 正常圖門檻參考值  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
        f"      🔬 濾波前後正常圖對照 (Ablation): "
        f"[未濾波 Raw] Max={raw_max_s:.4f}, Mean={raw_mean_s:.4f}, Std={raw_std_s:.4f} vs  "
        f"[四合一濾波] Max={filt_max_s:.4f}, Mean={filt_mean_s:.4f}, Std={filt_std_s:.4f}\n"
        f"      🩺 模型過擬合自檢 (Overfit)    : {overfit_badge}\n"
        f"                                       [Golden平均={g_mean_score:.4f} vs 非Golden中位數={median_s:.4f} | Overfit_Ratio={overfit_ratio:.2f}x | Gap={golden_gap:+.4f}]"
    )
    if overfit_advice:
        dist_summary_lines += f"\n{overfit_advice}"

    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== FoundAD High-Res Anomaly Detection Report (Inference Only) ===\n")
        f.write(
            f"Score Mode: {'RAW_UNFILTERED' if raw_score_mode else 'FILTERED_ROBUST'} | "
            f"Threshold Mode: {thresh_mode} | Effective Threshold: {effective_threshold:.4f} | "
            f"Stitch Canvas: {canvas_desc} | "
            f"Wafer Edge Mode: {'off(raw)' if raw_score_mode else wafer_edge_mode} (Margin: {wafer_edge_margin:.3f})\n"
            f"Core Box Ratio: {core_ratio:.2f} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_normal_count}\n"
            f"{metrics_report_header}"
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
    default_workers = min(8, os.cpu_count() or 4)
    parser = argparse.ArgumentParser(description="FoundAD High-Res Zero-GT Anomaly Inference Pipeline (無損加速版)")
    parser.add_argument("--pool_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./output/2_fab_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="mvtec_sim_4shot")
    parser.add_argument("--category", type=str, default="auto")
    parser.add_argument("--app_test", type=str, default="test")
    parser.add_argument("--diy_name", type=str, default="sim_escape")
    parser.add_argument("--ckpt_step", type=int, default=2000)

    parser.add_argument(
        "--thresh_mode", type=str, default="auto",
        choices=["auto", "hybrid", "mean_std", "percentile", "top_k", "fixed"],
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min_threshold", type=float, default=0.27)
    parser.add_argument("--auto_percentile", type=float, default=95.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--sigma_k", type=float, default=3.5)

    parser.add_argument("--raw_score_mode", action="store_true")
    parser.add_argument("--core_ratio", type=float, default=0.55)
    parser.add_argument("--border_ratio", type=float, default=0.02)
    parser.add_argument("--wafer_edge_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--wafer_edge_margin", type=float, default=0.04)
    parser.add_argument("--speckle_filter_ratio", type=float, default=0.008)
    parser.add_argument("--min_area_ratio", type=float, default=0.00012)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--force_infer", action="store_true")
    parser.add_argument("--mock_from_mvtec", type=str, default="")

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--bg_std_thresh", type=float, default=1.5)
    parser.add_argument("--disable_tiling", action="store_true")

    # 無損加速參數
    parser.add_argument("--infer_batch_size", type=int, default=64, help="GPU 推論 Batch Size (預設由 16 提升至 64)")
    parser.add_argument("--num_workers", type=int, default=default_workers, help="切塊、讀圖與縫合之多核心 Worker 數")
    parser.add_argument("--stitch_max_dim", type=int, default=1024, help="縫合與濾波畫布最大邊長 (預設 1024，設 0 則退回原圖解析度)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "off"], help="傳遞給 AD.py 推論的混合精度設定")
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
        f"\n{'=' * 75}\n"
        f"[Task 2 無GT抓漏純推論啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_out_dir}\n"
        f"[無損加速設定] BatchSize={args.infer_batch_size} | Workers={args.num_workers} | "
        f"StitchCanvas={args.stitch_max_dim}px | AMP={args.amp_dtype}\n"
        f"{'=' * 75}",
        log_files,
    )

    if args.mock_from_mvtec:
        existing_pool_imgs = scan_pool_images(args.pool_dir) if os.path.isdir(args.pool_dir) else []
        if existing_pool_imgs and not args.force_infer:
            log_and_print(
                f"[⚡ 模擬無GT盲池快取命中] 直接沿用既有 {len(existing_pool_imgs)} 張盲池圖片 "
                f"-> {os.path.abspath(args.pool_dir)}",
                log_files,
            )
        else:
            prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, log_files, defect_limit=8)

    pool_signature = compute_pool_signature(args.pool_dir)
    tile_config_key = (
        f"t{args.tile_size}_ov{args.tile_overlap}_bg{args.bg_std_thresh:g}_dt{int(args.disable_tiling)}"
    )

    golden_filenames = load_existing_golden_filenames(shot_dir, run_out_dir, log_files)
    actual_step, ckpt_dirs, ckpt_files = prepare_ckpt_for_demo(
        args.data_name, args.diy_name, args.ckpt_step, log_files
    )
    ckpt_signature = compute_ckpt_signature(ckpt_files)

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
            wafer_edge_mode=args.wafer_edge_mode,
            category=args.category,
            num_workers=args.num_workers,
        )
        test_tile_sec = time.perf_counter() - t_test_tile_start
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)

    safe_rmtree(tile_amap_tmp_dir)
    os.makedirs(tile_amap_tmp_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FAB_AMAP_DUMP_DIR"] = os.path.abspath(tile_amap_tmp_dir)
    env["FAB_SKIP_TILE_HEATMAP_PNG"] = "1"
    env["FAB_INFER_BATCH_SIZE"] = str(args.infer_batch_size)
    env["FAB_NUM_WORKERS"] = str(args.num_workers)
    env["FAB_AMP_DTYPE"] = str(args.amp_dtype)
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
        f"(共 {total_test_tiles} 張推論圖塊, BatchSize={args.infer_batch_size}, "
        f"Workers={args.num_workers}, AMP={args.amp_dtype}, "
        f"暫存於 {os.path.abspath(tile_amap_tmp_dir)} 並將於 Step 4 RAM 拼接後立即銷毀):",
        log_files,
    )
    run_cmd_and_log(demo_cmd, log_files, env=env)
    model_infer_sec = time.perf_counter() - t_infer_start

    for old_viz in glob.glob(f"./logs/{args.data_name}/**/no_gt_results", recursive=True):
        safe_rmtree(old_viz)
    for old_hm in glob.glob(f"./logs/{args.data_name}/**/demo/heatmaps", recursive=True):
        safe_rmtree(old_hm)

    t_stitch_start = time.perf_counter()
    anomaly_count, skipped_count, eff_thresh, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines = stitch_and_filter_anomalies(
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
        log_files=log_files,
        stitch_max_dim=args.stitch_max_dim,
        num_workers=args.num_workers,
    )
    stitch_vis_sec = time.perf_counter() - t_stitch_start

    total_infer_time = test_tile_sec + model_infer_sec + stitch_vis_sec
    avg_infer_per_img = total_infer_time / max(1, num_test_images)
    total_wall_time = time.perf_counter() - pipeline_start_t

    infer_done_summary = (
        f"\n{'=' * 75}\n"
        f" [✅ 推論完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-' * 75}\n"
        f"  [1] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ Step 3.1 測試池高解析度切塊 ({total_test_tiles} 塊)   : {format_duration(test_tile_sec)}"
        f"{' (Cached)' if used_png_tile_cache else ''}\n"
        f"      ├─ Step 3.2 FoundAD 模型推論 (step={actual_step}, BS={args.infer_batch_size}) : {format_duration(model_infer_sec)}\n"
        f"      ├─ Step 4 RAM 縫合銷毀碎片與{'未濾波' if args.raw_score_mode else '四合一濾波'}畫框 (Canvas<={args.stitch_max_dim}px) : {format_duration(stitch_vis_sec)}\n"
        f"      └─ 平均每張原圖推論耗時 ({num_test_images} 張原圖)         : {avg_infer_per_img:.2f} 秒/張\n"
        f"{'-' * 75}\n"
        f"  [2] 任務全程總執行時間 (Total Wall Time)      : {format_duration(total_wall_time)}\n"
        f"{'-' * 75}\n"
        f"  [3] 門檻篩選統計、分數分佈與過擬合自檢 (Score={'RAW' if args.raw_score_mode else 'FILTERED'}, Mode={args.thresh_mode}, Eff_Thresh={eff_thresh:.4f}):\n"
        f"      🚨 超過門檻判定異常並畫框輸出 : {anomaly_count} 張\n"
        f"      ✅ 未達門檻或微塵濾除直接略過 : {skipped_count} 張 (不輸出圖片)\n"
        f"{dist_summary_lines}\n"
        f"{'-' * 75}\n"
        f"  [4] 權重檔與本次推論結果儲存資料夾清單 (Saved Directories & Files):\n"
        f"      📦 載入訓練權重檔資料夾       : {ckpt_dirs[0]}\n"
        f"      🧹 子圖塊碎片 Tile .npy 狀態  : 已於 RAM 拼接完成後立即自動銷毀 (0 殘留)\n"
        f"      📂 本次時間戳記推論主資料夾   : {run_out_dir}\n"
        f"          ├── 異常紅框三合一對照圖  : {anomaly_dir}\n"
        f"          ├── 異常1:1全解析度紅框圖 : {anomaly_fullres_bbox_dir}\n"
        f"          ├── 異常熱力圖矩陣(.npy)  : {amap_dump_dir}\n"
        f"          ├── 完整排名報表 (TXT)    : {report_txt_path}\n"
        f"          ├── 完整排名報表 (CSV)    : {report_csv_path}\n"
        f"          └── 本次執行日誌 (log.txt): {fixed_log_txt}\n"
        f"{'=' * 75}"
    )

    log_and_print(infer_done_summary, log_files)

    if os.path.exists(report_txt_path):
        with open(report_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + infer_done_summary + "\n")


if __name__ == "__main__":
    main()
