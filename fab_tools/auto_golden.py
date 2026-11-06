# 檔案位置: FoundAD/fab_tools/auto_golden.py (全選覆蓋此檔)
import os
import glob
import shutil
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def scan_pool_images(pool_dir: str):
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


def load_pil_rgb_uint8(img_path: str) -> Image.Image:
    """安全讀取 8-bit / 16-bit TIFF 或單通道灰階圖，統一正規化為 8-bit RGB PIL Image。"""
    with Image.open(img_path) as pil_img:
        if pil_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(pil_img, dtype=np.float32)
            a_min, a_max = float(np.min(arr)), float(np.max(arr))
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min) * 255.0
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
            rgb_arr = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)
            return Image.fromarray(rgb_arr, mode="RGB")
        return pil_img.convert("RGB")


def resolve_dino_ckpt_path(ckpt_path: str) -> str:
    """若指定的 ckpt_path 不存在，依序優先搜尋 dinov3 -> dinov2 -> dino 權重檔"""
    if ckpt_path and os.path.exists(ckpt_path):
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


def inspect_and_extract_patch_proj(ckpt_path, device):
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


@torch.no_grad()
def select_golden_images(pool_dir, ckpt_path, out_dir, category="wafer", shots=4, report_dir=None):
    """
    Stage 1: 全域 DINO 特徵中心度 (Medoid Centrality) + 局部 Patch Token 尖峰懲罰挑選 Golden 原圖。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_imgs = scan_pool_images(pool_dir)
    if len(all_imgs) < shots:
        raise ValueError(f"資料夾 (含子資料夾) 內圖片數量 ({len(all_imgs)}) 少於所需的 shots ({shots})！")

    proj_conv, qkv_weights, patch_size = inspect_and_extract_patch_proj(ckpt_path, device)
    img_size = 518 if patch_size == 14 else 512
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features, token_spike_scores, valid_paths = [], [], []
    print(f"[Auto-Golden Stage 1] 遞迴掃描共找到 {len(all_imgs)} 張圖片，正在提取全域特徵與局部 Token 尖峰指標...")
    for path in all_imgs:
        try:
            img = load_pil_rgb_uint8(path)
            x = transform(img).unsqueeze(0).to(device)
            tokens = proj_conv(x).flatten(2).transpose(1, 2)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))

            # 計算單圖內部 Patch Token 離群尖峰度 (防止局部明顯瑕疵混入 Stage 1)
            token_mean = tokens.mean(dim=1, keepdim=True)
            token_dev = torch.norm(tokens - token_mean, dim=-1).squeeze(0)  # (N_patches,)
            top_k_tok = max(1, int(token_dev.numel() * 0.02))
            top_vals, _ = torch.topk(token_dev, top_k_tok)
            med_val = torch.median(token_dev).clamp(min=1e-6)
            spike_ratio = float((top_vals.mean() / med_val).item())
            token_spike_scores.append(spike_ratio)

            feat_list = [tokens.mean(dim=1), tokens.std(dim=1)]
            for w in qkv_weights[:3]:
                qkv = F.linear(tokens, w)
                feat_list.append(qkv.std(dim=1))

            feat = F.normalize(torch.cat(feat_list, dim=-1), dim=-1)
            features.append(feat.cpu())
            valid_paths.append(path)
        except Exception as e:
            print(f"  [略過損毀圖片] {path}: {e}")

    if len(valid_paths) < shots:
        raise ValueError(f"有效可讀取圖片數量 ({len(valid_paths)}) 少於所需的 shots ({shots})！")

    feats_tensor = torch.cat(features, dim=0)
    sim_matrix = torch.mm(feats_tensor, feats_tensor.t()).numpy()
    np.fill_diagonal(sim_matrix, 0.0)
    raw_centrality = sim_matrix.sum(axis=1) / max(1, len(valid_paths) - 1)

    # 對局部 Token 尖峰異常高的圖片施加微幅純淨度懲罰
    spike_arr = np.array(token_spike_scores, dtype=np.float32)
    spike_med = float(np.median(spike_arr))
    spike_mad = float(1.4826 * np.median(np.abs(spike_arr - spike_med))) + 1e-5
    spike_z = np.maximum(0.0, (spike_arr - spike_med) / spike_mad)
    purity_penalty = 0.0025 * np.clip(spike_z, 0.0, 6.0)
    hybrid_centrality = raw_centrality - purity_penalty

    sorted_indices = np.argsort(hybrid_centrality)[::-1]
    selected_indices = sorted_indices[:shots]
    selected_paths = [valid_paths[idx] for idx in selected_indices]

    dest_dir = os.path.join(out_dir, "train", category)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    record_txt = os.path.join(out_dir, f"selected_{shots}_golden_images.txt")
    with open(record_txt, "w", encoding="utf-8") as f:
        f.write("=== Auto-Selected Golden Images (Stage 1: Medoid Centrality + Token Spike Guard) ===\n")
        for i, idx in enumerate(selected_indices):
            src_path = valid_paths[idx]
            h_score = hybrid_centrality[idx]
            r_score = raw_centrality[idx]
            s_ratio = spike_arr[idx]
            ext = os.path.splitext(src_path)[1].lower()
            dst_path = os.path.join(dest_dir, f"{i:03d}{ext}")
            shutil.copy2(src_path, dst_path)
            rel_name = os.path.relpath(src_path, pool_dir).replace("\\", "/")
            log_line = (
                f"Golden #{i+1} (-> {i:03d}{ext}): {rel_name} | "
                f"Hybrid_Score: {h_score:.6f} (Raw_Centrality: {r_score:.6f}, Token_Spike: {s_ratio:.3f})"
            )
            print("  " + log_line)
            f.write(log_line + "\n")

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        shutil.copy2(record_txt, os.path.join(report_dir, f"selected_{shots}_golden_images.txt"))

    print(f"[Auto-Golden Stage 1] 已建立 FoundAD 初始 Golden 目錄: {dest_dir}")
    return selected_paths


# =====================================================================
# Stage 2: 圖塊級二次純化 (Two-Stage Hybrid Tile-Level Filtering)
# =====================================================================
@torch.no_grad()
def _extract_lightweight_dino_tile_embeddings(
    tiles_rgb: List[np.ndarray],
    ckpt_path: str,
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    """
    使用 DINO patch_embed 與前 3 層 QKV 權重，對候選高解析度子圖塊批次提取特徵向量。
    包含 [Mean, Std, P98_Max] 以同時捕捉圖塊整體結構與局部高頻異常。
    """
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

            # 同時納入 mean, std 與局部最大值 (amax)，使微米級瑕疵在特徵空間突顯
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


def _compute_local_patch_peak_residuals(crops: List[np.ndarray], eval_res: int = 256, patch_grid: int = 16) -> np.ndarray:
    """
    計算同座標 (y, x) 跨 Golden 圖塊之「微位移容忍局部區塊殘差峰值 (Shift-Tolerant Local Patch-Peak Residual)」：
    1. 將每個 512x512 圖塊轉為 256x256 並以 3x3 高斯平滑吸收 1~2px 晶圓對位微偏移。
    2. 計算跨 Golden 的像素中位數基準圖 (Median Reference Tile)。
    3. 將每張圖塊與中位數基準圖相減後，切分為 16x16 局部小區塊，取 Top-2 局部區塊殘差均值作為峰值指標。
       任何在 Stage 1 被稀釋的微小刮傷或微塵，在該 16x16 局部區塊內都會產生顯著殘差尖峰！
    """
    n = len(crops)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    blurred_stack = []
    for c in crops:
        small = cv2.resize(c, (eval_res, eval_res), interpolation=cv2.INTER_AREA).astype(np.float32)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        blurred_stack.append(blur)

    arr = np.stack(blurred_stack, axis=0)  # (N, H, W, 3)
    med_tile = np.median(arr, axis=0, keepdims=True)  # (1, H, W, 3)
    abs_diff = np.mean(np.abs(arr - med_tile), axis=-1)  # (N, H, W)

    # 將 (N, H, W) 劃分為 (N, patch_grid, H//patch_grid, patch_grid, W//patch_grid) 計算局部區塊平均殘差
    cell_h = eval_res // patch_grid
    cell_w = eval_res // patch_grid
    patched = abs_diff.reshape(n, patch_grid, cell_h, patch_grid, cell_w).mean(axis=(2, 4)).reshape(n, -1)

    # 取每個圖塊內部最異常的前 2 個局部區塊殘差 + 整體平均殘差
    top2_patch_peak = np.sort(patched, axis=1)[:, -2:].mean(axis=1)
    global_l1 = abs_diff.mean(axis=(1, 2))
    return (0.70 * top2_patch_peak + 0.30 * global_l1).astype(np.float32)


def purify_golden_tiles_two_stage(
    coord_Execution_items: List[Tuple[int, int, float, List[Tuple[int, np.ndarray]]]],
    max_train_tiles: int,
    ckpt_path: str = "auto",
) -> Tuple[List[Tuple[int, int, int, float]], Dict[str, int]]:
    """
    Stage 2 圖塊級二次純化器 (Two-Stage Hybrid Filtering)：
    輸入:
      coord_Execution_items: List of (y, x, radius, non_bg_items)
        其中 non_bg_items 為 [(g_idx, crop_rgb_uint8), ...]
      max_train_tiles: 欲選出的代表性高解析度訓練圖塊上限 (如 48)
      ckpt_path: DINO 權重路徑 (用於提取圖塊級深層語意特徵)
    輸出:
      selected_specs: List of (y, x, g_idx, purity_cost)
      stats: 診斷統計字典 (總評估圖塊數、跨片剔除離群圖塊數、徑向分層純化替換數等)
    """
    flat_tiles: List[np.ndarray] = []
    coord_slices: List[Tuple[int, int]] = []
    for (_, _, _, non_bg_items) in coord_Execution_items:
        start_idx = len(flat_tiles)
        for (_, crop) in non_bg_items:
            flat_tiles.append(crop)
        coord_slices.append((start_idx, len(flat_tiles)))

    total_evaluated_tiles = len(flat_tiles)
    dino_feats = _extract_lightweight_dino_tile_embeddings(flat_tiles, ckpt_path=ckpt_path, batch_size=32)

    # Step 2.1: 同座標 (y, x) 跨 Golden 圖塊之「DINO 語意距離 + 局部 16x16 區塊殘差尖峰」混合評分與離群剔除
    purified_candidates = []
    rejected_outlier_tiles = 0

    for idx, (y, x, radius, non_bg_items) in enumerate(coord_Execution_items):
        s_idx, e_idx = coord_slices[idx]
        n_items = len(non_bg_items)
        if n_items == 1:
            g_idx_0 = non_bg_items[0][0]
            purified_candidates.append(
                {
                    "radius": float(radius),
                    "y": int(y),
                    "x": int(x),
                    "best_g_idx": int(g_idx_0),
                    "best_cost": 0.0,
                    "safe_extras": [],
                }
            )
            continue

        crops = [item[1] for item in non_bg_items]
        peak_res = _compute_local_patch_peak_residuals(crops)  # (n_items,)
        norm_peak = peak_res / (float(np.median(peak_res)) + 1e-4)

        if dino_feats is not None:
            sub_f = dino_feats[s_idx:e_idx]  # (n_items, D)
            sim_mat = np.matmul(sub_f, sub_f.T)
            dist_mat = np.maximum(0.0, 1.0 - sim_mat)
            dino_dist = dist_mat.sum(axis=1) / max(1, n_items - 1)
            norm_dino = dino_dist / (float(np.median(dino_dist)) + 1e-5)
            hybrid_cost = 0.55 * norm_peak + 0.45 * norm_dino
            raw_abs_cost = 0.5 * peak_res + 10.0 * dino_dist
        else:
            hybrid_cost = norm_peak
            raw_abs_cost = peak_res

        sorted_order = np.argsort(hybrid_cost)
        best_local_i = int(sorted_order[0])
        best_g_idx = int(non_bg_items[best_local_i][0])
        best_abs_cost = float(raw_abs_cost[best_local_i])

        # 自適應離群門檻：剔除排名最後一名，且若其他圖塊的局部殘差峰值超過最優圖塊的 1.65 倍亦強制剔除
        med_cost = float(np.median(hybrid_cost))
        mad_cost = float(1.4826 * np.median(np.abs(hybrid_cost - med_cost))) + 1e-4
        adaptive_gate = min(float(hybrid_cost[best_local_i]) * 1.65 + 0.15, med_cost + 2.0 * mad_cost)

        safe_extras: List[Tuple[int, float]] = []
        if n_items >= 3:
            candidate_extra_order = sorted_order[1:-1]
            rejected_outlier_tiles += 1  # 最差的一張必然剔除
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

    # Step 2.2: 跨座標徑向分層純淨度篩選 (Radial Stratified Purity Binning)
    # 依半徑由圓心到外環排序後，切分為 max_train_tiles 個同心徑向區間，在每個徑向區間內挑出最純淨的座標！
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
            # 在該徑向區間內尋找跨片殘差最低 (最純淨) 的座標
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
            # 將所有通過離群檢驗的跨片公差候補圖塊，按純淨度由高至低排序後補齊
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
    }
    return selected_specs, stats