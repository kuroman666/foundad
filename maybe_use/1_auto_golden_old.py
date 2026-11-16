# 檔案位置: FoundAD/1_auto_golden_old.py (原始檔案為 FoundAD/fab_tools/auto_golden.py(1104))
import os
import glob
import shutil
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

    features, valid_paths = [], []
    print(f"[Auto-Golden] 遞迴掃描共找到 {len(all_imgs)} 張圖片，正在從全部 {len(all_imgs)} 張無標註圖片提取特徵以尋找純淨中心點...")
    for path in all_imgs:
        try:
            img = load_pil_rgb_uint8(path)
            x = transform(img).unsqueeze(0).to(device)
            tokens = proj_conv(x).flatten(2).transpose(1, 2)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))

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
    centrality_scores = sim_matrix.sum(axis=1) / max(1, len(valid_paths) - 1)

    sorted_indices = np.argsort(centrality_scores)[::-1]
    selected_indices = sorted_indices[:shots]
    selected_paths = [valid_paths[idx] for idx in selected_indices]

    dest_dir = os.path.join(out_dir, "train", category)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    record_txt = os.path.join(out_dir, f"selected_{shots}_golden_images.txt")
    with open(record_txt, "w", encoding="utf-8") as f:
        f.write("=== Auto-Selected Golden Images (Medoid Centrality) ===\n")
        for i, idx in enumerate(selected_indices):
            src_path = valid_paths[idx]
            score = centrality_scores[idx]
            ext = os.path.splitext(src_path)[1].lower()
            dst_path = os.path.join(dest_dir, f"{i:03d}{ext}")
            shutil.copy2(src_path, dst_path)
            rel_name = os.path.relpath(src_path, pool_dir).replace("\\", "/")
            log_line = f"Golden #{i+1} (-> {i:03d}{ext}): {rel_name} | Centrality: {score:.6f}"
            print("  " + log_line)
            f.write(log_line + "\n")

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        shutil.copy2(record_txt, os.path.join(report_dir, f"selected_{shots}_golden_images.txt"))

    print(f"[Auto-Golden] 已建立 FoundAD 訓練目錄: {dest_dir}")
    return selected_paths