# 檔案位置: FoundAD/foundad/src/AD.py (全選覆蓋此檔)
import os
import sys
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule          

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
    )


@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == "mvtec":
        classnames = cfg["data"]["mvtec_classnames"]
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == "visa":
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError

    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")

    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]
            labels.extend(batch["is_anomaly"])
            name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            pred = model.predict(enc)

            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)

            topk = torch.topk(l, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu())
            img_buf.append(img.cpu())
            mask_buf.append(mask.cpu())

            dump_dir = os.environ.get("FAB_AMAP_DUMP_DIR", "")
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                img_fname = os.path.basename(paths[0])
                np.save(os.path.join(dump_dir, f"{img_fname}.npy"), pix.squeeze().detach().cpu().numpy())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8)

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(
            mask_np, pix_norm,
            max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"]
        )

        logger.info(
            "%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
            cls, inst["auroc"], inst["aupr"], pix["auroc"], pro
        )
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"])
        inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"])
        pro_auc.append(pro)

        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0, 2, 3, 1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info(
        "Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
        np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc)
    )
    csv_logger.log(
        ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
        np.mean(pix_auc), np.mean(pro_auc)
    )


@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    dump_dir = os.environ.get("FAB_AMAP_DUMP_DIR", "")
    skip_tile_png = os.environ.get("FAB_SKIP_TILE_HEATMAP_PNG", "1" if dump_dir else "0") == "1"
    infer_bs = max(1, int(os.environ.get("FAB_INFER_BATCH_SIZE", "16")))

    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    if not skip_tile_png:
        out_root.mkdir(parents=True, exist_ok=True)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    test_root = Path(cfg["data"]["test_root"])
    exts = (
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP"
    )
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    total_imgs = len(img_paths)
    print(
        f"[INFO] Found {total_imgs} tile images under {test_root} "
        f"(BatchSize={infer_bs}, SaveTilePNG={'OFF' if skip_tile_png else 'ON'})"
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _load_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as pil:
            pil_rgb = pil.convert("RGB")
            if pil_rgb.size != (crop, crop):
                pil_rgb = pil_rgb.resize((crop, crop), Image.BILINEAR)
            arr = np.array(pil_rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        import cv2
        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for start_idx in range(0, total_imgs, infer_bs):
        batch_paths = img_paths[start_idx:start_idx + infer_bs]
        batch_tensors = torch.stack([_load_tensor(p) for p in batch_paths], dim=0).to(device, non_blocking=True)
        batch_norm = (batch_tensors - mean) / std
        str_paths = [str(p) for p in batch_paths]

        enc = model.target_features(batch_norm, str_paths, n_layer=n_layer)
        pred = model.predict(enc)

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)
        h = w = int(math.sqrt(l.size(1)))
        pix_batch = F.interpolate(
            l.view(-1, 1, h, w), size=(crop, crop), mode="bilinear", align_corners=False
        ).squeeze(1)

        pix_np_batch = pix_batch.detach().cpu().numpy()

        for b_idx, path in enumerate(batch_paths):
            raw_amap = pix_np_batch[b_idx]
            if dump_dir:
                np.save(os.path.join(dump_dir, f"{path.name}.npy"), raw_amap)

            if not skip_tile_png:
                pmin, pmax = float(raw_amap.min()), float(raw_amap.max())
                pix_norm = (raw_amap - pmin) / (pmax - pmin + 1e-8)
                img_uint8 = (batch_tensors[b_idx].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                rel = path.relative_to(test_root)
                save_dir = out_root / rel.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{path.stem}_heatmap.png"
                _save_overlay_heatmap(img_uint8, pix_norm, save_path)

        done_cnt = min(start_idx + infer_bs, total_imgs)
        if done_cnt == total_imgs or (done_cnt // infer_bs) % 5 == 0:
            print(f"[Tile-Infer] Progress: [{done_cnt}/{total_imgs}] tiles processed (.npy saved)")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")


if __name__ == "__main__":
    main()