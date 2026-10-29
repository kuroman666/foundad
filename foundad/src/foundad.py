# 檔案位置: FoundAD/foundad/src/foundad.py (全選覆蓋此檔)
import os
import glob
from typing import Any, Dict, Tuple, Optional, List
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from transformers import AutoProcessor, SiglipVisionModel, CLIPVisionModel


def _infer_dino_arch_from_ckpt(ckpt_path: str, default_arch: str) -> Tuple[str, dict]:
    """自動從本地 .pth 權重檔推斷它是 vit_small / vit_base / vit_large 以及 patch 14 或 16"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("teacher", ckpt.get("state_dict", ckpt)))
    clean_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    embed_dim, patch_size = 768, 16
    for k, v in clean_sd.items():
        if "patch_embed.proj.weight" in k:
            embed_dim, _, patch_size, _ = v.shape
            break

    size_map = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}
    vit_prefix = size_map.get(embed_dim, "vitb")
    prefix = "dinov2" if "dinov2" in default_arch else "dinov3"
    inferred_arch = f"{prefix}_{vit_prefix}{patch_size}"
    print(f"[Offline Loader] 偵測本地權重 {os.path.basename(ckpt_path)} -> 自動匹配架構: {inferred_arch}")
    return inferred_arch, clean_sd


class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(
        self, model_name: str, pred_depth: int, pred_emb_dim: int,
        use_cuda: bool = True, if_pe: bool = True, feat_normed: bool = False
    ):
        super().__init__()
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](
            num_patches=self.num_patches, embed_dim=self.embed_dim,
            predictor_embed_dim=pred_emb_dim, depth=pred_depth, if_pe=if_pe, feat_normed=feat_normed
        )
        self._init_predictor(self.predictor)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()
        self.feat_normed = self.predictor.feat_normed
        print(f"Normed features: {self.feat_normed}")

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def target_features(self, images, paths, n_layer=3):
        with torch.no_grad():
            return self._extract(images, paths, n_layer=n_layer)

    def context_features(self, images, paths, n_layer=3):
        z = self._extract(images, paths, n_layer=n_layer)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):
        projector = processor = None
        # 讀取環境變數：若進公司設定了本地權重與本地 repo，則 100% 走離線載入
        local_ckpt = os.environ.get("FAB_LOCAL_DINO_CKPT", "")
        local_repo = os.environ.get("FAB_LOCAL_DINO_REPO", "")

        if model == "dinov2":
            if local_ckpt and os.path.exists(local_ckpt):
                arch_name, state_dict = _infer_dino_arch_from_ckpt(local_ckpt, "dinov2_vitb14")
                if local_repo and os.path.exists(local_repo):
                    enc = torch.hub.load(local_repo, arch_name, source="local", pretrained=False)
                    enc.load_state_dict(state_dict, strict=False)
                else:
                    # 直接使用專案內建的 src.dinov2 建立模型，完全不需依賴外部 repo！
                    vit_fn = {"dinov2_vits14": "vit_small", "dinov2_vitb14": "vit_base", "dinov2_vitl14": "vit_large"}.get(arch_name, "vit_base")
                    enc = vit.__dict__[vit_fn](patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
                    enc.load_state_dict(state_dict, strict=False)
                enc = enc.eval()
            else:
                enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim

        elif model == "dinov3":
            if local_ckpt and os.path.exists(local_ckpt):
                arch_name, state_dict = _infer_dino_arch_from_ckpt(local_ckpt, "dinov3_vitb16")
                # 自動尋找本地快取的 facebookresearch_dinov3_main 資料夾
                if not (local_repo and os.path.exists(local_repo)):
                    candidates = glob.glob(os.path.expanduser("~/.cache/torch/hub/*dinov3*")) + glob.glob("./hub_cache/*dinov3*")
                    if candidates:
                        local_repo = candidates[0]
                if local_repo and os.path.exists(local_repo):
                    print(f"[Offline Loader] 從本地 Repo 載入 DINOv3: {local_repo}")
                    enc = torch.hub.load(local_repo, arch_name, source="local", pretrained=False)
                    enc.load_state_dict(state_dict, strict=False)
                    enc = enc.eval()
                else:
                    enc = torch.hub.load("facebookresearch/dinov3", arch_name, source="github", pretrained=False)
                    enc.load_state_dict(state_dict, strict=False)
                    enc = enc.eval()
            else:
                enc = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", source="github").eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim

        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval()
            num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval()
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
            num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval()
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
            num_patches, embed_dim = 196, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config
            config = Config()
            enc = init_vit_backbone(config)
            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")

        if model != "dinosiglip":
            for p in enc.parameters():
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    def _extract(self, imgs: torch.Tensor, paths: List[str], n_layer: int = 3):
        if self.model_name in ("dinov2", "dinov3"):
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0]
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer)[0][:, 1:, :]
        elif self.model_name == "siglip":
            pil_list = [Image.open(p).convert("RGB") for p in paths]
            proc = self.processor(images=pil_list, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(imgs.device)
            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states
            L = len(hs) - 1
            n = max(1, min(n_layer, L))
            h = hs[-n][:, :, :]
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            L = len(hs) - 1
            n = max(1, min(n_layer, L))
            h = hs[-n][:, 1:, :]
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0, 2, 1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            h = F.normalize(h, dim=-1)
        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
