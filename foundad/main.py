# 檔案位置: FoundAD/foundad/main.py (全選覆蓋並存檔)
import os
import sys
import socket
import logging
import multiprocessing as mp

import pprint
import yaml
import torch

from src.train import main as app_main_mvtec
from src.AD import main as AD, _demo

import hydra
from omegaconf import DictConfig, OmegaConf


def resolve_safe_master_port(dist_cfg: dict) -> int:
    """
    自動分配安全且未被佔用的 DDP master_port，徹底根除 Port 40112 Address already in use：
    1. 優先讀取外部啟動腳本傳入之環境變數 FAB_MASTER_PORT（並驗證是否可用）。
    2. 若設定檔指定非 40112 之自訂 Port，先驗證是否可用。
    3. 若為預設 40112 或上述 Port 已被佔用，自動由 OS 動態分配保證空閒的本地 Port。
    """
    env_port = os.environ.get("FAB_MASTER_PORT", "").strip()
    candidates = []
    if env_port.isdigit():
        candidates.append(int(env_port))

    req_port = int(dist_cfg.get("master_port", 40112))
    if req_port != 40112 and req_port not in candidates:
        candidates.append(req_port)

    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def process_main(rank: int, cfg_dict: dict, world_size: int):
    """
    rank: local rank
    cfg_dict: dict config
    world_size: len(devices)
    """
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    devices = cfg_dict.get("devices", ["cuda:0"])
    mode = cfg_dict.get("mode", "train")
    dist = cfg_dict.get("dist", {})
    master_addr = dist.get("master_addr", "localhost")
    master_port = str(dist.get("master_port", 40112))
    backend = dist.get("backend", "nccl")

    dev_str = devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_str.split(":")[-1])

    params = cfg_dict.get("app", {})
    params.update(cfg_dict)

    logger.info("Params:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(params)

    # ---- DDP
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    try:
        if mode == "train":
            if rank == 0:
                log_dir = params["logging"]["folder"]
                os.makedirs(log_dir, exist_ok=True)
                params_save_path = os.path.join(log_dir, "params.yaml")
                with open(params_save_path, "w") as f:
                    yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)
                print(f"Config saved to {params_save_path}", flush=True)

            app_main_mvtec(args=params)
        elif mode == "AD":
            load_path = os.path.join(
                "logs", cfg_dict["data"]["data_name"], params.get("model_name", "") + cfg_dict["diy_name"]
            )
            saved_path = os.path.join(load_path, "params.yaml")
            if os.path.exists(saved_path):
                with open(saved_path, "r") as f:
                    saved_params = yaml.safe_load(f)
                    assert cfg_dict["diy_name"] == saved_params["diy_name"]
                    params["meta"] = saved_params["meta"]
                    params["ckpt_path"] = os.path.join(
                        saved_params["logging"]["folder"], f"train-step{params['ckpt_step']}.pth.tar"
                    )
                    params["logging"]["folder"] = os.path.join(
                        saved_params["logging"]["folder"], f"eval/{str(params['ckpt_step'])}"
                    )
                AD(args=params)
            else:
                raise FileNotFoundError(f"No ckpt path is found at {saved_path}")
        elif mode == "demo":
            load_path = os.path.join(
                "logs", cfg_dict["data"]["data_name"], params.get("model_name", "") + cfg_dict["diy_name"]
            )
            saved_path = os.path.join(load_path, "params.yaml")
            if os.path.exists(saved_path):
                with open(saved_path, "r") as f:
                    saved_params = yaml.safe_load(f)
                    assert cfg_dict["diy_name"] == saved_params["diy_name"]
                    params["meta"] = saved_params["meta"]
                    params["ckpt_path"] = os.path.join(saved_params["logging"]["folder"], "pretrained.pth.tar")
                    params["logging"]["folder"] = os.path.join(saved_params["logging"]["folder"], "demo")
                print(f"loading {params['ckpt_path']}...", flush=True)
                _demo(params["ckpt_path"], params)
            else:
                raise FileNotFoundError(f"No ckpt is found at {saved_path}")
        else:
            if rank == 0:
                logger.error(f"Unknown mode: {mode}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if "dist" not in cfg_dict or not isinstance(cfg_dict["dist"], dict):
        cfg_dict["dist"] = {}
    cfg_dict["dist"]["master_port"] = resolve_safe_master_port(cfg_dict["dist"])

    devices = cfg_dict.get("devices", ["cuda:0"])
    world_size = len(devices)

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=process_main, args=(rank, cfg_dict, world_size))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            sys.exit(p.exitcode)


if __name__ == "__main__":
    main()