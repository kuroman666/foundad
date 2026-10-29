# 檔案位置: FoundAD/2_run_fab_no_gt.py (全選覆蓋並存檔)
import os
import re
import glob
import shutil
import subprocess
import argparse

from fab_tools.auto_golden import select_golden_images, IMG_EXTENSIONS


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
            shutil.copy2(target_ckpt, pretrained_alias)
            print(f"[Auto-Ckpt] 已將訓練權重 {target_ckpt} 對接至 {pretrained_alias}")
            found_any = True

    if not found_any:
        raise FileNotFoundError(
            f"在 ./logs/{data_name}/ 下找不到任何 train-step*.pth.tar 權重檔！代表 Step 2 訓練未成功完成。"
        )
    return chosen_step


def prepare_mock_unlabeled_pool_from_mvtec(mvtec_cat_dir, mock_pool_dir, defect_limit=8):
    if os.path.exists(mock_pool_dir):
        shutil.rmtree(mock_pool_dir)
    os.makedirs(mock_pool_dir, exist_ok=True)
    test_dir = os.path.join(mvtec_cat_dir, "test")

    idx = 1
    good_dir = os.path.join(test_dir, "good")
    for f in sorted(os.listdir(good_dir)):
        if f.lower().endswith(IMG_EXTENSIONS):
            shutil.copy2(os.path.join(good_dir, f), os.path.join(mock_pool_dir, f"wafer_{idx:04d}_clean.png"))
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
                shutil.copy2(os.path.join(sub_dir, f), os.path.join(mock_pool_dir, f"wafer_{idx:04d}_escape_{sub}.png"))
                defect_count += 1
                idx += 1

    print(f"[模擬無GT盲池建立完成] 共 {idx-1} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵) -> {mock_pool_dir}")


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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)

    if args.mock_from_mvtec:
        prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, defect_limit=8)

    abs_datasets_dir = os.path.abspath(args.datasets_dir)
    shot_dir = os.path.join(abs_datasets_dir, args.data_name)

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

    # Step 2: 執行 FoundAD 少樣本訓練
    if not args.skip_train:
        train_cmd = (
            f"python foundad/main.py mode=train data.batch_size={args.shots} "
            f"data.dataset=mvtec data.data_name={args.data_name} "
            f"data.data_path={abs_datasets_dir} data.train_root={shot_dir} "
            f"data.mvtec_classnames='[{args.category}]' "
            f"diy_name={args.diy_name} app={args.app_train}"
        )
        print(f"\n[Step 2] 執行 FoundAD 訓練 (2000 Epochs):\n{train_cmd}")
        subprocess.run(train_cmd, shell=True, check=True)

    # 確認權重已生成並複製為 pretrained.pth.tar 供 mode=demo 讀取
    actual_step = prepare_ckpt_for_demo(args.data_name, args.diy_name, args.ckpt_step)

    # Step 3: 執行 mode=demo 計算熱力圖並直接畫出紅色瑕疵外框
    amap_dump_dir = os.path.join(args.out_dir, "amap_npy")
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
            shutil.copytree(hm_src, hm_dst, dirs_exist_ok=True)

    print(f"\n[✓] 無 GT 抓漏所有文字報表與紅框圖片已統一整合至: {args.out_dir}")


if __name__ == "__main__":
    main()
