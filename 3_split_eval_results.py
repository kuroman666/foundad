# 檔案位置: FoundAD/3_split_eval_results.py
import os
import re
import csv
import glob
import json
import random
import shutil
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np


def find_latest_run_dir(base_dir: str) -> str:
    """若傳入的是父目錄 (例如 ./output/2_fab_no_gt)，自動尋找底下最新的時間戳記結果資料夾。"""
    abs_base = os.path.abspath(base_dir)
    if os.path.isfile(os.path.join(abs_base, "escape_ranking_report.csv")):
        return abs_base

    csv_candidates = sorted(glob.glob(os.path.join(abs_base, "*", "escape_ranking_report.csv")))
    if not csv_candidates:
        raise FileNotFoundError(
            f"在 {abs_base} 及其子目錄下找不到 escape_ranking_report.csv！請確認已完成推論。"
        )
    return os.path.dirname(csv_candidates[-1])


def parse_header_metadata(txt_path: str) -> Dict[str, str]:
    """從原始 escape_ranking_report.txt 讀取 Effective Threshold 與模型設定資訊。"""
    meta = {
        "score_mode": "FILTERED_ROBUST",
        "thresh_mode": "hybrid",
        "eff_thresh": "0.2500",
        "wafer_edge_info": "auto",
        "overfit_diag": "N/A",
    }
    if not os.path.isfile(txt_path):
        return meta

    with open(txt_path, "r", encoding="utf-8") as f:
        head = f.read(2000)

    m_sm = re.search(r"Score Mode:\s*([A-Z_]+)", head)
    if m_sm:
        meta["score_mode"] = m_sm.group(1).strip()

    m_tm = re.search(r"Threshold Mode:\s*([a-zA-Z0-9_]+)", head)
    if m_tm:
        meta["thresh_mode"] = m_tm.group(1).strip()

    m_et = re.search(r"Effective Threshold:\s*([0-9.]+)", head)
    if m_et:
        meta["eff_thresh"] = m_et.group(1).strip()

    m_we = re.search(r"Wafer Edge Mode:\s*([^\n]+)", head)
    if m_we:
        meta["wafer_edge_info"] = m_we.group(1).strip()

    m_ov = re.search(r"Overfit Diagnostic\s*:\s*([^\n]+)", head)
    if m_ov:
        meta["overfit_diag"] = m_ov.group(1).strip()

    return meta


def compute_pack_sizes(
    total_items: int,
    num_packs: int,
    counts_str: str,
    ratios_str: str,
) -> List[int]:
    """
    根據使用者設定的「每包張數 (--counts)」或「比例 (--ratios)」計算每一包應分配的張數。
    - counts_str: 例如 "50" (每包50張，共3包) 或 "40,40,20" (分別指定3包張數)
    - ratios_str: 例如 "1:1:1" 或 "0.4,0.3,0.3" 或 "2:1:1"
    """
    if counts_str and ratios_str:
        raise ValueError("請勿同時指定 --counts 與 --ratios，請擇一使用！")

    # 模式 A：指定每包張數 (--counts)
    if counts_str:
        parts = [int(x.strip()) for x in re.split(r"[,:/\s]+", counts_str.strip()) if x.strip()]
        if len(parts) == 1:
            per_pack = parts[0]
            if per_pack <= 0:
                raise ValueError("--counts 數值必須大於 0")
            sizes = []
            rem = total_items
            for i in range(num_packs):
                take = min(per_pack, rem)
                sizes.append(take)
                rem -= take
            return sizes
        else:
            if any(p < 0 for p in parts):
                raise ValueError("--counts 不可包含負數")
            sizes = []
            rem = total_items
            for p in parts:
                take = min(p, rem)
                sizes.append(take)
                rem -= take
            return sizes

    # 模式 B：指定比例 (--ratios)，或預設均分 3 份 (1:1:1)
    if not ratios_str:
        ratios_str = ":".join(["1"] * num_packs)

    raw_ratios = [float(x.strip()) for x in re.split(r"[,:/\s]+", ratios_str.strip()) if x.strip()]
    if not raw_ratios or any(r <= 0 for r in raw_ratios):
        raise ValueError(f"無效的比例設定: {ratios_str}")

    total_weight = sum(raw_ratios)
    exact_counts = [total_items * (r / total_weight) for r in raw_ratios]
    base_counts = [int(np.floor(c)) for c in exact_counts]
    remainder = total_items - sum(base_counts)

    # 依小數點餘數由大到小補足剩餘張數，確保總和 100% 等於 total_items
    frac_with_idx = sorted(
        enumerate([exact_counts[i] - base_counts[i] for i in range(len(raw_ratios))]),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remainder):
        base_counts[frac_with_idx[i][0]] += 1

    return base_counts


def split_records(
    records: List[Dict],
    pack_sizes: List[int],
    split_order: str,
    seed: int,
) -> List[List[Dict]]:
    """
    支援 4 種排序/抽樣拆分策略：
    - random     : 固定隨機種子打亂後依張數/比例切分 (適合一般隨機分包 Eval)
    - stratified : 分層抽樣 (確保每一包裡面的 ANOMALY / GOLDEN / PASS 比例一致)
    - rank       : 依照異常分數排名由高到低依序切分 (例如 Eval_1 高分群、Eval_2 中分群、Eval_3 低分群)
    - filename   : 依照原始檔名字母順序排列後切分
    """
    n_packs = len(pack_sizes)
    work_records = list(records)

    if split_order == "rank":
        work_records.sort(key=lambda r: float(r["Max_Score"]), reverse=True)
    elif split_order == "filename":
        work_records.sort(key=lambda r: r["Original_Filename"])
    elif split_order == "random":
        rng = random.Random(seed)
        rng.shuffle(work_records)
    elif split_order == "stratified":
        rng = random.Random(seed)
        # 依狀態分群後交錯發牌至各包，確保異常與正常比例在 3 份 Eval 中均勻分佈
        buckets: Dict[str, List[Dict]] = {}
        for r in work_records:
            key = "ESCAPE" if r["Original_Filename"].startswith("escape_") else r["Status"]
            buckets.setdefault(key, []).append(r)

        interleaved = []
        for k in sorted(buckets.keys()):
            rng.shuffle(buckets[k])
        max_len = max((len(v) for v in buckets.values()), default=0)
        for i in range(max_len):
            for k in sorted(buckets.keys()):
                if i < len(buckets[k]):
                    interleaved.append(buckets[k][i])
        work_records = interleaved

    packs: List[List[Dict]] = [[] for _ in range(n_packs)]
    if split_order == "stratified" and sum(pack_sizes) == len(work_records):
        # 依照各包目標容量進行權重輪詢分配，讓分層樣本均勻落入各包
        remaining_cap = list(pack_sizes)
        idx = 0
        for item in work_records:
            while remaining_cap[idx % n_packs] <= 0:
                idx += 1
            target_pack = idx % n_packs
            packs[target_pack].append(item)
            remaining_cap[target_pack] -= 1
            idx += 1
    else:
        cursor = 0
        for p_idx, sz in enumerate(pack_sizes):
            packs[p_idx] = work_records[cursor : cursor + sz]
            cursor += sz

    # 每包內部重新依 Max_Score 由高到低排序，方便查看報表
    for p in packs:
        p.sort(key=lambda r: float(r["Max_Score"]), reverse=True)
    return packs


def link_or_copy_file(src: str, dst: str, use_symlink: bool = False):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    if use_symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def generate_pack_eval_report(
    pack_idx: int,
    pack_records: List[Dict],
    src_run_dir: str,
    pack_out_dir: str,
    header_meta: Dict[str, str],
    use_symlink: bool = False,
) -> Dict:
    """為單一拆分包建立完整的 Eval 資料夾、複製對應的紅框圖/熱力圖並產出獨立的 TXT/CSV 評估報表。"""
    os.makedirs(pack_out_dir, exist_ok=True)
    anom_vis_dir = os.path.join(pack_out_dir, "anomaly_images")
    anom_full_dir = os.path.join(anom_vis_dir, "full_res_bbox_only")
    amap_dir = os.path.join(pack_out_dir, "amap_npy")
    os.makedirs(anom_vis_dir, exist_ok=True)
    os.makedirs(anom_full_dir, exist_ok=True)
    os.makedirs(amap_dir, exist_ok=True)

    src_anom_vis_dir = os.path.join(src_run_dir, "anomaly_images")
    src_anom_full_dir = os.path.join(src_anom_vis_dir, "full_res_bbox_only")
    src_amap_dir = os.path.join(src_run_dir, "amap_npy")

    eff_thresh = float(header_meta.get("eff_thresh", 0.25))
    anomaly_count = 0
    skipped_count = 0

    # 複製該包專屬的異常圖片與 .npy，並更新 pack 內排名
    updated_records = []
    for local_rank, r in enumerate(pack_records, 1):
        row = dict(r)
        row["Global_Rank"] = r["Rank"]
        row["Pack_Rank"] = local_rank
        saved_fname = r.get("Saved_Anomaly_Filename", "")

        if "ANOMALY" in r["Status"] and saved_fname and not saved_fname.startswith("("):
            anomaly_count += 1
            link_or_copy_file(
                os.path.join(src_anom_vis_dir, saved_fname),
                os.path.join(anom_vis_dir, saved_fname),
                use_symlink=use_symlink,
            )
            link_or_copy_file(
                os.path.join(src_anom_full_dir, saved_fname),
                os.path.join(anom_full_dir, saved_fname),
                use_symlink=use_symlink,
            )
            orig_stem = re.sub(r"[\\/]+", "__", os.path.splitext(r["Original_Filename"])[0])
            npy_matches = glob.glob(os.path.join(src_amap_dir, f"*_{orig_stem}.npy"))
            for npy_src in npy_matches:
                link_or_copy_file(
                    npy_src,
                    os.path.join(amap_dir, os.path.basename(npy_src)),
                    use_symlink=use_symlink,
                )
        else:
            skipped_count += 1

        updated_records.append(row)

    # 計算本包專屬的統計指標
    total = len(updated_records)
    non_golden = [r for r in updated_records if not r["Status"].startswith("GOLDEN")]
    non_golden_normal = [
        r for r in non_golden if not os.path.basename(r["Original_Filename"]).startswith("escape_")
    ]
    escape_defects = [
        r for r in updated_records if os.path.basename(r["Original_Filename"]).startswith("escape_")
    ]

    target_normal = non_golden_normal if non_golden_normal else (non_golden if non_golden else updated_records)

    if target_normal:
        norm_scores = [float(r["Max_Score"]) for r in target_normal]
        raw_scores = [float(r.get("Raw_Max_Score", r["Max_Score"])) for r in target_normal]
        filt_scores = [float(r.get("Filtered_Max_Score", r["Max_Score"])) for r in target_normal]
        max_s = float(np.max(norm_scores))
        p99_s = float(np.percentile(norm_scores, 99))
        p95_s = float(np.percentile(norm_scores, 95))
        median_s = float(np.median(norm_scores))
        mean_s = float(np.mean(norm_scores))
        std_s = float(np.std(norm_scores))
        raw_max_s, raw_mean_s = float(np.max(raw_scores)), float(np.mean(raw_scores))
        filt_max_s, filt_mean_s = float(np.max(filt_scores)), float(np.mean(filt_scores))
    else:
        max_s = p99_s = p95_s = median_s = mean_s = std_s = 0.0
        raw_max_s = raw_mean_s = filt_max_s = filt_mean_s = 0.0

    m1std = mean_s + 1.0 * std_s
    m2std = mean_s + 2.0 * std_s
    m3std = mean_s + 3.0 * std_s

    fp_normal = [r for r in target_normal if "ANOMALY" in r["Status"]]
    caught_escape = [r for r in escape_defects if "ANOMALY" in r["Status"]]
    fpr_normal = (len(fp_normal) / len(target_normal) * 100.0) if target_normal else 0.0
    escape_recall = (len(caught_escape) / len(escape_defects) * 100.0) if escape_defects else 0.0

    # 輸出本包的 escape_ranking_report.txt
    report_txt = os.path.join(pack_out_dir, "escape_ranking_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"=== FoundAD Split Eval Report — Pack #{pack_idx} ({ os.path.basename(pack_out_dir) }) ===\n")
        f.write(
            f"Source Run Dir: {src_run_dir}\n"
            f"Score Mode: {header_meta['score_mode']} | Threshold Mode: {header_meta['thresh_mode']} | "
            f"Effective Threshold: {eff_thresh:.4f}\n"
            f"Pack Total Images: {total} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_count}\n"
            f"Normal False Positives (FPR): {len(fp_normal)}/{len(target_normal)} ({fpr_normal:.2f}%) | "
            f"Escape Caught (Recall): {len(caught_escape)}/{len(escape_defects)} ({escape_recall:.2f}%)\n"
            f"Non-Golden Normal Score Dist: Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | "
            f"Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
            f"Ablation Compare (Normal)   : Raw_Max={raw_max_s:.4f}, Raw_Mean={raw_mean_s:.4f} vs "
            f"Filtered_Max={filt_max_s:.4f}, Filtered_Mean={filt_mean_s:.4f}\n"
            f"Threshold Ref (Mean+k*Std)  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
            f"Overfit Diagnostic          : {header_meta['overfit_diag']}\n"
        )
        f.write("-" * 155 + "\n")
        f.write(
            f"{'P_Rank':<6} | {'G_Rank':<6} | {'Original_Filename':<26} | {'Status':<19} | "
            f"{'Max_Score':<9} | {'Raw_Score':<9} | {'Filt_Score':<10} | {'Delta':<8} | {'BBoxes':<6} | {'Saved_Anomaly_Filename'}\n"
        )
        f.write("-" * 155 + "\n")
        for r in updated_records:
            raw_v = float(r.get("Raw_Max_Score", r["Max_Score"]))
            flt_v = float(r.get("Filtered_Max_Score", r["Max_Score"]))
            dlt_v = float(r.get("Filter_Delta", flt_v - raw_v))
            f.write(
                f"{r['Pack_Rank']:<6} | {r['Global_Rank']:<6} | {r['Original_Filename']:<26} | "
                f"{r['Status']:<19} | {float(r['Max_Score']):<9.4f} | {raw_v:<9.4f} | "
                f"{flt_v:<10.4f} | {dlt_v:<+8.4f} | {r['Num_BBoxes']:<6} | {r['Saved_Anomaly_Filename']}\n"
            )

    # 輸出本包的 escape_ranking_report.csv
    report_csv = os.path.join(pack_out_dir, "escape_ranking_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pack_Rank", "Global_Rank", "Original_Filename", "Status", "Max_Score",
            "Raw_Max_Score", "Filtered_Max_Score", "Filter_Delta",
            "TopK_Mean_Score", "Num_BBoxes", "Saved_Anomaly_Filename", "Resolution"
        ])
        for r in updated_records:
            writer.writerow([
                r["Pack_Rank"],
                r["Global_Rank"],
                r["Original_Filename"],
                r["Status"],
                r["Max_Score"],
                r.get("Raw_Max_Score", r["Max_Score"]),
                r.get("Filtered_Max_Score", r["Max_Score"]),
                r.get("Filter_Delta", "0.000000"),
                r.get("TopK_Mean_Score", r["Max_Score"]),
                r["Num_BBoxes"],
                r["Saved_Anomaly_Filename"],
                r.get("Resolution", ""),
            ])

    return {
        "pack_idx": pack_idx,
        "pack_name": os.path.basename(pack_out_dir),
        "pack_dir": pack_out_dir,
        "report_csv": report_csv,
        "report_txt": report_txt,
        "total": total,
        "num_golden": total - len(non_golden),
        "num_normal": len(target_normal),
        "num_escape": len(escape_defects),
        "caught_escape": len(caught_escape),
        "escape_recall": escape_recall,
        "fp_normal": len(fp_normal),
        "fpr_normal": fpr_normal,
        "anomaly_exported": anomaly_count,
        "max_s": max_s,
        "p99_s": p99_s,
        "p95_s": p95_s,
        "median_s": median_s,
        "mean_s": mean_s,
        "std_s": std_s,
        "m1std": m1std,
        "m2std": m2std,
        "m3std": m3std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="將 FoundAD 全部推論結果依「每包張數 (--counts)」或「比例 (--ratios)」拆分為多份 (預設3份) 獨立 Eval 結果。"
    )
    parser.add_argument(
        "--src_dir", type=str, default="./output/2_fab_no_gt",
        help="來源結果目錄：可指定時間戳記目錄 (如 ./output/2_fab_no_gt/2026xxxx) 或父目錄 (自動抓最新一筆)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="",
        help="拆分後輸出目錄 (預設存在該次結果目錄下的 split_eval_3packs/)"
    )
    parser.add_argument(
        "--num_packs", type=int, default=3,
        help="預設拆分成幾份 Eval 結果 (預設: 3)"
    )
    parser.add_argument(
        "--counts", type=str, default="",
        help="依【每包張數】拆分：可填單一數字如 '30' (代表每包30張，共拆 num_packs 包) 或逗號分隔如 '30,30,20'"
    )
    parser.add_argument(
        "--ratios", type=str, default="",
        help="依【比例】拆分：例如 '1:1:1' (均分3份)、'2:1:1' 或 '0.5,0.25,0.25' (若未指定 counts 也未指定 ratios，預設按 1:1:1 拆為3份)"
    )
    parser.add_argument(
        "--split_order", type=str, default="stratified",
        choices=["stratified", "random", "rank", "filename"],
        help="拆分抽樣方式：stratified(分層抽樣，確保3份Eval中異常/正常比例一致)、random(純隨機)、rank(依分數高低分段)、filename(依檔名順序)"
    )
    parser.add_argument(
        "--exclude_golden", action="store_true",
        help="若設定此旗標，拆分前會先排除作為訓練集的 Golden 圖片，只對非 Golden 測試圖進行拆分"
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="使用軟連結 (Symlink) 取代實體複製異常紅框圖與 .npy，節省硬碟空間並秒速完成"
    )
    parser.add_argument("--seed", type=int, default=42, help="隨機與分層抽樣的亂數種子")
    args = parser.parse_args()

    src_run_dir = find_latest_run_dir(args.src_dir)
    src_csv = os.path.join(src_run_dir, "escape_ranking_report.csv")
    src_txt = os.path.join(src_run_dir, "escape_ranking_report.txt")

    header_meta = parse_header_metadata(src_txt)
    with open(src_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        raise ValueError(f"報表 {src_csv} 內沒有任何紀錄！")

    if args.exclude_golden:
        records = [r for r in records if not r["Status"].startswith("GOLDEN")]

    total_avail = len(records)
    pack_sizes = compute_pack_sizes(
        total_items=total_avail,
        num_packs=args.num_packs,
        counts_str=args.counts,
        ratios_str=args.ratios,
    )

    packs = split_records(
        records=records,
        pack_sizes=pack_sizes,
        split_order=args.split_order,
        seed=args.seed,
    )

    out_root = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(src_run_dir, f"split_eval_{len(packs)}packs")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    mode_desc = f"每包張數 (--counts={args.counts})" if args.counts else f"比例 (--ratios={args.ratios or '1:1:1'})"
    print("\n" + "=" * 95)
    print(f" 📦 [FoundAD 推論結果拆分器] 來源目錄: {src_run_dir}")
    print(f" 🔧 拆分模式: {mode_desc} | 排序/抽樣策略: {args.split_order} | 有效門檻: {header_meta['eff_thresh']}")
    print(f" 📊 總處理張數: {total_avail} 張 -> 分配為 {len(packs)} 份 Eval: {pack_sizes} (總計分配 {sum(pack_sizes)} 張)")
    print("=" * 95)

    summaries = []
    for i, pack_recs in enumerate(packs, 1):
        pack_dir = os.path.join(out_root, f"eval_pack_{i}")
        s = generate_pack_eval_report(
            pack_idx=i,
            pack_records=pack_recs,
            src_run_dir=src_run_dir,
            pack_out_dir=pack_dir,
            header_meta=header_meta,
            use_symlink=args.symlink,
        )
        summaries.append(s)

    # 輸出三份 Eval 的並列總表 (TXT 與 CSV)
    overview_txt = os.path.join(out_root, "all_3_evals_comparison.txt")
    overview_csv = os.path.join(out_root, "all_3_evals_comparison.csv")

    lines = [
        "=" * 95,
        f" 🎯 [最終 {len(summaries)} 份 Eval 結果並列總覽 (All Split Evaluations Comparison)]",
        "-" * 95,
        f"  📂 來源完整推論目錄 : {src_run_dir}",
        f"  📂 拆分結果根目錄   : {out_root}",
        f"  🎚️ 採用有效門檻     : {header_meta['eff_thresh']} (Mode: {header_meta['thresh_mode']}, Score: {header_meta['score_mode']})",
        "-" * 95,
        f"{'Eval包名稱':<14} | {'總張數':<6} | {'非Golden正常':<12} | {'正常誤報(FPR)':<16} | {'抓出Escape瑕疵':<16} | {'Normal Mean±Std':<17} | {'mean+3*std'}",
        "-" * 95,
    ]
    for s in summaries:
        fpr_str = f"{s['fp_normal']}/{s['num_normal']} ({s['fpr_normal']:.1f}%)"
        esc_str = f"{s['caught_escape']}/{s['num_escape']} ({s['escape_recall']:.1f}%)" if s["num_escape"] > 0 else "0/0 (N/A)"
        ms_str = f"{s['mean_s']:.4f}±{s['std_s']:.4f}"
        lines.append(
            f"{s['pack_name']:<14} | {s['total']:<6} | {s['num_normal']:<12} | {fpr_str:<16} | {esc_str:<16} | {ms_str:<17} | {s['m3std']:.4f}"
        )
    lines.append("-" * 95)

    for s in summaries:
        lines.extend([
            f"  🔹 [{s['pack_name']}] (共 {s['total']} 張, 輸出異常紅框圖 {s['anomaly_exported']} 張)",
            f"     ├── 正常圖分數分佈 : Max={s['max_s']:.4f} | P99={s['p99_s']:.4f} | P95={s['p95_s']:.4f} | Median={s['median_s']:.4f} | Mean={s['mean_s']:.4f} | Std={s['std_s']:.4f}",
            f"     ├── 門檻參考值     : mean+1*std={s['m1std']:.4f} | mean+2*std={s['m2std']:.4f} | mean+3*std={s['m3std']:.4f}",
            f"     ├── 評估報表 (TXT) : {s['report_txt']}",
            f"     └── 異常圖片資料夾 : {os.path.join(s['pack_dir'], 'anomaly_images')}",
        ])
    lines.append("=" * 95)

    full_summary_text = "\n".join(lines)
    print(full_summary_text)

    with open(overview_txt, "w", encoding="utf-8") as f:
        f.write(full_summary_text + "\n")

    with open(overview_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Eval_Pack", "Total_Images", "Golden_Images", "NonGolden_Normal_Images",
            "Normal_FP_Count", "Normal_FPR_Percent", "Escape_Total", "Escape_Caught",
            "Escape_Recall_Percent", "Normal_Max", "Normal_P99", "Normal_P95",
            "Normal_Median", "Normal_Mean", "Normal_Std", "Mean_Plus_1Std",
            "Mean_Plus_2Std", "Mean_Plus_3Std", "Pack_Dir"
        ])
        for s in summaries:
            writer.writerow([
                s["pack_name"], s["total"], s["num_golden"], s["num_normal"],
                s["fp_normal"], f"{s['fpr_normal']:.2f}", s["num_escape"], s["caught_escape"],
                f"{s['escape_recall']:.2f}", f"{s['max_s']:.6f}", f"{s['p99_s']:.6f}", f"{s['p95_s']:.6f}",
                f"{s['median_s']:.6f}", f"{s['mean_s']:.6f}", f"{s['std_s']:.6f}", f"{s['m1std']:.6f}",
                f"{s['m2std']:.6f}", f"{s['m3std']:.6f}", s["pack_dir"]
            ])


if __name__ == "__main__":
    main()
