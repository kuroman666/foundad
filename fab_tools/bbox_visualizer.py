# 檔案位置: FoundAD/fab_tools/bbox_visualizer.py
import os
import csv
import cv2
import numpy as np

def compute_dynamic_thresholds(anomaly_maps_dict, golden_filenames, sigma_k=3.5, min_area_ratio=0.0005):
    """
    利用 Golden Images 與整批初篩正常池的統計分佈，決定絕對異常像素門檻 (Pixel Threshold)。
    避免正常晶圓因為相對正規化而被亂畫框！
    """
    golden_vals = []
    all_vals = []
    for fname, amap in anomaly_maps_dict.items():
        all_vals.append(amap.flatten())
        if fname in golden_filenames:
            golden_vals.append(amap.flatten())

    all_pixels = np.concatenate(all_vals)
    if len(golden_vals) > 0:
        ref_pixels = np.concatenate(golden_vals)
    else:
        # 取前 80% 低分像素作為正常背景分佈估計
        p80 = np.percentile(all_pixels, 80)
        ref_pixels = all_pixels[all_pixels <= p80]

    mean_val = float(np.mean(ref_pixels))
    std_val = float(np.std(ref_pixels)) + 1e-8
    pixel_thresh = max(mean_val + sigma_k * std_val, float(np.percentile(ref_pixels, 99.5)))
    vis_max = max(float(np.percentile(all_pixels, 99.95)), pixel_thresh * 1.25)

    return pixel_thresh, mean_val, vis_max, min_area_ratio

def draw_defect_bboxes_and_save(
    img_path, amap, pixel_thresh, vis_min, vis_max, save_path, min_area_ratio=0.0005
):
    """
    輸入原圖路徑與 2D Anomaly Map (H, W)，自動框出瑕疵位置並儲存三合一檢視圖：
    [左: 原始晶圓圖 | 中: 異常熱力圖疊圖 | 右: 紅色 Bounding Box 瑕疵定位圖]
    回傳: (整張圖異常分數, 偵測到的瑕疵框數量, 瑕疵框座標列表 [x, y, w, h])
    """
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        raise ValueError(f"無法讀取圖片: {img_path}")
    h, w = orig_bgr.shape[:2]

    # 1. 將 Anomaly Map 縮放至原圖尺寸
    amap_resized = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

    # 整張圖的異常分數：取 Top 0.1% 最異常像素的平均值（比單一最大值更抗噪點）
    top_k_pixels = max(1, int(h * w * 0.001))
    image_score = float(np.mean(np.sort(amap_resized.flatten())[-top_k_pixels:]))

    # 2. 產生異常熱力圖疊圖 (使用固定全局基準 vis_min ~ vis_max，確保顏色深淺有跨圖可比性)
    norm_map = np.clip((amap_resized - vis_min) / (vis_max - vis_min + 1e-8), 0, 1)
    heatmap_u8 = (norm_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(orig_bgr, 0.55, heatmap_color, 0.45, 0)

    # 3. 二值化分割並用 OpenCV 找出瑕疵連通域與 Bounding Box
    binary_mask = (amap_resized >= pixel_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # 連接鄰近碎點

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = orig_bgr.copy()
    min_area = h * w * min_area_ratio
    bboxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 過濾極小孤立噪點
        x, y, bw, bh = cv2.boundingRect(cnt)
        roi_score = float(np.max(amap_resized[y:y+bh, x:x+bw]))
        bboxes.append((x, y, bw, bh, roi_score))

        # 畫出細緻瑕疵輪廓(黃色) + 外接矩形紅框(紅色)
        cv2.drawContours(bbox_img, [cnt], -1, (0, 255, 255), 1)
        cv2.rectangle(bbox_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(
            bbox_img, f"{roi_score:.2f}", (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

    # 4. 在圖片上方加上標題列並拼接成三合一圖
    status_str = f"SUSPECT ESCAPE (Boxes: {len(bboxes)})" if len(bboxes) > 0 else "CLEAN"
    title_color = (0, 0, 255) if len(bboxes) > 0 else (0, 180, 0)

    panel = np.hstack([orig_bgr, overlay_bgr, bbox_img])
    banner = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    header_text = (
        f"File: {os.path.basename(img_path)} | Score: {image_score:.4f} | "
        f"Thresh: {pixel_thresh:.4f} | Status: {status_str}"
    )
    cv2.putText(banner, header_text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)
    final_vis = np.vstack([banner, panel])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final_vis)
    return image_score, len(bboxes), bboxes

def export_no_gt_reports(
    pool_dir, anomaly_maps_dict, golden_filenames, out_dir, sigma_k=3.5, save_top_k=100
):
    """
    批次處理整批無標註圖片的 Anomaly Maps，產生排序 .txt/.csv 與畫好紅框的圖片。
    """
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "bbox_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    pixel_thresh, vis_min, vis_max, min_area_ratio = compute_dynamic_thresholds(
        anomaly_maps_dict, golden_filenames, sigma_k=sigma_k
    )

    records = []
    for fname, amap in anomaly_maps_dict.items():
        img_path = os.path.join(pool_dir, fname)
        if not os.path.exists(img_path):
            continue
        temp_save = os.path.join(vis_dir, f"TEMP_{fname}")
        score, num_boxes, boxes = draw_defect_bboxes_and_save(
            img_path, amap, pixel_thresh, vis_min, vis_max, temp_save, min_area_ratio
        )
        records.append({
            "filename": fname,
            "score": score,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "temp_vis": temp_save
        })

    # 依異常分數由高到低排序（最可疑的 Escape 排第一名）
    records.sort(key=lambda x: x["score"], reverse=True)

    txt_path = os.path.join(out_dir, "escape_ranking_report.txt")
    csv_path = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(txt_path, "w", encoding="utf-8") as f_txt, \
         open(csv_path, "w", newline="", encoding="utf-8") as f_csv:

        writer = csv.writer(f_csv)
        writer.writerow(["Rank", "Filename", "Anomaly_Score", "Defect_Boxes_Count", "Boxes_XYWH"])

        f_txt.write("========================================================================\n")
        f_txt.write(f"Wafer Escape Screening Report (No-GT Mode) | Pixel Thresh: {pixel_thresh:.4f}\n")
        f_txt.write("========================================================================\n")
        f_txt.write(f"{'Rank':<6} | {'Filename':<28} | {'Score':<10} | {'Boxes':<6} | {'BBox Coordinates (x,y,w,h)'}\n")
        f_txt.write("-" * 88 + "\n")

        for rank, rec in enumerate(records, 1):
            box_str = "; ".join([f"({b[0]},{b[1]},{b[2]},{b[3]})" for b in rec["boxes"]]) or "None"
            f_txt.write(f"{rank:<6} | {rec['filename']:<28} | {rec['score']:<10.4f} | {rec['num_boxes']:<6} | {box_str}\n")
            writer.writerow([rank, rec["filename"], f"{rec['score']:.6f}", rec["num_boxes"], box_str])

            # 將排名前 save_top_k 或有框出瑕疵的圖片加上 Rank 編號保留，其餘暫存圖刪除以節省空間
            final_vis_name = os.path.join(vis_dir, f"Rank{rank:04d}_Box{rec['num_boxes']}_{rec['filename']}")
            if rank <= save_top_k or rec["num_boxes"] > 0:
                os.replace(rec["temp_vis"], final_vis_name)
            else:
                if os.path.exists(rec["temp_vis"]):
                    os.remove(rec["temp_vis"])

    print(f"\n[完成] 抓漏排序報表 (.txt) 已存至: {txt_path}")
    print(f"[完成] 抓漏排序報表 (.csv) 已存至: {csv_path}")
    print(f"[完成] 標註紅框的瑕疵定位圖已存至: {vis_dir}")
