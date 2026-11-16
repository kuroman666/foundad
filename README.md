# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 產線純推論與拆分評估指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為**「已完成模型訓練、針對不同製程/資料集類型使用對應權重檔與固定門檻進行無標註（No Ground Truth）晶圓盲池抓漏推論」**的產線部署場景設計。

系統透過一鍵主控腳本 `run_inside_infer_eval_old.sh` 串聯 **高解析度切塊推論 (`3_infer_fab_no_gt_old.py`)** 與 **評估結果多包拆分 (`4_split_eval_results.py`)**，支援：
- **多資料集類型模型與固定門檻對接**：支援透過腳本內變數 (`CUSTOM_CKPT_PATH` / `CUSTOM_THRESHOLD`) 或命令列參數，針對三種（或多種）不同類型的資料集分別指定已訓練好的 `.pth.tar` 權重檔與固定門檻（未指定時自動回退至預設對應權重與預設門檻 `0.18`）。
- **高解析度切塊推論與 RAM 即時縫合（0 碎片殘留）**：針對高解析度晶圓圖自動進行重疊切塊推論（支援切塊 PNG 快取加速），於 RAM 內以 2D 餘弦平滑權重（Raised-Cosine Window）無縫拼接回全解析度，並**立即自動銷毀所有子圖塊 `.npy` 碎片**。
- **四合一防誤報熱力圖純化器**：整合「矩形邊框 Padding 抑制 + 圓形晶圓 Bevel/Notch 邊緣餘弦衰減 + 灰階形態學開運算微塵濾波 + 跨 Golden 結構複雜區容差補償」，精準定位紅色 Bounding Box 並過濾正常晶圓。
- **自動多包拆分與並列總覽報表**：自動將推論結果依比例或指定張數拆分為多份獨立 Eval 報表（`eval_pack_1` ~ `eval_pack_N`），並同步匯出單池與跨批次總覽表。

---

## 1. 環境與公司內斷網（Offline）部署準備

### 1.1 基礎環境需求
- **作業系統**：Linux (Ubuntu 20.04 / 22.04+)
- **Python 版本**：Python 3.10 ~ 3.12
- **GPU 硬體**：NVIDIA GPU 伺服器（如 H100 / A100 / RTX 系列）

### 1.2 套件安裝
```bash
# 1. 建立並啟動虛擬環境
conda create -n foundad python=3.10 -y
conda activate foundad

# 2. 安裝 PyTorch (請依機器 CUDA 版本安裝)
pip install torch torchvision torchaudio

# 3. 安裝專案依賴套件
pip install -r requirements.txt
```

### 1.3 公司內完全斷網（Offline）注意事項
1. **本地 DINO 架構快取 (`./hub_cache/`)**：專案根目錄已包含 `./hub_cache/facebookresearch_dinov3_main`，執行時會直接從本地快取載入模型架構，無須連外網下載 GitHub。
2. **斷網套件安裝**：若公司環境無法連網，請先於外部機器執行 `pip download -r requirements.txt -d ./offline_wheels`，帶入公司後執行：
   ```bash
   pip install --no-index --find-links=./offline_wheels -r requirements.txt
   ```
3. **執行權限確認**：首次拉取專案後，請賦予執行腳本權限：
   ```bash
   chmod +x run_inside_infer_eval_old.sh
   ```

---

## 2. 專案資料夾與核心檔案結構

```text
FoundAD/
├── foundad/                                  # [核心] FoundAD 模型與推論底層原始碼
│   ├── configs/                              # Hydra 設定檔 (含 app/test.yaml 等)
│   ├── src/
│   │   ├── AD.py                             # 無 GT 熱力圖批次推論與 .npy 匯出 Hook
│   │   └── foundad.py                        # 支援離線載入本地 DINOv2 / DINOv3 權重之視覺模組
│   └── main.py                               # FoundAD 主入口程式 (支援 FAB_MASTER_PORT 動態分配)
├── hub_cache/                                # [離線快取] 本地 DINOv3 架構定義檔
├── datasets/                                 # [資料集與快取區]
│   ├── <你的測試盲池資料夾>/                  # (Input) 待推論之無 GT 晶圓圖片資料夾
│   ├── <DATA_NAME>_tiled_pool/               # (自動生成) 測試盲池重疊切塊 PNG 快取
│   └── <DATA_NAME>_cached_tile_manifest.json # (自動生成) 切塊座標映射快取
├── logs/                                     # [已訓練模型權重區] 存放三種類型資料集訓練好的權重
│   └── <DATA_NAME>/<DIY_NAME>/
│       ├── train-step*.pth.tar               # 已訓練完成之模型權重檔
│       └── pretrained.pth.tar                # 推論時自動對接之權重別名檔
├── wafer_escape_results/                     # [唯一結果輸出根目錄]
│   ├── inside_no_gt_old_<UNIQUE_TAG>/        # 各資料集獨立推論與拆分評估結果目錄
│   │   └── <YYYYMMDD_HHMMSS>/                # 每次執行獨立時間戳記資料夾 (絕不覆蓋歷史紀錄)
│   │       ├── anomaly_images/               # 異常紅框三合一對照圖 & full_res_bbox_only/
│   │       ├── amap_npy/                     # 異常圖片之全解析度熱力圖矩陣 (.npy)
│   │       ├── escape_ranking_report.txt/csv # 完整無 GT 抓漏排名與消融對照報表
│   │       ├── split_eval_3packs/            # Step 2 拆分後之獨立 Eval 包與跨包比較總表
│   │       └── log.txt                       # 本次執行完整終端機日誌
│   ├── inside_infer_eval_old_summary_<時間戳>.txt # 本次執行之彙整總表
│   └── latest_inside_infer_eval_old_summary.txt/.csv # 最新一次執行之跨盲池總覽報表
├── 3_infer_fab_no_gt_old.py                  # [核心程式 1] 高解析度切塊推論、RAM 縫合與四合一防誤報畫框
├── 4_split_eval_results.py                   # [核心程式 2] 推論結果自動拆分與跨包比較報表生成器
├── run_inside_infer_eval_old.sh              # [產線主控腳本] 一鍵執行推論 + 結果拆分評估 (支援自訂權重與門檻)
└── requirements.txt                          # Python 環境依賴套件清單
```

---

## 3. 操作指南：針對三種類型資料集進行推論與評估

進入專案根目錄並啟動環境：
```bash
conda activate foundad
cd /path/to/FoundAD
```

`run_inside_infer_eval_old.sh` 支援**「命令列直接傳參（推薦）」**與**「修改腳本參數區」**兩種方式，腳本會自動辨識參數類型（資料夾路徑、`.pth.tar` 權重檔路徑、數字門檻）。

### 3.1 方式 A：直接從命令列傳入「資料集路徑 + 權重檔路徑 + 固定門檻」（最推薦）

適合針對三種不同類型的資料集，各自指定專屬模型權重與固定門檻執行：

```bash
# 【類型 1 資料集】指定模型 1 權重檔 + 固定門檻 0.32
bash run_inside_infer_eval_old.sh /path/to/dataset_type1 ./logs/model_type1/dinov3/train-step2000.pth.tar 0.32

# 【類型 2 資料集】指定模型 2 權重檔 + 固定門檻 0.28
bash run_inside_infer_eval_old.sh /path/to/dataset_type2 ./logs/model_type2/dinov3/train-step2000.pth.tar 0.28

# 【類型 3 資料集】指定模型 3 權重檔 + 固定門檻 0.22
bash run_inside_infer_eval_old.sh /path/to/dataset_type3 ./logs/model_type3/dinov3/train-step2000.pth.tar 0.22
```

> **使用背景執行（防止 SSH 斷線中斷推論）：**
> ```bash
> nohup bash run_inside_infer_eval_old.sh /path/to/dataset_type1 ./logs/model_type1/dinov3/train-step2000.pth.tar 0.32 > infer_type1.log 2>&1 &
> ```

### 3.2 方式 B：不傳權重或門檻，自動抓取預設值

若未在命令列或腳本內指定權重與門檻，腳本會自動：
1. 依據盲池資料夾名稱對齊 `./logs/fab_inside_old_<UNIQUE_TAG>_4shot/`（或 `./logs/fab_inside_<UNIQUE_TAG>_4shot/`）下的預設訓練權重。
2. 套用腳本預設固定門檻 `DEFAULT_THRESHOLD="0.18"`。

```bash
# 自動抓對應預設權重 + 預設門檻 0.18
bash run_inside_infer_eval_old.sh ./datasets/fab_wafer_pool
```

### 3.3 方式 C：在腳本內固定寫死參數

打開 `run_inside_infer_eval_old.sh` 修改頂部 `[可自由修改的參數區]`：
```bash
# 1. 指定訓練好的權重檔 (.pth.tar) 或權重資料夾路徑 (留空 "" 則抓預設對應權重)
CUSTOM_CKPT_PATH="./logs/fab_inside_old_TypeA_4shot/dinov3fab/train-step2000.pth.tar"

# 2. 指定固定門檻數值 (留空 "" 且命令列未傳入數字時，自動抓 DEFAULT_THRESHOLD="0.18")
CUSTOM_THRESHOLD="0.32"
DEFAULT_THRESHOLD="0.18"
```
設定優先順序為：**命令列傳入參數 > 腳本內 `CUSTOM_*` 變數 > 系統預設值**。

---

## 4. 關鍵可調參數說明 (`run_inside_infer_eval_old.sh`)

| 參數名稱 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `CUSTOM_CKPT_PATH` | `""` | 指定已訓練好的 `.pth.tar` 權重檔或權重資料夾路徑；腳本會自動解析出對應的 `DATA_NAME`、`DIY_NAME` 與 `STEP`。 |
| `CUSTOM_THRESHOLD` | `""` | 指定固定異常判定門檻（例如 `"0.32"`）；留空時使用 `DEFAULT_THRESHOLD`（`"0.18"`）。 |
| `THRESH_MODE` | `"fixed"` | 門檻判定模式。目前產線預設固定為 `"fixed"`；亦支援 `"hybrid"`、`"mean_std"`、`"percentile"`、`"top_k"`。 |
| `TILE_SIZE` / `TILE_OVERLAP` | `512` / `64` | 推論切塊像素大小與重疊寬度（須與訓練該模型時的設定一致，以確保特徵尺度對齊與快取命中）。 |
| `INFER_BATCH_SIZE` | `16` | GPU 推論子圖塊的 Mini-Batch 大小（可依顯卡 VRAM 調整為 `16` 或 `32` 加速推論）。 |
| `FORCE_INFER` | `""` | 設為 `"--force_infer"` 可忽略既有測試池切塊 PNG 快取、強制重新切塊推論。 |
| `WAFER_EDGE_MODE` | `"auto"` | 圓形晶圓邊緣（Bevel/Notch）偽影抑制模式：`"auto"`（自動偵測圓形工件）、`"on"`（強制開啟）、`"off"`（關閉）。 |
| `WAFER_EDGE_MARGIN` | `0.04` | 晶圓外圈平滑衰減抑制寬度佔半徑比例（`0.04` 代表外圈 4% 半徑）。 |
| `RUN_SPLIT_EVAL` | `1` | `1`：推論完成後自動執行 `4_split_eval_results.py` 拆分為多包 Eval 報表；`0`：僅執行推論。 |
| `NUM_PACKS` | `3` | Step 2 評估結果要拆分的包數（預設均分為 `3` 包，採 `stratified` 分層抽樣）。 |

---

## 5. 輸出報表與結果檢驗說明 (`./wafer_escape_results/`)

每次執行完成後，結果會儲存於 `./wafer_escape_results/inside_no_gt_old_<UNIQUE_TAG>/<YYYYMMDD_HHMMSS>/`：

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `anomaly_images/` | **異常紅框三合一對照圖（僅輸出分數 >= 門檻之異常晶圓）：**<br>檔名為 `Rank001_Score0.3450_Box01_<原檔名>.png`（依異常分數由高至低排列），由左至右依序為 `[原始圖 -> 純化熱力疊圖 -> 紅色 Bounding Box 瑕疵定位框]`。 |
| `anomaly_images/full_res_bbox_only/` | **1:1 原始全解析度紅框定位圖：**<br>維持原圖 100% 原始解析度（不縮放），僅畫上紅色瑕疵框與分數標籤，供工程師放大檢視微米級細節。 |
| `escape_ranking_report.txt` | **無 GT 抓漏總排名與診斷報表（純文字版）：**<br>包含有效門檻、盲池正常核心分數分佈（Max / P99 / P95 / Median / Mean / Std）、濾波前後消融對照（Raw vs Filtered）、`mean + 1/2/3*std` 參考值，以及所有圖片的排名明細。 |
| `escape_ranking_report.csv` | 同上之完整 CSV 報表（含 `Max_Score`、`Raw_Max_Score`、`Filtered_Max_Score`、`Num_BBoxes`、`Resolution`），可直接以 Excel 開啟篩選。 |
| `amap_npy/` | 超過門檻之異常圖片全解析度熱力圖矩陣（`.npy`）；所有子圖塊碎片皆已於 RAM 拼接後立即銷毀（0 殘留）。 |
| `split_eval_<N>packs/` | **獨立拆分評估包（`eval_pack_1` ~ `eval_pack_N`）與並列比較總表：**<br>內含各包獨立重排報表、對應異常紅框圖軟連結，以及跨包比較總表 `all_<N>_evals_comparison.txt/.csv`。 |
| `latest_inside_infer_eval_old_summary.txt/.csv` | 位於 `./wafer_escape_results/` 根目錄，彙整最新一次執行之各盲池檢出率、有效門檻與正常分數分佈總覽。 |

---

## 6. 常見問題排錯（Troubleshooting）

- **Q1：同時跑三個不同資料集時，可以開三個終端機平行（背景同時）執行嗎？**
  - **若三個資料集各自使用不同的權重資料夾**（例如 `./logs/model_type1/...`、`./logs/model_type2/...`、`./logs/model_type3/...`）：**完全可以同時平行執行**，各資料集的快取與輸出目錄皆完全獨立。
  - **若兩個資料集共用同一個權重資料夾**：請依序執行（等前一個跑完再跑下一個），避免 `3_infer_fab_no_gt_old.py` 同時寫入該資料夾下的 `pretrained.pth.tar` 發生衝突。
- **Q2：如何確認該資料集是否成功載入正確的權重與固定門檻？**
  - 啟動腳本後，終端機與 `log.txt` 最上方會明確印出解析結果，請核對以下兩行：
    ```text
    [🎯 指定權重載入] 路徑: ./logs/model_type1/dinov3/train-step2000.pth.tar
       -> 自動解析 DATA_NAME=model_type1 | DIY_NAME=dinov3 | STEP=2000
    門檻設定 : THRESH_MODE=fixed | THRESHOLD=0.32
    ```
- **Q3：如何強制重新切塊測試盲池？**
  - 當測試盲池圖片有增減時，系統會透過資料夾內所有檔案的 MD5 指紋自動偵測變動並重新切塊；若你修改了影像內容但檔名與大小未變，可將 `run_inside_infer_eval_old.sh` 中的 `FORCE_INFER="--force_infer"` 開啟後執行。