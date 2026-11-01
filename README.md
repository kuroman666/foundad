# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 完整建置與操作指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為「晶圓廠工程師初篩為正常品、完全無標註（No Ground Truth）之高解析度 Wafer 影像池」設計。系統支援三大核心驗證與部署模式：

* **模式一：有 GT 公開資料集驗證模式（`1_run_public_with_gt.py` / 公司外基準測試）：** 自動完成少樣本抽樣、訓練，並將終端機之 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC` 分數與各階段耗時匯出為 `.txt` 與 `.csv` 報表。
* **模式二：無 GT 晶圓廠盲池抓漏模式（`2_run_fab_no_gt.py` & `run_inside.sh` / 公司內即插即用）：** 自動透過 DINO 特徵中心度（Medoid Centrality）從無標註圖片池挑出 4 張最純淨的 Golden Image -> 高解析度無損切塊（Tiling）訓練流形預測器 -> 全解析度餘弦權重無縫拼接（Feathering Stitching） -> 晶圓圓周邊緣抑制與微塵濾波 -> 支援 5 種動態統計門檻自動框出可疑瑕疵位置（紅色 Bounding Box）並輸出異常排名表（正常圖片自動略過不輸出）。
* **模式三：100% 純正常資料集誤報率（FPR）專項驗證模式（`run_pure_normal_test.sh`）：** 針對完全無瑕疵（0 張異常）之純正常圖片池進行壓力測試，自動驗算整體誤報率（Overall FPR）、扣除 Golden 後誤報率，以及非 Golden 正常圖之分數分佈與 `mean + 1*std / 2*std / 3*std` 門檻參考值。

---

## 1. 環境、套件安裝（含公司內斷網環境準備）

### 1.1 基礎環境需求
* **作業系統：** Linux (Ubuntu 20.04 / 22.04+)
* **Python 版本：** Python 3.10 ~ 3.12
* **GPU 硬體：** 外部測試機 或 廠內伺服器（如 NVIDIA H100 / RTX 系列）

### 1.2 情況 A：在「有網路」環境安裝（或進公司前打包環境）
請依序執行以下指令建立 Conda 虛擬環境並安裝套件：

```bash
# 1. 建立並啟動虛擬環境
conda create -n foundad python=3.10 -y
conda activate foundad

# 2. 安裝 PyTorch (請依機器之 CUDA 版本選擇，例如 CUDA 12.1 / 12.4)
pip install torch torchvision torchaudio

# 3. 安裝專案所需核心套件
pip install hydra-core omegaconf pyyaml numpy pillow opencv-python matplotlib scikit-learn scipy tqdm transformers
```

### 1.3 情況 B：進公司「完全斷網（Offline）」環境的前置打包與安裝
由於公司內部通常無法使用 `pip install` 連外網，且 `torch.hub` 無法連上 GitHub 下載 DINO 模型定義檔，請在進公司前，先在外部機器執行以下 3 步打包動作：

#### Step 1：把 DINO 的 GitHub 架構快取打包進專案 `./hub_cache/`
```bash
# 在外部已跑過一次 FoundAD 的機器上執行：
mkdir -p ./hub_cache
cp -r ~/.cache/torch/hub/*dino* ./hub_cache/
```
> **註：** 若廠內使用的是 `dinov2_vitl14` 或 `dinov2_vitb14`，本專案 `foundad/src/dinov2/` 已內建完整架構，甚至不需依賴 `hub_cache` 即可離線運作；若使用 `dinov3`，則務必保留 `./hub_cache/facebookresearch_dinov3_main` 資料夾。

#### Step 2：預先下載離線套件包 (`offline_wheels/`) 或打包 Conda 環境
**方法一（推薦：打包 `.whl` 離線安裝包）：**
```bash
# 在外部有網路機器下載所有 .whl 檔
mkdir -p ./offline_wheels
pip download hydra-core omegaconf pyyaml numpy pillow opencv-python matplotlib scikit-learn scipy tqdm transformers -d ./offline_wheels

# 進公司後，在公司已經有 PyTorch 的環境中直接離線安裝：
pip install --no-index --find-links=./offline_wheels hydra-core omegaconf pyyaml numpy pillow opencv-python matplotlib scikit-learn scipy tqdm transformers
```

**方法二（使用 `conda-pack` 整包帶入）：**
```bash
conda install -c conda-forge conda-pack -y
conda pack -n foundad -o foundad_env.tar.gz
# 進公司後解壓縮至 ~/miniconda3/envs/foundad 並執行 conda activate foundad
```

#### Step 3：確認自訂類別的前景遮罩修補已生效 (`synthesis.py`)
原版 FoundAD 的 `foundad/src/utils/synthesis.py` 僅寫死 MVTec 與 VisA 的原生類別，遇到 `"wafer"` 等自訂類別會在 Epoch 1 報 `NotImplementedError`。請確認第 74 行已修補為回傳 `(H, W)` 維度的 `np.ones_like(img_gray, dtype=np.uint8)`（可執行以下指令自動檢查與修補）：

```bash
python -c '
p = "foundad/src/utils/synthesis.py"
txt = open(p).read()
old_1 = "raise NotImplementedError(\"Unsupported foreground segmentation category\")"
old_2 = "return np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)"
new_code = "return np.ones_like(img_gray, dtype=np.uint8)  # 正確回傳 (H, W) 全前景遮罩 (支援 wafer 等自訂類別)"
if old_1 in txt:
    open(p, "w").write(txt.replace(old_1, new_code))
    print("[OK] 已成功修補 synthesis.py！")
elif old_2 in txt:
    open(p, "w").write(txt.replace(old_2, new_code))
    print("[OK] 已將 synthesis.py 更新為正確的 (H, W) 維度遮罩！")
else:
    print("[OK] synthesis.py 已經是支援 wafer 自訂類別的正確狀態，無需修改。")
'
```

---

## 2. 專案完整資料夾結構（含自動生成目錄說明）

每次執行實驗時，系統皆會自動建立以當下時間命名的子資料夾（`<YYYYMMDD_HHMMSS>`），**絕不覆蓋歷史實驗結果**：

```text
FoundAD/
├── foundad/                                  # [核心] FoundAD 模型與訓練/推論原始碼
│   ├── configs/                              # Hydra 設定檔 (含 app/train_dinov3.yaml, app/test.yaml)
│   ├── src/
│   │   ├── AD.py                             # 評估與無 GT 熱力圖推論 (已內嵌 .npy 匯出與畫框 Hook)
│   │   ├── foundad.py                        # 支援離線載入本地 DINOv2 / DINOv3 權重之視覺模組
│   │   ├── train.py                          # 流形預測器 (Manifold Predictor) 訓練迴圈
│   │   ├── sample.py                         # 有 GT 模式之少樣本抽樣腳本
│   │   └── utils/synthesis.py                # CutPaste 異常合成模組 (已修正支援 wafer 等自訂類別)
│   └── main.py                               # FoundAD 主入口程式 (含 DDP 安全釋放機制)
├── fab_tools/                                # [晶圓廠專用外掛模組]
│   ├── auto_golden.py                        # 無 GT 自動計算特徵中心挑選 4 張純淨 Golden Image
│   └── bbox_visualizer.py                    # 輔助視覺化工具模組
├── hub_cache/                                # [離線快取] 存放 facebookresearch_dinov3_main 等本地架構檔
├── datasets/                                 # [資料集與自動生成訓練集專區]
│   ├── mvtec/                                # (Input) 原始 MVTec AD 資料集 (bottle, cable, capsule...)
│   ├── mvtec_4shot/                          # (自動生成) Task 1 有 GT 模式抽出的 4-shot 訓練集
│   ├── mvtec_sim_4shot/                      # (自動生成) Task 2 無 GT 模式挑出的 Golden 訓練圖塊與原圖備份
│   │   ├── train/<類別>/                     # 切分後的代表性高解析度訓練圖塊 (000_tile_y..._x....png)
│   │   └── golden_full_res_backup/<類別>/    # 自動挑選出的 4 張原始全解析度 Golden Image 備份
│   ├── mvtec_sim_4shot_tiled_pool/           # (自動生成) 無 GT 測試池切分後的重疊推論子圖塊
│   ├── sim_fab_capsule_pool/                 # (自動生成) 模擬無 GT 盲池 (含 clean_* 與隱藏 escape_* 瑕疵圖)
│   └── sim_pure_normal_<類別>_pool/          # (自動生成) 100% 純正常無瑕疵測試盲池 (供誤報率 FPR 驗證用)
├── logs/                                     # [模型權重區] 訓練自動產生的 .pth.tar 權重檔與 params.yaml
│   ├── mvtec_4shot/dinov3dbug/               # Task 1 有 GT 模式之訓練權重
│   ├── mvtec_sim_4shot/dinov3sim_escape/     # Task 2/3 外部無 GT 模擬與純正常測試之訓練權重
│   └── mvtec_fab_4shot/dinov3fab_h100/       # 公司內實測晶圓池之訓練權重
├── output/                                   # [外部測試唯一結果出口] 依任務與時間戳記分類存放
│   ├── 1_public_with_gt/                     # Task 1 有 GT 公開資料集評估結果
│   │   ├── <YYYYMMDD_HHMMSS>/                # 單次執行專屬目錄 (含 metrics_summary.csv/.txt, segmentation/, log.txt)
│   │   └── latest_log.txt                    # 最新一次執行的即時日誌連結副本
│   ├── 2_fab_no_gt/                          # Task 2 無 GT 模擬盲池抓漏結果
│   │   ├── <YYYYMMDD_HHMMSS>/                # 單次執行專屬目錄 (含 anomaly_images/, amap_npy/, 排名報表)
│   │   └── latest_log.txt                    # 最新一次執行的即時日誌連結副本
│   └── 3_pure_normal_<類別>/                 # Task 3 100% 純正常池誤報率 (FPR) 驗證結果 (bottle, cable...)
│       ├── <YYYYMMDD_HHMMSS>/                # 單次執行專屬目錄
│       └── latest_log.txt                    # 最新一次執行的即時日誌連結副本
├── wafer_escape_results/                     # [公司內唯一結果出口] run_inside.sh 實測 5000x5000 晶圓之輸出根目錄
│   ├── <YYYYMMDD_HHMMSS>/                    # 單次執行專屬目錄
│   └── latest_log.txt                        # 最新一次執行的即時日誌連結副本
├── 1_run_public_with_gt.py                   # [核心腳本 1] 有 GT 公開資料集抽樣 + 訓練 + 評估自動化腳本
├── 2_run_fab_no_gt.py                        # [核心腳本 2] 無 GT 自動挑 Golden + 高解析度切塊訓練 + 縫合濾波畫紅框腳本
├── run_outside.sh                            # [執行檔 1] 公司外一鍵測試腳本 (連跑 Task 1 有 GT + Task 2 無 GT 模擬抓漏)
├── run_pure_normal_test.sh                   # [執行檔 2] 100% 純正常資料集誤報率 (FPR) 專項驗證腳本
└── run_inside.sh                             # [執行檔 3] 公司內 5000x5000 晶圓盲池即插即用抓漏腳本
```

---

## 3. 核心檔案用途、使用方式與 Input / Output 路徑對照總表

以下詳細說明本專案 6 個核心修改與新增檔案的設計目的、輸入來源（Input）、輸出位置（Output）與操作方式：

### 3.1 檔案一覽與 Input / Output 路徑速查表

| 檔案名稱 | 核心目的與功能 | Input (輸入資料夾 / 檔案) | 中間產物與權重位置 (`datasets/` & `logs/`) | Output (最終結果輸出資料夾) |
| :--- | :--- | :--- | :--- | :--- |
| **`1_run_public_with_gt.py`** | **有 GT 公開資料集驗證：** 自動執行 Few-Shot 抽樣、模型訓練與 `mode=AD` 指標驗證，自動統計各階段耗時並匯出報表。 | `./datasets/mvtec/<類別>/` | - 抽樣訓練集：`./datasets/mvtec_4shot/`<br>- 訓練權重：`./logs/mvtec_4shot/dinov3dbug/train-step2000.pth.tar` | `./output/1_public_with_gt/<YYYYMMDD_HHMMSS>/` |
| **`2_run_fab_no_gt.py`** | **無 GT 晶圓抓漏核心引擎：** 自動挑 4 張 Golden 圖 -> 高解析度切塊訓練 -> 子圖塊推論與 2D 餘弦無縫拼接 -> 晶圓圓周與微塵抑制 -> 動態門檻篩選與畫紅框。 | `--pool_dir` 指定之無標註圖片池（或由 `--mock_from_mvtec` 自動自 MVTec 生成） | - Golden 訓練圖塊：`./datasets/<data_name>/train/<類別>/`<br>- Golden 原圖備份：`./datasets/<data_name>/golden_full_res_backup/<類別>/`<br>- 測試切塊池：`./datasets/<data_name>_tiled_pool/`<br>- 訓練權重：`./logs/<data_name>/dinov3<diy_name>/` | `<out_dir>/<YYYYMMDD_HHMMSS>/` |
| **`run_outside.sh`** | **公司外一鍵綜合測試：** 依序自動執行 Task 1（6 類別有 GT 驗證）與 Task 2（無 GT 混合盲池抓漏驗證），最後印出抓漏率與正常圖分數分佈統計。 | `./datasets/mvtec/` | - `./datasets/mvtec_4shot/`<br>- `./datasets/sim_fab_capsule_pool/`<br>- `./datasets/mvtec_sim_4shot/`<br>- `./logs/mvtec_4shot/` & `./logs/mvtec_sim_4shot/` | - Task 1: `./output/1_public_with_gt/<時間戳>/`<br>- Task 2: `./output/2_fab_no_gt/<時間戳>/` |
| **`run_pure_normal_test.sh`** | **100% 純正常池誤報率 (FPR) 驗證：** 自動從 `test/good` 建立 0 瑕疵純正常盲池（或讀取自訂純正常晶圓池），驗證防誤報濾波與動態門檻之穩定性。 | `./datasets/mvtec/<類別>/test/good/`（或 `CUSTOM_PURE_NORMAL_DIR`） | - 純正常盲池：`./datasets/sim_pure_normal_<類別>_pool/`<br>- 訓練圖塊：`./datasets/mvtec_sim_4shot/`<br>- 權重：`./logs/mvtec_sim_4shot/dinov3sim_escape/` | `./output/3_pure_normal_<類別>/<YYYYMMDD_HHMMSS>/` |
| **`run_inside.sh`** | **公司內 5000x5000 晶圓實測部署：** 強制啟用斷網離線模式，讀取廠內本地 DINO 權重與工程師初篩過之無 GT 晶圓池，執行 512x512 原生切塊抓漏。 | - 圖片池：`WAFER_POOL_DIR`<br>- 本地權重：`COMPANY_DINO_CKPT`<br>- 本地架構：`./hub_cache/` | - Golden 圖塊：`./datasets/mvtec_fab_4shot/train/wafer/`<br>- 測試切塊池：`./datasets/mvtec_fab_4shot_tiled_pool/`<br>- 權重：`./logs/mvtec_fab_4shot/dinov3fab_h100/` | `./wafer_escape_results/<YYYYMMDD_HHMMSS>/` |
| **`foundad/src/utils/synthesis.py`** | **CutPaste 合成異常前景遮罩生成：** 支援 MVTec/VisA 原生類別，並對 `wafer` 等自訂類別回傳正確 `(H, W)` 全前景遮罩。 | 訓練階段載入之 Golden 圖片 Tensor `(C, H, W)` | 記憶體內即時合成異常 Patch 供 `train.py` 計算 Loss | 無獨立檔案輸出（直接影響訓練權重品質） |

---

### 3.2 各檔案詳細功能與執行方式

#### (1) `1_run_public_with_gt.py` — 有 GT 公開資料集自動化驗證腳本
* **功能亮點：**
  * 自動解除抽樣圖片唯讀權限（防止重新抽樣時發生 `PermissionError`）。
  * 新增 `--max_batch_size`（預設 16）自動限制多類別聯合訓練時的 Batch Size 上限，防止高解析度下 GPU OOM。
  * 自動對齊最新訓練步數權重（`find_latest_ckpt_info`），並精確記錄「少樣本抽樣耗時」、「模型訓練耗時」、「推論耗時」與「全程總耗時」。
* **單獨使用指令：**
  ```bash
  # 完整執行抽樣 + 訓練 + 測試
  python 1_run_public_with_gt.py \
      --dataset mvtec \
      --test_root ./datasets/mvtec \
      --datasets_dir ./datasets \
      --data_name mvtec_4shot \
      --classnames "[bottle,cable,capsule,grid,tile,transistor]" \
      --diy_name dbug \
      --ckpt_step 2000 \
      --shots 4 \
      --max_batch_size 16 \
      --save_dir ./output/1_public_with_gt

  # 若已訓練完成，想跳過抽樣與訓練直接重新輸出評估報表：加上 --skip_sample --skip_train
  ```

#### (2) `2_run_fab_no_gt.py` — 無 GT 高解析度切塊、防誤報濾波與動態門檻抓漏腳本
* **功能亮點：**
  1. **5000x5000 高解析度防失真切塊（Tiling）與無縫縫合（Feathering Stitching）：**
     * **訓練階段：** 將 Step 1 自動挑出的 4 張 Golden 大圖備份至 `golden_full_res_backup/`，並無損切分為 `--tile_size`（如 512 或 1024）的重疊子圖塊，自動過濾純黑/純白背景角落塊（`--bg_std_thresh 1.5`），跨 Golden 選出最多 `--max_train_tiles`（預設 48）張涵蓋中心、中環、外環的代表性圖塊進行訓練，讓模型同時學會局部細節與跨片正常公差。
     * **推論階段：** 將測試池圖片切塊推論後，使用 **2D 餘弦平滑權重遮罩（Raised-Cosine Window）** 將子圖塊 `.npy` 無縫拼接回原圖完整解析度（如 $5000 \times 5000$），徹底消除切塊邊界接縫。
  2. **四合一防誤報熱力圖純化器（`refine_amap_for_wafer_and_noise`）：**
     * **跨 Golden 結構複雜區容差補償：** 自動扣除 Golden 圖在固定複雜結構區的背景基準值。
     * **灰階形態學開運算微塵濾波（`--speckle_filter_ratio 0.008`）：** 濾除單一 Patch 的針狀極端值噪點，解決 `tile` / `grid` 等紋理類別的單點誤報。
     * **矩形邊框抑制（`--border_ratio 0.02`）：** 抑制影像四週 2% 矩形邊緣的卷積 Padding 偽影。
     * **晶圓圓周邊緣 / 倒角抑制（`--wafer_edge_mode [auto|on|off]`、`--wafer_edge_margin 0.04`）：** 自動偵測圓形晶圓幾何中心與半徑，對最外圈 4% 半徑的倒角反光區（Bevel / Notch）進行餘弦平滑衰減，消除晶圓邊緣反光誤報。
  3. **5 種異常判定門檻模式（`--thresh_mode`）：**
     * `hybrid`（預設推薦）：結合盲池分位數 `P95` + 抗極端值之 MAD 穩健統計上界（`Median + sigma_k * Robust_Sigma`）+ 最低安全底線 `--min_threshold` 保底。
     * `mean_std`：直接以非 Golden 正常圖之 `mean + sigma_k * std` 作為門檻（不受 `min_threshold` 強制墊高影響）。
     * `percentile`：自動對齊盲池分位數（`--auto_percentile`，如 `95.0` 或 `98.0`），並以 `--min_threshold` 保底。
     * `top_k`：自動對齊盲池第 $K$ 高分（搭配 `--top_k` 限制最多只輸出前 $K$ 張最可疑圖片），並以 `--min_threshold` 保底。
     * `fixed`：直接使用 `--threshold` 作為固定絕對門檻。
  4. **智慧紅框收斂與「只輸出異常圖」機制：**
     * 透過峰值自適應核心收斂（`--core_ratio 0.55`）與最小有效缺陷面積過濾（`--min_area_ratio 0.00012`），自動合併重疊碎片框。
     * **僅有超過有效門檻且具備有效紅框的異常圖片才會輸出影像檔**，其餘正常圖片於報表中標記為 `PASS_SKIPPED` 或 `PASS_NOISE_FILTERED` 並直接略過不輸出圖片，大幅節省磁碟空間與工程師複檢時間。

#### (3) `run_outside.sh` — 公司外一鍵測試腳本（Task 1 有 GT + Task 2 無 GT 模擬抓漏）
* **用途：** 在外部環境一次驗證「公開資料集指標」與「無 GT 混入 8 張隱藏 Escape 瑕疵的抓漏能力」。執行完畢後會自動解析最新 CSV 報表，在終端機印出 **成功抓出瑕疵張數、正常圖誤報張數、非 Golden 正常圖分數分佈（Max / P99 / P95 / Median / Mean / Std）以及 `mean + 1*std / 2*std / 3*std` 門檻參考值**。
* **執行指令：**
  ```bash
  bash run_outside.sh
  ```

#### (4) `run_pure_normal_test.sh` — 100% 純正常資料集誤報率（FPR）專項驗證腳本
* **用途：** 專為驗證「當盲池內 100% 都是正常品（0 張瑕疵）時，系統是否會發生過殺誤報（Overkill）」而設計。
  * 在公司外：預設自動迴圈測試 `CATEGORIES=("bottle" "cable" "capsule" "grid" "tile" "transistor")` 6 大類別，自動從 `mvtec/<類別>/test/good` 複製建立 100% 純正常盲池 `./datasets/sim_pure_normal_<類別>_pool`。
  * 在公司內：可將腳本內的 `CUSTOM_PURE_NORMAL_DIR="/path/to/pure_normal_wafers"` 指向廠內已知 100% 正常的晶圓資料夾進行驗算。
* **執行指令：**
  ```bash
  bash run_pure_normal_test.sh
  ```

#### (5) `run_inside.sh` — 公司內 $5000 \times 5000$ 晶圓盲池即插即用腳本
* **用途：** 進公司部署的主力腳本。已預設 `TILE_SIZE=512`（1:1 對應模型原生輸入解析度，零壓縮失真）、`WAFER_EDGE_MODE="on"`（強制開啟晶圓圓周倒角反光抑制）、`HF_HUB_OFFLINE=1`（完全斷網模式）。
* **進公司只需打開 `run_inside.sh` 修改最上方 2 個路徑變數：**
  ```bash
  # 1. 工程師初篩為正常品、無 Ground Truth 的 5000x5000 圖片資料夾路徑 (保留原始檔名不更動)
  WAFER_POOL_DIR="/path/to/company/engineer_passed_wafers"

  # 2. 公司內部 H100 上的本地 DINOv3 或 DINOv2 .pth 權重路徑
  COMPANY_DINO_CKPT="/path/to/company/dinov3_or_dinov2.pth"
  ```
* **執行指令：**
  ```bash
  bash run_inside.sh
  ```

---

## 4. 執行訓練與測試（支援前景 bash、背景 nohup 與秒速重調門檻）

執行腳本前，請先確認已啟動環境並位於專案根目錄 `FoundAD/`：

```bash
conda activate foundad
cd /path/to/FoundAD
```

### 4.1 方式一：使用 bash 前景直接執行
```bash
# 1. 公司外綜合測試 (Task 1 有 GT + Task 2 無 GT 模擬盲池)
bash run_outside.sh

# 2. 100% 純正常資料集誤報率 (FPR) 驗證
bash run_pure_normal_test.sh

# 3. 公司內 5000x5000 無標註晶圓池抓漏實測
bash run_inside.sh
```

### 4.2 方式二：使用 nohup 背景執行（適合 5000x5000 大型圖片池、防止 SSH 斷線中斷）
在連線至實驗室或公司伺服器時，建議使用 `nohup` 將任務掛在背景執行：

```bash
# 公司外綜合測試背景執行
nohup ./run_outside.sh > train_outside.log 2>&1 &

# 100% 純正常池 FPR 測試背景執行
nohup ./run_pure_normal_test.sh > pure_normal_test.log 2>&1 &

# 公司內實測背景執行
nohup ./run_inside.sh > train_inside.log 2>&1 &
```

#### 常用進度監控指令：
* **即時查看最新執行進度**（除了查看 `nohup` 的 `.log` 外，各輸出目錄下皆有即時同步的 `latest_log.txt`）：
  ```bash
  tail -f ./output/2_fab_no_gt/latest_log.txt
  # 或在公司內：
  tail -f ./wafer_escape_results/latest_log.txt
  ```
* **確認背景程序與 GPU 狀態：**
  ```bash
  ps aux | grep main.py
  nvidia-smi
  ```

### 4.3 已訓練完成後，如何「跳過訓練」秒速重調門檻或邊緣參數？
當 Step 2（2000 Epochs 訓練）完成後，權重檔已經保存在 `./logs/` 下。若想嘗試不同的門檻模式（例如從 `hybrid` 改測 `mean_std` 並調整 `SIGMA_K`）或調整晶圓圓周抑制寬度（`WAFER_EDGE_MARGIN`），**完全不需要重新訓練模型**！

#### 作法 A：修改 `.sh` 腳本中的 `SKIP_TRAIN` 後重跑
打開 `run_inside.sh`（或 `run_outside.sh` / `run_pure_normal_test.sh`），修改以下參數：
```bash
THRESH_MODE="mean_std"        # 例如改測 mean + k*std 統計門檻模式 (或維持 "hybrid")
SIGMA_K=3.0                   # 調整標準差倍數 k (調小抓更細微瑕疵，調大更嚴格防誤報)
SKIP_TRAIN="--skip_train"     # 關鍵：設為 "--skip_train" 即可跳過訓練，直接載入既有權重推論畫框！
```
接著再次執行 `bash run_inside.sh`，即可在極短時間內產出全新時間戳記資料夾與新門檻下的紅框結果。

#### 作法 B：直接使用 Python 指令單獨重跑無 GT 推論
```bash
python 2_run_fab_no_gt.py \
    --pool_dir /path/to/company/engineer_passed_wafers \
    --ckpt /path/to/company/dinov3_or_dinov2.pth \
    --out_dir ./wafer_escape_results \
    --data_name mvtec_fab_4shot \
    --category wafer \
    --diy_name fab_h100 \
    --ckpt_step 2000 \
    --thresh_mode mean_std \
    --sigma_k 3.0 \
    --wafer_edge_mode on \
    --wafer_edge_margin 0.04 \
    --tile_size 512 \
    --skip_train
```

---

## 5. 確認訓練完的權重檔位置 (`./logs/`)

在訓練過程中，`train.py` 會自動將權重儲存於 `./logs/<data_name>/<模型名稱><diy_name>/` 目錄下：

* **Task 1：有 GT 公開資料集（`run_outside.sh`）**
  * 路徑：`./logs/mvtec_4shot/dinov3dbug/train-step2000.pth.tar`
* **Task 2 & Task 3：外部無 GT 模擬盲池與純正常池測試（`run_outside.sh` / `run_pure_normal_test.sh`）**
  * 路徑：`./logs/mvtec_sim_4shot/dinov3sim_escape/train-step2000.pth.tar`
  * 對接權重：`./logs/mvtec_sim_4shot/dinov3sim_escape/pretrained.pth.tar`（由 `2_run_fab_no_gt.py` 自動複製對接，供 `mode=demo` 推論讀取）
* **公司內晶圓實測（`run_inside.sh`）**
  * 路徑：`./logs/mvtec_fab_4shot/dinov3fab_h100/train-step2000.pth.tar`
  * 對接權重：`./logs/mvtec_fab_4shot/dinov3fab_h100/pretrained.pth.tar`

可使用以下指令快速確認權重檔是否已生成：
```bash
ls -lh ./logs/*/*/*.pth.tar
```

---

## 6. 檢驗測試結果與輸出報表說明 (`./output/` 與 `./wafer_escape_results/`)

每次執行都會在對應的輸出根目錄下建立 `<YYYYMMDD_HHMMSS>/` 時間戳記資料夾：

### 6.1 有 GT 公開資料集結果：`./output/1_public_with_gt/<YYYYMMDD_HHMMSS>/`

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `metrics_summary.txt` | **實驗指標與耗時總結報表：** 列出各類別與平均之 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC`，以及訓練/推論階段詳細耗時與權重路徑。 |
| `metrics_summary.csv` | 上述指標與耗時數據之 CSV 表格檔，可直接用 Excel 開啟整理。 |
| `segmentation/<類別名>/` | **四合一視覺化對照圖：** `[Original 原圖 -> GT 真實瑕疵遮罩 -> Anomaly Map 預測熱力圖 -> Segmentation 二值化預測結果]` |
| `log.txt` | 本次執行的完整終端機輸出日誌（外層 `./output/1_public_with_gt/latest_log.txt` 會同步保留最新一次紀錄）。 |

### 6.2 無 GT 盲池抓漏與純正常驗證結果：`./output/2_fab_no_gt/<時間戳>/`、`./output/3_pure_normal_<類別>/<時間戳>/` 或廠內 `./wafer_escape_results/<時間戳>/`

這是交給公司複檢與寫入研究報告的核心產出目錄：

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `anomaly_images/` | **【核心產出 1】異常紅框三合一對照圖（僅輸出超過門檻之異常圖）：**<br>檔名格式為 `Rank001_Score0.3450_Box01_<原檔名>.png`（依異常分數由高至低排序，打開資料夾第一張即為最可疑的 Escape 晶圓）。<br>頂部附帶黃字狀態橫幅（顯示原檔名、分數、有效門檻、紅框數），下方由左至右為：`[左：原始圖 -> 中：純化後異常熱力疊圖 -> 右：紅色 Bounding Box 瑕疵定位框]`。 |
| `anomaly_images/full_res_bbox_only/` | **【核心產出 2】1:1 原始全解析度紅框定位圖：**<br>維持原圖 $100\%$ 原始解析度（例如 $5000 \times 5000$ 不縮放），僅在原圖畫上紅色瑕疵框與局部最高分標籤，供工程師放大檢視微米級缺陷細節。 |
| `escape_ranking_report.txt` | **無 GT 異常抓漏總排名與統計報表（純文字版）：**<br>1. 標頭完整記錄本次門檻模式、有效門檻、非 Golden 正常圖分數分佈（`Max`, `P99`, `P95`, `Median`, `Mean`, `Std`）與 `mean + 1*std / 2*std / 3*std` 門檻參考值。<br>2. 表格依序排列所有圖片之 `Rank`、`Original_Filename`、`Status`（`ANOMALY` / `GOLDEN_PASS` / `PASS_SKIPPED` / `PASS_NOISE_FILTERED`）、`Max_Score`、`BBoxes` 數量與對應輸出檔名。<br>3. 檔案末尾自動附加訓練與推論各子階段的詳細計時總結。 |
| `escape_ranking_report.csv` | 同上之 CSV 報表（含 `Max_Score`、`TopK_Mean_Score`、`Num_BBoxes`、`Resolution` 等欄位），方便匯入 Excel 篩選分析。 |
| `selected_4_golden_images.txt` | **自動挑選之 Golden Image 紀錄：** 列出 Step 1 從無標註圖片池自動算出的前 4 名最純淨中心樣本原檔名與特徵中心度距離分數。 |
| `amap_npy/` | **異常樣本之全解析度熱力圖矩陣（`.npy`）：** 儲存超過門檻之異常圖片縫合且濾波後的 2D 矩陣（若解析度 $> 2048$ 自動轉存 `float16` 節省空間）。 |
| `tile_manifest.json` | 記錄每張測試原圖切分為高解析度子圖塊之 `(y, x, h, w)` 座標映射表。 |
| `log.txt` | 本次執行的完整終端機日誌與時間統計。 |

---

## 7. 常見問題與預先排錯指南（Troubleshooting）

### Q1：在公司內執行時報錯 `URLError` / `ConnectionError`（嘗試連網下載 GitHub 或 HuggingFace）
* **原因：** 公司內網阻擋連向 GitHub (`torch.hub.load`) 或 HuggingFace。
* **解法：**
  1. 確認已照 **第 1.3 節 Step 1** 將外部機器 `~/.cache/torch/hub/*dino*` 複製到專案下的 `./hub_cache/`。
  2. 確認 `run_inside.sh` 內的環境變數 `COMPANY_DINO_CKPT` 已正確指向公司伺服器上的 `.pth` 權重檔路徑，且 `FAB_LOCAL_DINO_REPO` 指向 `./hub_cache/facebookresearch_dinov3_main`。
  3. 若公司內的 DINOv3 權重與架構不匹配，可改用公司內的 `dinov2` 權重，並將設定檔中的 `model: dinov3` 改為 `model: dinov2`、`crop_size: 518`（本專案對 `dinov2` 支援 100% 內建離線實例化，完全不需外部 repo 快取）。

### Q2：訓練在 Epoch 1 就崩潰，報錯 `NotImplementedError` 或 CutPaste 貼圖只出現在圖片最上方？
* **原因：** 原版 `foundad/src/utils/synthesis.py` 對非 MVTec 原生類別會拋出例外；又或者舊版修補寫成 `np.ones((img.shape[0], img.shape[1]))`，因輸入 `img` 為 `(3, H, W)` Tensor，導致產生 `(3, H)` 的錯誤維度遮罩。
* **解法：** 請確認 `foundad/src/utils/synthesis.py` 第 74 行的 `else:` 分支為：
  ```python
  return np.ones_like(img_gray, dtype=np.uint8)
  ```
  （可直接執行 **第 1.3 節 Step 3** 的自動修補指令）。

### Q3：如何根據報表中的「非 Golden 正常圖分數分佈」微調門檻，減少誤報（Overkill）或抓出更微弱瑕疵？
* **解法：**
  每次跑完 `run_inside.sh`、`run_outside.sh` 或 `run_pure_normal_test.sh`，終端機最後都會印出如下的統計面板：
  ```text
  📊 非 Golden 正常圖分數分佈    : Max=0.2410 | P99=0.2350 | P95=0.2180 | Median=0.1850 | Mean=0.1880 | Std=0.0160
  📐 非 Golden 正常圖門檻參考值  : mean+1*std=0.2040 | mean+2*std=0.2200 | mean+3*std=0.2360
  ```
  * **想直接用統計標準差倍數當門檻：** 將腳本中的 `THRESH_MODE` 設為 `"mean_std"`，並將 `SIGMA_K` 設為 `3.0`（即 `mean + 3*std`）或 `3.5`，再設定 `SKIP_TRAIN="--skip_train"` 重跑一次推論。
  * **想指定固定分數當門檻：** 將 `THRESH_MODE` 設為 `"fixed"`，並設定 `THRESHOLD=0.25`（搭配 `SKIP_TRAIN="--skip_train"` 重跑）。
  * **晶圓最外圈圓弧邊緣反光被誤框：** 確認 `WAFER_EDGE_MODE="on"`，並將 `WAFER_EDGE_MARGIN=0.04` 調大為 `0.06`（代表外圈 6% 半徑平滑抑制）。
  * **畫面出現微小單點灰塵誤報：** 將 `SPECKLE_FILTER_RATIO=0.008` 調大為 `0.012`，或將 `MIN_AREA_RATIO=0.00012` 調大為 `0.00025`。

### Q4：執行時報錯 `RuntimeError: Address already in use`（Port 40112 衝突）
* **原因：** 前一次執行的 PyTorch 分散式程序（NCCL `master_port: 40112`）異常中斷尚未釋放 Port，或同一台伺服器上有其他程序佔用。
* **解法：**
  * 清除殘留程序：`pkill -f "foundad/main.py"`
  * 或在呼叫 `foundad/main.py` 時加上自訂 Port 參數：`dist.master_port=40125`。