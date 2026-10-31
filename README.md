# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 完整建置與操作指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為「晶圓廠工程師初篩為正常品、完全無標註（No Ground Truth）之 Wafer 影像池」設計。系統支援兩大模式：

* **有 GT 公開資料集驗證模式（公司外測試 / 報告用）：** 自動完成少樣本抽樣、訓練，並將終端機之 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC` 分數匯出為 `.txt` 與 `.csv` 報表。
* **無 GT 晶圓廠盲池抓漏模式（公司內即插即用）：** 自動透過 DINO 特徵中心度（Medoid Centrality）從無標註圖片池挑出 4 張最純淨的 Golden Image → 訓練流形預測器 → 自動以動態統計門檻（`μ + kσ`）框出可疑瑕疵位置（紅色 Bounding Box）並輸出異常排名表。

---

## 1. 環境、套件安裝（含公司內斷網環境準備）

### 1.1 基礎環境需求
* **作業系統：** Linux (Ubuntu 20.04 / 22.04+)
* **Python 版本：** Python 3.10 ~ 3.12
* **GPU 硬體：** 外部測試機 或 廠內伺服器

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

#### Step 2：預先下載離線套件包 (`wheels/`) 或打包 Conda 環境
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

#### Step 3：確認自訂類別的前景遮罩修補已生效
原版 FoundAD 的 `foundad/src/utils/synthesis.py` 僅寫死 MVTec 的 15 個原生類別，遇到 `"wafer"` 等自訂類別會在 Epoch 1 報 `NotImplementedError`。請執行以下指令確認已修補完成：

```bash
python -c '
p = "foundad/src/utils/synthesis.py"
txt = open(p).read()
old = "raise NotImplementedError(\"Unsupported foreground segmentation category\")"
new = "return np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) # 支援 wafer 等自訂類別"
if old in txt:
    open(p, "w").write(txt.replace(old, new))
    print("[✓] 已成功修補 synthesis.py！")
else:
    print("[✓] synthesis.py 已經是支援 wafer 自訂類別的狀態，無需修改。")
'
```

---

## 2. 資料集與專案資料夾結構建置

為保持目錄整潔，本專案已將所有檔案收斂為以下標準結構。進公司部署時，請確認根目錄結構如下：

```text
FoundAD/
├── foundad/                        # [核心] FoundAD 模型與訓練/推論原始碼
│   ├── configs/                    # Hydra 設定檔 (含 app/train_dinov3.yaml, app/test.yaml)
│   ├── src/
│   │   ├── AD.py                   # 評估與無 GT 熱力圖推論 (已內嵌 BBox 畫框 Hook)
│   │   ├── foundad.py              # 支援離線載入本地 DINOv2 / DINOv3 權重之視覺模組
│   │   ├── train.py                # 流形預測器 (Manifold Predictor) 訓練迴圈
│   │   ├── sample.py               # 有 GT 模式之少樣本抽樣腳本
│   │   └── utils/synthesis.py      # CutPaste 異常合成模組 (已支援 wafer 類別)
│   └── main.py                     # FoundAD 主入口程式
├── fab_tools/                      # [晶圓廠專用外掛模組]
│   ├── auto_golden.py              # 無 GT 自動計算特徵中心挑選 4 張純淨 Golden Image
│   └── bbox_visualizer.py          # 動態門檻計算、繪製紅色瑕疵外框與輸出排名表
├── hub_cache/                      # [離線快取] 存放 facebookresearch_dinov3_main 等本地架構檔
├── datasets/                       # [資料集專區] 所有輸入圖片與挑出的 Golden Image 皆存放於此
│   ├── mvtec/                      # (公司外測試用) 原始 MVTec AD 資料集
│   │   ├── bottle/
│   │   └── cable/
│   ├── mvtec_4shot/                # (自動生成) 有 GT 模式抽出的 4-shot 訓練集
│   │   └── train/bottle/000.png...
│   └── mvtec_wafer_4shot/          # (自動生成) 無 GT 模式由 auto_golden.py 挑出的 4 張 Golden 圖
│       ├── train/wafer/000.png...
│       └── selected_4_golden_images.txt
├── logs/                           # [模型權重區] 訓練過程自動產生的 .pth.tar 權重檔與訓練曲線
├── output/                         # [唯一結果出口] 所有文字數據 (.txt/.csv) 與紅框圖片皆集中於此
│   ├── 1_public_with_gt/           # 有 GT 公開資料集之實驗結果
│   ├── 2_fab_no_gt/                # 公司外無 GT 模擬盲池之抓漏結果
│   └── fab_results/           # 進公司 實測晶圓池之抓漏結果
├── 1_run_public_with_gt.py         # 有 GT 實驗自動化串接腳本
├── 2_run_fab_no_gt.py              # 無 GT 自動挑圖 + 訓練 + 畫紅框一條龍腳本
├── run_outside.sh             # [執行檔 1] 公司外 測試腳本
└── run_inside.sh              # [執行檔 2] 公司內 即插即用腳本
```

### 資料夾準備方式
* **在公司外測試（使用公開資料集 MVTec AD）：** 只需將下載好的 MVTec 類別（如 `bottle`、`cable`）放進 `./datasets/mvtec/` 下即可，其餘少樣本資料夾（`mvtec_4shot`、`sim_fab_bottle_pool`）腳本會自動建立。
* **在公司內實測（使用工程師初篩正常之 Wafer 圖片池）：** 不需要手動分類或改檔名！假設公司內的晶圓圖片全部放在某個路徑（例如 `/data/fab_wafers/passed_images/`），只需打開 `run_inside.sh`，修改最上方的兩個路徑變數：

```bash
# 1. 指向工程師初篩為正常品、無 Ground Truth 的晶圓圖片資料夾路徑
WAFER_POOL_DIR="/data/fab_wafers/passed_images"

# 2. 指向公司內部 server 上的本地 DINOv3 或 DINOv2 .pth 權重檔案路徑
COMPANY_DINO_CKPT="/data/weights/dinov3_vitb16_pretrain.pth"
```
程式執行時會自動從 `WAFER_POOL_DIR` 挑出 4 張最純淨的 Golden Image，並自動建立 `./datasets/mvtec_wafer_4shot/train/wafer/000.png ~ 003.png` 供模型訓練。

---

## 3. 執行訓練（提供 bash 互動式與 nohup 背景執行兩種方式）

執行腳本前，請先確認已啟動環境並位於專案根目錄 `FoundAD/`：

```bash
conda activate foundad
cd /path/to/FoundAD
```

> ⚠️ **訓練開關提醒：**
> * 在 `run_outside.sh` 中，若要從頭訓練，請確保 `SKIP_GT_TRAIN=""` 且 `SKIP_FAB_TRAIN=""`（設為空字串代表要執行 2000 Epochs 訓練）。
> * 在 `run_inside.sh` 中，第一次執行請確保 `SKIP_TRAIN=""`。

### 方式一：使用 bash 直接執行（適合在 Terminal 即時觀察 Loss 變化）
4-Shot 訓練 2000 個 Epochs 在 測試用的小型資料集 上僅需約 1.5 ~ 3 分鐘，可直接於前景執行：

* **公司外（測試公開資料集 + 模擬盲池）：**
  ```bash
  bash run_outside.sh
  ```
* **公司內（實測無標註晶圓圖片池）：**
  ```bash
  bash run_inside.sh
  ```

### 方式二：使用 nohup 背景執行（適合圖片池龐大、防止 SSH 斷線中斷）
若在公司內連線至 伺服器時，為避免網路瞬斷或關閉視窗導致訓練中斷，強烈建議使用 `nohup` 將任務掛在背景執行，並將所有輸出導向 `.log` 檔案：

* **公司外背景執行：**
  ```bash
  nohup bash run_outside.sh > train_outside.log 2>&1 &
  ```
* **公司內背景執行：**
  ```bash
  nohup bash run_inside.sh > train_inside.log 2>&1 &
  ```

#### 常用 nohup 監控與管理指令：
* **即時查看訓練進度與 Epoch Loss**（按 `Ctrl + C` 可退出觀看，不會中斷背景程式）：
  ```bash
  tail -f train_inside.log
  ```
* **確認程式是否還在背景運行：**
  ```bash
  ps aux | grep main.py
  ```
* **查看 GPU 記憶體與使用率：**
  ```bash
  nvidia-smi
  ```
* **若需強制中止背景訓練：** 先由 `ps aux | grep main.py` 找到該程序的 PID（第二欄數字），再執行 `kill -9 <PID>`。

---

## 4. 確認訓練完的權重檔位置

在訓練過程中（預設 2000 個 Epochs），`train.py` 每 100 個 Iteration (`gstep % 100 == 0`) 會自動儲存一次權重檔。訓練完成後，權重檔會存放在 `./logs/` 目錄下：

### 4.1 權重存放路徑規則
`./logs/<資料集名稱>/<模型名稱><diy_name>/`

* **情況 A：有 GT 公開資料集（`run_outside.sh` 的 Task 1）**
  * 資料夾位置：`./logs/mvtec_4shot/dinov3dbug/`
  * 最終權重檔：`./logs/mvtec_4shot/dinov3dbug/train-step2000.pth.tar`
  * 訓練紀錄表：`./logs/mvtec_4shot/dinov3dbug/train.csv`（記錄每個 Epoch 的 Loss 與耗時）
* **情況 B：無 GT 模擬盲池（`run_outside.sh` 的 Task 2）**
  * 資料夾位置：`./logs/mvtec_sim_4shot/dinov3sim_escape/`
  * 最終權重檔：
    * `./logs/mvtec_sim_4shot/dinov3sim_escape/train-step2000.pth.tar`
    * `./logs/mvtec_sim_4shot/dinov3sim_escape/pretrained.pth.tar`（由腳本自動自 `train-step2000.pth.tar` 複製對接，專門提供給 `mode=demo` 無標註推論讀取）
* **情況 C：公司內資料集實測（`run_inside.sh`）**
  * 資料夾位置：`./logs/mvtec_wafer_4shot/dinov3fab/`
  * 最終權重檔：
    * `./logs/mvtec_wafer_4shot/dinov3fab/train-step2000.pth.tar`
    * `./logs/mvtec_wafer_4shot/dinov3fab/pretrained.pth.tar`

### 4.2 快速檢查權重是否成功生成的指令
```bash
ls -lh ./logs/*/*/*.pth.tar
```
若看到 `train-step2000.pth.tar`（以及無 GT 模式下的 `pretrained.pth.tar`），且檔案大小約為數十 MB，即代表訓練已 100% 成功完成。

---

## 5. 執行測試（含單獨重跑測試與調整畫框門檻 SIGMA_K）

### 5.1 預設全自動測試
當執行第 3 節的 `run_outside.sh` 或 `run_inside.sh` 時，腳本在跑完 **Step 2（訓練）** 後，會自動接續執行 **Step 3（測試與畫紅框）**，無需手動介入。

### 5.2 已訓練完成後，如何「跳過訓練」單獨執行測試？
在晶圓廠實測時，你可能會想微調紅色瑕疵框的敏感度門檻 `SIGMA_K`（例如覺得紅框太多想調嚴格一點，或微弱瑕疵沒被框出來想調敏感一點）。此時完全不需要重新訓練模型，只需設定 `--skip_train`，幾秒鐘內就能重新產出紅框圖與排名表：

#### 作法 A：修改 `.sh` 腳本中的參數後重跑
打開 `run_inside.sh`（或 `run_outside.sh`），修改以下兩行：

```bash
SIGMA_K=3.5               # 調整門檻：調小 (如 2.6) 會框出更多微弱瑕疵；調大 (如 3.8) 只框最明顯的嚴重瑕疵
SKIP_TRAIN="--skip_train" # 設為 "--skip_train" 即可跳過 2000 epochs 訓練，直接載入既有權重推論畫框！
```

接著執行：
```bash
bash run_inside.sh
```

#### 作法 B：直接使用 Python 指令單獨跑測試
* **單獨測試「有 GT 公開資料集」並輸出 `.txt`/`.csv` 分數：**
  ```bash
  python 1_run_public_with_gt.py \
      --dataset mvtec \
      --test_root ./datasets/mvtec \
      --datasets_dir ./datasets \
      --data_name mvtec_4shot \
      --classnames "[bottle,cable]" \
      --diy_name dbug \
      --ckpt_step 2000 \
      --save_dir ./output/1_public_with_gt \
      --skip_sample --skip_train
  ```

* **單獨測試「無 GT 晶圓池」並重新畫紅框：**
  ```bash
  python 2_run_fab_no_gt.py \
      --pool_dir /data/fab_wafers/passed_images \
      --datasets_dir ./datasets \
      --ckpt /data/weights/dinov3_vitb16_pretrain.pth \
      --out_dir ./output/fab_results \
      --data_name mvtec_wafer_4shot \
      --category wafer \
      --diy_name fab \
      --ckpt_step 2000 \
      --sigma_k 3.0 \
      --skip_train
  ```

---

## 6. 檢驗測試結果位置

所有的文字報表（`.txt`、`.csv`）與視覺化圖片皆已統一整合至專案根目錄下的 `./output/` 資料夾內：

### 6.1 有 GT 公開資料集結果：`./output/1_public_with_gt/`

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `metrics_summary_<時間戳>.txt` | **精簡實驗分數報表：** 直接列出各類別與平均之 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC`，可直接複製貼入進度報告。 |
| `metrics_summary_<時間戳>.csv` | 實驗分數之 Excel / CSV 表格檔。 |
| `terminal_full_log_<時間戳>.txt` | 完整的終端機執行紀錄（包含所有超參數設定與訓練過程）。 |
| `segmentation/<類別名>/` | **四合一視覺化對照圖：** `[Original 原圖 → GT 真實瑕疵遮罩 → Anomaly Map 預測熱力圖 → Segmentation 二值化預測結果]` |

### 6.2 無 GT 晶圓抓漏結果：`./output/2_fab_no_gt/`（或廠內 `./output/fab_results/`）
這是進公司後交給產線工程師複檢的核心產出目錄：

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `selected_4_golden_images.txt` | **自動挑選之 Golden Image 紀錄：** 列出系統從無標註圖片池中自動算出的前 4 名最純淨中心樣本原檔名與 Centrality 分數（可用於向工程師或老師佐證挑出的樣本具代表性）。 |
| `escape_ranking_report.txt` | **無 GT 異常抓漏總排名表（純文字版）：** 將圖片池內所有 Wafer 依異常分數（Anomaly Score）由高到低排序。排在最前面的 Rank 1, 2, 3... 即為最可能是被工程師誤判放行的 Escape 瑕疵晶圓，並附帶偵測到的瑕疵框數量與 `(x, y, w, h)` 座標。 |
| `escape_ranking_report.csv` | 同上之 CSV 報表，方便用 Excel 篩選 `Defect_Boxes_Count > 0` 的可疑晶圓。 |
| `bbox_visualizations/` | **【核心產出】三合一紅色瑕疵框定位圖：** 檔名格式為 `Rank0001_Box2_<原檔名>.png`（依異常程度排序編號，打開資料夾第一張就是最可疑的圖片）。每張圖由左至右為：`[左：原始晶圓圖 → 中：異常能量熱力疊圖 → 右：紅色 Bounding Box 瑕疵定位框]` |
| `heatmaps/` | FoundAD 原生輸出的單張熱力圖疊圖。 |
| `amap_npy/` | 每張測試圖的原始 2D 異常分數矩陣（`.npy` 檔），保留供後續快速調整門檻使用。 |

---

## 7. 常見問題與預先排錯指南（Troubleshooting）

以下整理了在跨環境遷移與執行過程中可能遇到的問題與秒解方案：

### Q1：在公司內執行時報錯 `URLError` / `ConnectionError`（嘗試連網下載 GitHub 或 HuggingFace）
* **原因：** 公司內網阻擋連向 GitHub (`torch.hub.load`)。
* **解法：**
  1. 確認已照 **第 1.3 節** 將外部機器 `~/.cache/torch/hub/*dino*` 複製到專案下的 `./hub_cache/`。
  2. 確認 `run_inside.sh` 內的環境變數 `COMPANY_DINO_CKPT` 已正確指向公司伺服器上的 `.pth` 權重檔路徑，且 `FAB_LOCAL_DINO_REPO` 指向 `./hub_cache/facebookresearch_dinov3_main`。
  3. 若公司內的 DINOv3 權重與架構不匹配，可改用公司內的 `dinov2_vitl14` 權重，並將 `foundad/configs/app/train_dinov3.yaml` 與 `test.yaml` 中的 `model: dinov3` 改為 `model: dinov2`、`crop_size: 518`（本專案對 `dinov2` 支援 100% 內建離線模型實例化，完全不需外部 repo 快取）。

### Q2：訓練在 Epoch 1 就崩潰，報錯 `NotImplementedError: Unsupported foreground segmentation category`
* **原因：** `foundad/src/utils/synthesis.py` 第 75 行對非 MVTec 原生 15 類別名稱（如 `wafer`）拋出例外。
* **解法：** 執行 **第 1.3 節 Step 3** 的那段 `python -c ...` 修補指令，將該行替換為 `return np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)` 即可解決。

### Q3：測試時報錯 `No ckpt path is found.` 或 `FileNotFoundError: .../pretrained.pth.tar`
* **原因：**
  1. 訓練尚未跑完（或中途報錯中斷），導致 `./logs/` 底下沒有產生 `train-step2000.pth.tar`。
  2. 當訓練多個類別時，若 `batch_size` 小於總圖片數，1 個 Epoch 會走多個 Step，導致最終步數為 `step4000` 而非 `step2000`。
* **解法：**
  1. 先用 `ls ./logs/*/*/` 查看實際存在的 `train-stepXXXX.pth.tar` 數字是多少。
  2. 本專案新版 `1_run_public_with_gt.py` 與 `2_run_fab_no_gt.py` 已內建自動搜尋最新步數並自動複製為 `pretrained.pth.tar` 的機制；若手動執行 `main.py`，請確保 `app.ckpt_step` 與 `./logs/` 下的檔名步數一致。

### Q4：輸出的 `bbox_visualizations/` 裡面，正常晶圓也被畫上紅框（過殺 Overkill），或微弱瑕疵沒被框到（漏檢 Escape）？
* **原因：** 無 GT 模式下採用統計動態門檻 `Threshold = μ + SIGMA_K × σ`，不同產線製程的背景雜訊變異程度不同。
* **解法：**
  * **紅框太多（正常晶圓被誤框）：** 打開 `run_inside.sh`，將 `SIGMA_K=3.2` 調大為 `3.8` 或 `4.2`，並設定 `SKIP_TRAIN="--skip_train"` 重跑一次（只需幾秒鐘）。
  * **微弱瑕疵沒被框出來：** 將 `SIGMA_K=3.2` 調小為 `2.5` 或 `2.8`，並設定 `SKIP_TRAIN="--skip_train"` 重跑一次。

### Q5：執行時報錯 `RuntimeError: Address already in use`（Port 40112 衝突）
* **原因：** 前一次執行的 PyTorch 分散式程序（NCCL `master_port: 40112`）異常退出，尚未釋放 Port，或同一台 server 上有其他人正在跑程式。
* **解法：**
  * 清除殘留程序：`pkill -f "foundad/main.py"`
  * 或在 `foundad/main.py` 呼叫指令後加上隨機 Port 參數：`dist.master_port=40125`。
