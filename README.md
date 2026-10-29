# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 完整建置與操作指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為「晶圓廠工程師初篩為正常品、完全無標註（No Ground Truth）之 Wafer 影像池」設計。系統支援兩大模式：

* **有 GT 公開資料集驗證模式（公司外測試 / 報告用）：** 自動完成少樣本抽樣、訓練，並將終端機之 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC` 分數匯出為 `.txt` 與 `.csv` 報表。
* **無 GT 晶圓廠盲池抓漏模式（公司內 H100 即插即用）：** 自動透過 DINO 特徵中心度（Medoid Centrality）從無標註圖片池挑出 4 張最純淨的 Golden Image → 訓練流形預測器 → 自動以動態統計門檻（`μ + kσ`）框出可疑瑕疵位置（紅色 Bounding Box）並輸出異常排名表。

---

## 1. 環境、套件安裝（含公司內斷網環境準備）

### 1.1 基礎環境需求
* **作業系統：** Linux (Ubuntu 20.04 / 22.04+)
* **Python 版本：** Python 3.10 ~ 3.12
* **GPU 硬體：** 外部測試機（RTX 5090）或廠內伺服器（NVIDIA H100）

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
│   └── fab_h100_results/           # 進公司 H100 實測晶圓池之抓漏結果
├── 1_run_public_with_gt.py         # 有 GT 實驗自動化串接腳本
├── 2_run_fab_no_gt.py              # 無 GT 自動挑圖 + 訓練 + 畫紅框一條龍腳本
├── run_outside_5090.sh             # [執行檔 1] 公司外 (RTX 5090) 測試腳本
└── run_inside_h100.sh              # [執行檔 2] 公司內 (H100) 即插即用腳本
```

### 資料夾準備方式
* **在公司外測試（使用公開資料集 MVTec AD）：** 只需將下載好的 MVTec 類別（如 `bottle`、`cable`）放進 `./datasets/mvtec/` 下即可，其餘少樣本資料夾（`mvtec_4shot`、`sim_fab_bottle_pool`）腳本會自動建立。
* **在公司內實測（使用工程師初篩正常之 Wafer 圖片池）：** 不需要手動分類或改檔名！假設公司內的晶圓圖片全部放在某個路徑（例如 `/data/fab_wafers/passed_images/`），只需打開 `run_inside_h100.sh`，修改最上方的兩個路徑變數：

```bash
# 1. 指向工程師初篩為正常品、無 Ground Truth 的晶圓圖片資料夾路徑
WAFER_POOL_DIR="/data/fab_wafers/passed_images"

# 2. 指向公司內部 H100 上的本地 DINOv3 或 DINOv2 .pth 權重檔案路徑
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
> * 在 `run_outside_5090.sh` 中，若要從頭訓練，請確保 `SKIP_GT_TRAIN=""` 且 `SKIP_FAB_TRAIN=""`（設為空字串代表要執行 2000 Epochs 訓練）。
> * 在 `run_inside_h100.sh` 中，第一次執行請確保 `SKIP_TRAIN=""`。

### 方式一：使用 bash 直接執行（適合在 Terminal 即時觀察 Loss 變化）
4-Shot 訓練 2000 個 Epochs 在 RTX 5090 或 H100 上僅需約 1.5 ~ 3 分鐘，可直接於前景執行：

* **公司外（RTX 5090 測試公開資料集 + 模擬盲池）：**
  ```bash
  bash run_outside_5090.sh
  ```
* **公司內（H100 實測無標註晶圓圖片池）：**
  ```bash
  bash run_inside_h100.sh
  ```

### 方式二：使用 nohup 背景執行（適合圖片池龐大、防止 SSH 斷線中斷）
若在公司內連線至 H100 伺服器，為避免網路瞬斷或關閉視窗導致訓練中斷，強烈建議使用 `nohup` 將任務掛在背景執行，並將所有輸出導向 `.log` 檔案：

* **公司外背景執行：**
  ```bash
  nohup bash run_outside_5090.sh > train_outside.log 2>&1 &
  ```
* **公司內（H100）背景執行：**
  ```bash
  nohup bash run_inside_h100.sh > train_inside_h100.log 2>&1 &
  ```

#### 常用 nohup 監控與管理指令：
* **即時查看訓練進度與 Epoch Loss**（按 `Ctrl + C` 可退出觀看，不會中斷背景程式）：
  ```bash
