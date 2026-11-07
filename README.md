# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 完整建置與操作指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為「晶圓廠工程師初篩為正常品、完全無標註（No Ground Truth）之高解析度 Wafer 影像池」設計。系統全面升級為 **四階段模組化無 GT 抓漏與拆分評估流水線（4-Stage Modular Pipeline）**，支援三大核心驗證與部署場景：

* **模式一：公司內無 GT 晶圓廠盲池四階段抓漏模式（`run_all_layers.sh` 或單獨執行 `run_1` ~ `run_4` / 廠內即插即用）：**
  1. **Step 1 自動挑選與切塊投票純化 Golden（`1_auto_golden.py` / `run_1_auto_golden.sh`）：** 支援上千張大資料集固定種子快速抽樣（預設上限 250 張），結合「光度對比過濾 + 空間網格對齊 + 跨片局部異常峰值 + 近鄰多樣性去重 (Diversity NMS)」四合一初篩鎖定 Top-16 候選池；接著於 RAM 內切塊進行跨片同座標 PK 投票，選出最終 4 張最純淨 Golden 原圖與 48 張代表性高解析度訓練圖塊（若影像 <= 1280px 自動維持整張原圖訓練）。
  2. **Step 2 無 GT 高解析度流形訓練（`2_train_fab_no_gt.py` / `run_2_train_fab_no_gt.sh`）：** 內建防呆快取（有 Golden 不重挑、有權重不重訓），自動由系統動態分配空閒本地 Port（徹底解決 DDP Port 40112 衝突），並自動儲存 `train_meta_<category>.json` 與對接 `pretrained.pth.tar`。
  3. **Step 3 高解析度切塊推論、RAM 即時縫合與防誤報篩選（`3_infer_fab_no_gt.py` / `run_3_infer_fab_no_gt.sh`）：** 針對 > 1280px 大圖自動切塊推論，於 RAM 內以 2D 餘弦平滑權重（Raised-Cosine Window）無縫拼接回全解析度，並**立即自動銷毀所有子圖塊 `.npy` 碎片（0 殘留）**。結合「四合一防誤報熱力圖純化器（含 3x3 輕量微塵抑制保護細小裂痕/針孔）」、「真無 GT 左半部正常核心估計器（sigma_left）+ Golden 基準雙重校準動態門檻」與「無監督模型過擬合自檢（Overfitting Self-Diagnostic）」，自動框出紅色 Bounding Box 並輸出異常排名報表（正常圖片自動略過不輸出）。
  4. **Step 4 評估結果自動分包器（`4_split_eval_results.py` / `run_4_split_eval_results.sh`）：** 將推論結果依分層抽樣（`stratified`）、隨機、排名或檔名自動拆分為多份獨立 Eval 報表與圖片包（預設 3 包），支援 `--symlink` 軟連結秒速分包不佔用額外空間。
* **模式二：廠外公開資料集無 GT 模擬抓漏與純正常池 FPR 驗證模式（`run_outside.sh` / `run_pure_normal_test.sh`）：** 支援 MVTec AD 與 MVTec AD 2 多類別一鍵完成「建立模擬盲池 -> Step 1 Auto-Golden -> Step 2 訓練 -> Step 3 推論 -> Step 4 拆分評估」，自動於終端機輸出 Recall 抓漏率、FPR 誤報率、濾波前後消融對照（Ablation）與過擬合診斷。
* **模式三：有 GT 公開資料集基準驗證模式（`1_run_public_with_gt.py`）：** 自動完成少樣本抽樣、訓練，並將 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC` 分數與各階段耗時匯出為 `.txt` 與 `.csv` 報表。

---

## 1. 環境、套件安裝（含公司內斷網環境準備）

### 1.1 基礎環境需求
* **作業系統：** Linux (Ubuntu 20.04 / 22.04+)
* **Python 版本：** Python 3.10 ~ 3.12
* **GPU 硬體：** 外部測試機 或 廠內伺服器（如 NVIDIA H100 / A100 / RTX 系列）

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

## 2. 專案完整資料夾結構（含四階段自動生成目錄說明）

每次執行實驗時，系統皆會依資料集（`DATASET_ID`）、製程層（`LAYER_ID`）與時間戳記（`<YYYYMMDD_HHMMSS>`）建立獨立子資料夾，**絕不覆蓋歷史實驗結果**：

```text
FoundAD/
├── foundad/                                  # [核心] FoundAD 模型與訓練/推論原始碼
│   ├── configs/                              # Hydra 設定檔 (含 app/train_dinov3.yaml, app/test.yaml)
│   ├── src/
│   │   ├── AD.py                             # 評估與無 GT 熱力圖推論 (支援批次推論與 .npy 匯出 Hook)
│   │   ├── foundad.py                        # 支援離線載入本地 DINOv2 / DINOv3 權重之視覺模組
│   │   ├── train.py                          # 流形預測器訓練迴圈 (含訓練資料夾存在性防呆檢查與錯誤提示)
│   │   ├── sample.py                         # 有 GT 模式之少樣本抽樣腳本
│   │   └── utils/synthesis.py                # CutPaste 異常合成模組 (已修正支援 wafer 等自訂類別)
│   └── main.py                               # FoundAD 主入口程式 (支援 FAB_MASTER_PORT 動態 Port 分配)
├── fab_tools/                                # [晶圓廠輔助工具目錄]
│   ├── auto_golden.py                        # (向下相容副本) 同 1_auto_golden.py
│   ├── bbox_visualizer.py                    # 輔助視覺化工具模組
│   └── 3_split_eval_results.py               # (向下相容副本) 同 4_split_eval_results.py
├── hub_cache/                                # [離線快取] 存放 facebookresearch_dinov3_main 等本地架構檔
├── datasets/                                 # [資料集與自動生成訓練集/快取專區]
│   ├── mvtec/ & mvtec-2/                     # (Input) 原始 MVTec AD / MVTec AD 2 資料集
│   ├── wafer_<LAYER_ID>_4shot/               # (Step 1 自動生成) 廠內各製程層專屬 Golden 資料集
│   │   ├── train/wafer/                      # 切塊投票純化後之 48 張代表性訓練圖塊 (或 <=1280px 完整原圖)
│   │   ├── golden_full_res_backup/wafer/     # 切塊投票最終勝出的 4 張全解析度 Golden 原圖備份
│   │   ├── stage1_candidates_wafer.json      # Stage 1 初篩鎖定之 Top-16 候選晶圓詳細評分紀錄
│   │   ├── golden_tile_purity_report.txt     # Stage 2 候選池切塊投票排行榜與圖塊純化報表
│   │   ├── golden_cache_meta.json            # 盲池 MD5 指紋與 Golden 快取紀錄
│   │   └── train_meta_wafer.json             # Step 2 訓練完成後記錄之實際步數、耗時與權重路徑 Meta
│   ├── wafer_<LAYER_ID>_4shot_tiled_pool/    # (Step 3 自動生成) 測試盲池切分後的重疊推論子圖塊 PNG 快取
│   └── sim_fab_<類別>_pool/                  # (廠外自動生成) 模擬無 GT 盲池 (含 clean_* 與隱藏 escape_* 瑕疵圖)
├── logs/                                     # [模型權重區] 訓練自動產生的 .pth.tar 權重檔與 params.yaml
│   ├── wafer_<LAYER_ID>_4shot/dinov3fab_<LAYER_ID>/ # 廠內各 Layer 獨立訓練權重 (train-step*.pth.tar & pretrained.pth.tar)
│   └── mvtec_sim_<類別>_4shot/dinov3sim_escape_<類別>/ # 廠外公開資料集各類別獨立訓練權重
├── output/                                   # [廠外測試唯一結果出口] 依任務、類別與時間戳記分類存放
│   ├── 1_public_with_gt/<YYYYMMDD_HHMMSS>/   # Task 1 有 GT 公開資料集評估結果
│   ├── 2_fab_no_gt_<類別>/<YYYYMMDD_HHMMSS>/ # 廠外無 GT 模擬盲池抓漏結果 (內含 split_eval_3packs/)
│   └── 3_pure_normal_<類別>/<YYYYMMDD_HHMMSS>/ # 100% 純正常池誤報率 (FPR) 驗證結果
├── wafer_escape_results/                     # [公司內唯一結果出口] 實測高解析度晶圓之輸出根目錄
│   └── <DATASET_ID>/<LAYER_ID>/              # 依資料集 (如 W639) 與製程層 (如 TMQB28) 分層管理
│       ├── golden_reports/                   # Step 1 輸出之 Golden 挑選名單與切塊投票純化報表
│       ├── train_<YYYYMMDD_HHMMSS>/          # Step 2 訓練階段專屬日誌 (train_log.txt)
│       ├── <YYYYMMDD_HHMMSS>/                # Step 3 推論階段專屬結果目錄 (含紅框圖、排名報表與 split_eval_3packs/)
│       ├── latest_train_log.txt              # 最新一次訓練階段即時日誌
│       └── latest_log.txt                    # 最新一次推論階段即時日誌
├── 1_auto_golden.py                          # [核心程式 1] Stage 1 大資料集抽樣候選池 + Stage 2 切塊投票選 Golden 與純化
├── 2_train_fab_no_gt.py                      # [核心程式 2] 無 GT Golden 資料集載入防呆 + FoundAD 模型訓練與權重對接
├── 3_infer_fab_no_gt.py                      # [核心程式 3] 測試池切塊快取 + 批次推論 + RAM 縫合銷毀碎片 + 四合一防誤報畫紅框
├── 4_split_eval_results.py                   # [核心程式 4] 推論結果分層/隨機/排名自動分包器 (支援軟連結與排除 Golden)
├── 1_run_public_with_gt.py                   # [輔助程式] 有 GT 公開資料集抽樣 + 訓練 + 評估自動化腳本
├── run_1_auto_golden.sh                      # [廠內腳本 1] 單獨執行 Step 1 Auto-Golden 挑選與切塊純化
├── run_2_train_fab_no_gt.sh                  # [廠內腳本 2] 單獨執行 Step 2 無 GT 模型訓練
├── run_3_infer_fab_no_gt.sh                  # [廠內腳本 3] 單獨執行 Step 3 無 GT 推論、防誤報篩選與重調門檻
├── run_4_split_eval_results.sh               # [廠內腳本 4] 單獨執行 Step 4 評估結果拆分為多包
├── run_all_layers.sh                         # [廠內總控腳本] 多製程層 (Multi-Layer) 四階段全自動批次執行腳本
├── run_outside.sh                            # [廠外總控腳本] 公開資料集 (MVTec / MVTec-2) 四階段無 GT 模擬抓漏一鍵測試
└── run_pure_normal_test.sh                   # [廠外專項腳本] 100% 純正常資料集誤報率 (FPR) 專項驗證腳本
```

---

## 3. 核心檔案用途、演算法亮點與 Input / Output 路徑對照總表

### 3.1 檔案一覽與 Input / Output 路徑速查表

| 階段 / 檔案名稱 | 核心目的與功能 | Input (輸入資料夾 / 檔案) | 中間產物與權重位置 (`datasets/` & `logs/`) | Output (最終結果輸出資料夾) |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1**<br>`1_auto_golden.py`<br>`run_1_auto_golden.sh` | **兩階段 Golden 自動挑選與切塊投票純化：** Stage 1 抽樣並以四合一指標初篩 Top-16 候選晶圓；Stage 2 於 RAM 內切塊跨片同座標 PK 投票，選出最終 4 張 Golden 原圖與 48 張純淨訓練圖塊。 | `--pool_dir` 無標註盲池圖片資料夾（支援遞迴掃描子目錄與 8/16-bit TIFF） | - 訓練圖塊：`./datasets/<data_name>/train/<category>/`<br>- 原圖備份：`./datasets/<data_name>/golden_full_res_backup/<category>/`<br>- 候選池與快取：`stage1_candidates_*.json`、`golden_cache_meta.json` | `<FAB_RESULT_DIR>/golden_reports/`（含 `selected_4_golden_images.txt` 與 `golden_tile_purity_report.txt`） |
| **Step 2**<br>`2_train_fab_no_gt.py`<br>`run_2_train_fab_no_gt.sh` | **無 GT 模型訓練與防呆對接：** 自動檢查既有 Golden 與已訓練權重（防呆不重挑、不重訓），動態分配空閒 DDP Port 執行訓練，並自動對接 `pretrained.pth.tar`。 | `./datasets/<data_name>/train/<category>/` 訓練樣本 | - 訓練權重：`./logs/<data_name>/dinov3<diy_name>/train-step*.pth.tar` & `pretrained.pth.tar`<br>- 訓練紀錄：`./datasets/<data_name>/train_meta_<category>.json` | `<FAB_RESULT_DIR>/train_<YYYYMMDD_HHMMSS>/train_log.txt` |
| **Step 3**<br>`3_infer_fab_no_gt.py`<br>`run_3_infer_fab_no_gt.sh` | **高解析度切塊推論、RAM 縫合與防誤報畫框：** >1280px 自動切塊推論，於 RAM 內餘弦加權縫合並**立即銷毀碎片 `.npy`**；經四合一濾波、左半核心雙重校準門檻與過擬合自檢後輸出紅框與報表。 | - 測試盲池：`--pool_dir`<br>- 訓練權重：`./logs/<data_name>/.../pretrained.pth.tar` | - 測試切塊快取：`./datasets/<data_name>_tiled_pool/`<br>- 切塊座標映射：`./datasets/<data_name>_cached_tile_manifest.json` | `<FAB_RESULT_DIR>/<YYYYMMDD_HHMMSS>/`（含 `anomaly_images/`、`amap_npy/`、`escape_ranking_report.csv/.txt`） |
| **Step 4**<br>`4_split_eval_results.py`<br>`run_4_split_eval_results.sh` | **推論結果多包拆分器：** 將 Step 3 產出之排名報表、異常紅框圖與 `.npy` 依分層抽樣（`stratified`）或指定比例/張數拆分為 N 包獨立評估集。 | Step 3 輸出之 `<FAB_RESULT_DIR>`（自動鎖定最新時間戳記目錄） | 無（支援 `--symlink` 軟連結零空間負擔分包） | `<FAB_RESULT_DIR>/<時間戳>/split_eval_<N>packs/eval_pack_01 ~ 0N/` |
| **廠內多層總控**<br>`run_all_layers.sh` | **多製程層全自動流水線：** 依序針對多個晶圓 Layer（如 `TMQB28`、`TMTN86`、`TMUN87`）自動串聯執行 `run_1` -> `run_2` -> `run_3` -> `run_4`。 | `/workspace/dataset/DA_Wafer_use/<DATASET_ID>/<LAYER_ID>/other_status` | 各 Layer 獨立之 `datasets/wafer_<LAYER_ID>_4shot` 與 `logs/` 權重 | `./wafer_escape_results/<DATASET_ID>/<LAYER_ID>/` |
| **廠外多類別總控**<br>`run_outside.sh` | **公開資料集四階段模擬抓漏：** 支援 MVTec AD / MVTec AD 2 多類別自動建立含隱藏瑕疵之模擬盲池，並依序跑完 Step 1 ~ Step 4 與印出 Recall / FPR / 消融對照。 | `./datasets/mvtec/<類別>` 或 `./datasets/mvtec-2/<類別>` | `./datasets/sim_fab_<類別>_pool/` 與對應 `logs/` 權重 | `./output/2_fab_no_gt_<類別>/<YYYYMMDD_HHMMSS>/` |

---

### 3.2 四階段核心演算法與工程亮點詳解

#### (1) `1_auto_golden.py` — 大資料集抽樣 + Top-16 候選池四合一過濾 + Stage 2 先切塊再投票選 Golden
1. **Stage 1 大資料集快速抽樣與四合一純淨度初篩 (`select_golden_images`)：**
   * **大資料集抽樣防護 (`--max_coarse_scan 250`)：** 面對上千張晶圓盲池時，自動以固定亂數種子均勻抽樣 250 張進行全域評估，徹底消除全池 I/O 瓶頸。
   * **光度與對比度穩健門檻 (Photometric Gate)：** 計算亮度中位數、標準差與 P5 ~ P95 跨度之 MAD Z-score，自動攔截曝光不足 (`underexposed`) 與過曝 (`overexposed`) 樣本。
   * **空間網格對齊中心度 (Spatial Grid Alignment)：** 提取 8x8 空間池化特徵與跨片中位數比對，攔截平移或旋轉偏移 (`shift_1` / `shift_2`) 樣本。
   * **跨片同座標局部 Patch 異常峰值懲罰：** 比對 16x16 局部座標與單圖內部 Token Spike，攔截帶有局部大瑕疵的樣本。
   * **近鄰多樣性去重 (Diversity NMS)：** 防止同一顆工件的多重拍攝分身同時霸佔候選池，鎖定不重複的 **Top-16 候選晶圓池 (`--candidate_pool_size 16`)**。
2. **Stage 2 先切塊後投票選 Golden 與二次純化 (`prepare_tiled_golden_trainset`)：**
   * **智慧切塊觸發門檻 (`--min_tile_trigger 1280`)：** 當原圖尺寸 <= 1280px（如 MVTec 1000x1000）時，自動維持整張原圖訓練以保留完整空間結構對齊；當晶圓大圖 > 1280px 時才啟動高解析度切塊。
   * **跨片同座標切塊 PK 投票：** 將 Stage 1 選出的 16 張候選晶圓於 RAM 內切塊，在每個非背景網格座標 `(y, x)` 結合「零均值標準化結構殘差 + 像素中位數殘差 + 曝光偏離懲罰 + 輕量 DINO 圖塊特徵距離」進行同座標對決。
   * **統計勝出票數選出最終 4 張 Golden：** 統計 16 張候選圖在所有座標獲得前 2 名的勝出票數（`Win_Votes`），正式選出最終 **Top-4 張最純淨 Golden 原圖**備份至 `golden_full_res_backup/`，同時跨候選池徑向分層選出 **48 張最純淨訓練圖塊**輸出至 `train/<category>/`，並生成 `golden_tile_purity_report.txt`。

#### (2) `2_train_fab_no_gt.py` — 智慧防呆快取、動態步數換算與無衝突 DDP 訓練
* **雙重防呆略過機制：**
  * 只要 `datasets/<data_name>/train/<category>` 已存在挑選好的 Golden 樣本且未指定 `--rebuild_golden`，自動直接沿用，絕不重複挑選；若偵測到舊版誤將 <= 1280px 圖片切成碎片，還會自動從 `golden_full_res_backup` 無損還原完整 Golden 原圖。
  * 嚴格檢查當前 `data_name` 專屬 `./logs/<data_name>/` 下是否已有訓練完成的權重檔；若已存在且訓練集結構未變動，自動跳過訓練階段（若想強制重訓可加 `--force_train`）。
* **動態空閒 Port 分配 (`get_free_port`)：** 每次啟動子程序時自動由作業系統分配保證空閒的 `FAB_MASTER_PORT`，徹底根除多人共用伺服器或多層連跑時的 `Port 40112 Address already in use` 錯誤。

#### (3) `3_infer_fab_no_gt.py` — RAM 即時縫合銷毀碎片、輕量微塵抑制、左半核心雙重校準門檻與過擬合自檢
1. **測試池切塊快取與 RAM 即時縫合銷毀碎片：**
   * 自動驗證盲池 MD5 指紋與切塊參數（含 `min_tile_trigger=1280`），若未變動則秒速沿用既有推論圖塊 PNG 快取。
   * 推論完成後，於 RAM 內將單張大圖的所有子圖塊 `.npy` 以 2D 餘弦平滑權重縫合回完整解析度，**並在拼回整張 Wafer 的瞬間立刻刪除該圖所有碎片 `.npy`**，確保硬碟零碎片殘留。
2. **升級版四合一防誤報熱力圖純化器 (`refine_amap_for_wafer_and_noise`)：**
   * **跨 Golden 結構複雜區容差補償 (`golden_tol_weight=0.15`)：** 溫和扣除 Golden 圖在固定結構區的背景基準值。
   * **3x3 輕量單點微塵抑制：** 採用固定 3x3 橢圓核開運算結合高斯平滑，濾除單點微塵噪點的同時，**完整保護細小裂痕（Crack）與微米級針孔（Pinhole / Poke）不被大核吃掉**。
   * **矩形邊框抑制 (`--border_ratio 0.02`)** 與 **智慧圓形晶圓倒角抑制 (`--wafer_edge_mode [auto|on|off]`, `--wafer_edge_margin 0.04`)**：內建長寬比（>= 0.85）與圓形度（>= 0.82）雙重幾何檢驗，精準消除圓形晶圓外圈 Bevel 反光，且絕不會誤傷長條形工件（如 Capsule）頭尾。
   * **支援未濾波對照模式 (`--raw_score_mode`)：** 報表內建 Ablation 對照列，同時記錄每張圖的 `Raw_Max_Score`（未濾波單點最大值）與 `Filtered_Max_Score`（四合一純化後抗噪峰值）及差值 `Delta`。
3. **真無 GT 左半部正常核心估計器 + Golden 基準雙重校準動態門檻 (`estimate_blind_normal_core_stats`)：**
   * 完全不依賴任何檔名標籤，直接從盲池分佈的左半部（低於中位數/Golden 錨點）穩健估計正常樣本中心 `mu_core` 與左半單邊標準差 `sigma_left`，並與 Golden 訓練圖基準上界雙重校準，**徹底解決當盲池內異常比例偏高時，傳統 `mean + k*std` 或 `P95` 被異常高分拉爆導致門檻暴衝漏抓的問題**。
4. **無監督模型過擬合自檢診斷 (`evaluate_overfitting_diagnostics`)：**
   * 自動比對「Golden 訓練圖平均分數」與「盲池正常核心中位數」之比值（`Overfit_Ratio`）與差距（`Gap`），於終端機與報表自動標示 `🟢 良好`、`🟡 輕微過擬合傾向` 或 `🔴 過擬合警告` 並提供調參建議。

---

## 4. 執行訓練與測試操作指南（公司內實測 vs 公司外測試）

執行腳本前，請先確認已啟動環境並位於專案根目錄 `FoundAD/`：

```bash
conda activate foundad
cd /path/to/FoundAD
```

### 4.1 公司內實測：分步執行或一鍵批次跑完多製程層 (Multi-Layer)

#### 方式 A：使用 `run_all_layers.sh` 一鍵自動跑完所有製程層（Step 1 -> Step 4 全自動）
打開 `run_all_layers.sh` 確認要執行的 `DATASET_ID`、`LAYER_ID` 與盲池路徑後，直接執行：
```bash
# 前景執行
bash run_all_layers.sh

# 或使用 nohup 掛在背景執行 (推薦，防止 SSH 斷線)
nohup bash run_all_layers.sh > train_all_layers.log 2>&1 &
```

#### 方式 B：針對單一製程層（Layer）分階段獨立執行與調參
四支 `.sh` 腳本皆支援直接執行（使用腳本內預設值）或透過命令列傳入 `<DATASET_ID> <LAYER_ID> [WAFER_POOL_DIR]`：

```bash
# Step 1: 自動從盲池初篩 16 張候選圖並以 RAM 切塊投票選出 4 張 Golden 與 48 張純淨訓練圖塊
bash run_1_auto_golden.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 2: 載入 Golden 訓練集並執行 FoundAD 2000 步訓練 (若已訓練過會自動防呆跳過)
bash run_2_train_fab_no_gt.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 3: 執行測試池切塊推論、RAM 即時縫合銷毀碎片、四合一防誤報濾波與紅框輸出
bash run_3_infer_fab_no_gt.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 4: 將最新一次推論結果依分層抽樣 (stratified) 自動拆分為 3 包獨立 Eval 報表與圖片包
bash run_4_split_eval_results.sh W639 TMQB28
```

### 4.2 已訓練完成後，如何「零訓練成本」秒速重調門檻或防誤報參數？
由於訓練（`2_train_fab_no_gt.py`）與推論（`3_infer_fab_no_gt.py`）已完全解耦，當你想嘗試不同的門檻模式（如 `hybrid`、`mean_std`、`fixed`）、調整標準差倍數 `SIGMA_K`、晶圓邊緣抑制寬度 `WAFER_EDGE_MARGIN`，或切換 `--raw_score_mode` 進行消融實驗時，**完全不需要動到 Step 1 與 Step 2**！

只需打開 `run_3_infer_fab_no_gt.sh` 修改第 2、3 區參數：
```bash
THRESH_MODE="mean_std"     # 支援: "hybrid"、"mean_std"、"percentile"、"top_k"、"fixed"
SIGMA_K=3.5                # 調整標準差倍數 k (調小抓更細微瑕疵，調大更嚴格防誤報)
WAFER_EDGE_MODE="on"       # 圓形晶圓設 "on"
WAFER_EDGE_MARGIN=0.04     # 晶圓最外圈 4% 半徑平滑衰減
RAW_SCORE_MODE=""          # 若想對照未濾波原生分數可設為 "--raw_score_mode"
```
接著直接重跑 Step 3（與 Step 4）即可（測試池切塊 PNG 會自動命中快取，0 秒完成切塊準備）：
```bash
bash run_3_infer_fab_no_gt.sh W639 TMQB28
bash run_4_split_eval_results.sh W639 TMQB28
```

### 4.3 公司外公開資料集測試（`run_outside.sh`）
在外部環境可一鍵測試預設的 7 個公開資料集類別（或直接由命令列傳入指定類別路徑），自動串聯 Step 1 ~ Step 4：
```bash
# 執行預設類別清單 (capsule, mvtec-2/can, bottle, cable, grid, tile, transistor)
bash run_outside.sh

# 或指定單一/多個類別資料夾快速測試
bash run_outside.sh ./datasets/mvtec/capsule ./datasets/mvtec-2/can
```

#### 常用進度監控指令：
```bash
# 廠內即時查看最新訓練或推論進度
tail -f ./wafer_escape_results/W639/TMQB28/latest_train_log.txt
tail -f ./wafer_escape_results/W639/TMQB28/latest_log.txt

# 廠外即時查看最新進度
tail -f ./output/2_fab_no_gt_capsule/latest_log.txt
```

---

## 5. 確認訓練完的權重檔與 Golden 資料集位置

* **Golden 訓練集與候選池投票紀錄（`./datasets/<DATA_NAME>/`）：**
  * 廠內路徑：`./datasets/wafer_<LAYER_ID>_4shot/`
    * `train/wafer/`：最終用於訓練的 48 張代表性純淨子圖塊（或完整原圖）。
    * `golden_full_res_backup/wafer/`：切塊投票勝出的 4 張全解析度 Golden 原圖備份。
    * `stage1_candidates_wafer.json`：Stage 1 初篩 Top-16 候選晶圓詳細指標。
    * `golden_tile_purity_report.txt`：Stage 2 候選池切塊投票排行榜與各訓練圖塊座標清單。
    * `train_meta_wafer.json`：記錄訓練完成之步數、耗時與權重檔絕對路徑。
* **模型權重檔位置（`./logs/<DATA_NAME>/`）：**
  * 廠內路徑：`./logs/wafer_<LAYER_ID>_4shot/dinov3fab_<LAYER_ID>/train-step2000.pth.tar`（及自動對接之 `pretrained.pth.tar`）
  * 廠外路徑：`./logs/mvtec_sim_<類別>_4shot/dinov3sim_escape_<類別>/train-step*.pth.tar`

可使用以下指令快速確認所有已生成的權重檔：
```bash
ls -lh ./logs/*/*/*.pth.tar
```

---

## 6. 檢驗測試結果與輸出報表說明 (`./wafer_escape_results/` 與 `./output/`)

每次執行 `3_infer_fab_no_gt.py` 與 `4_split_eval_results.py` 後，會在 `<FAB_RESULT_DIR>/<YYYYMMDD_HHMMSS>/` 下生成完整產出：

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `anomaly_images/` | **【核心產出 1】異常紅框三合一對照圖（僅輸出超過有效門檻之異常圖）：**<br>檔名格式為 `Rank001_Score0.3450_Box01_<原檔名>.png`（依異常分數由高至低排序，第一張即為最可疑的 Escape 晶圓）。<br>頂部橫幅顯示 `[FILT/RAW]`、原檔名、最終分數、未濾波原始分、濾波後分數、有效門檻與紅框數；下方為 `[左：原始圖 -> 中：純化後異常熱力疊圖 -> 右：紅色 Bounding Box 瑕疵定位框]`。 |
| `anomaly_images/full_res_bbox_only/` | **【核心產出 2】1:1 原始全解析度紅框定位圖：**<br>維持原圖 100% 原始解析度（例如 5000x5000 不縮放），僅在原圖畫上紅色瑕疵框與局部分數標籤，供工程師放大檢視微米級缺陷細節。 |
| `escape_ranking_report.txt` | **無 GT 異常抓漏總排名、消融對照與過擬合診斷報表（純文字版）：**<br>1. 標頭完整記錄計分模式、有效門檻、盲池正常核心分數分佈、**濾波前後正常圖對照 (Ablation)**、`mean + 1*std / 2*std / 3*std` 門檻參考值，以及 **Overfit Diagnostic 過擬合自檢結果**。<br>2. 表格依序列出所有圖片之 `Rank`、`Original_Filename`、`Status`（`ANOMALY` / `GOLDEN_PASS` / `PASS_SKIPPED` / `PASS_NOISE_FILTERED`）、`Max_Score`、`Raw_Score`、`Filt_Score`、`Delta`、`BBoxes` 數量與對應輸出檔名。<br>3. 末尾自動附加訓練與推論各階段詳細耗時總結。 |
| `escape_ranking_report.csv` | 同上之完整 CSV 報表（含 `Max_Score`、`Raw_Max_Score`、`Filtered_Max_Score`、`Filter_Delta`、`TopK_Mean_Score`、`Num_BBoxes`、`Resolution`），可直接用 Excel 開啟篩選。 |
| `golden_tile_purity_report.txt` | **Stage 2 候選池切塊投票排行榜與圖塊純化報表：** 完整列出 16 張候選晶圓的切塊得票數（`Win_Votes`）、平均圖塊殘差成本，以及最終選出的 4 張 Golden 與 48 張訓練圖塊來源。 |
| `selected_4_golden_images.txt` | 最終勝出之 4 張 Golden Image 原檔名、切塊勝出票數與特徵評分紀錄。 |
| `amap_npy/` | **異常樣本之全解析度熱力圖矩陣（`.npy`）：** 僅儲存超過門檻之異常圖片縫合且濾波後的整張 Wafer 2D 矩陣（子圖塊碎片已於 RAM 拼接後立即自動銷毀）。 |
| `split_eval_3packs/` | **【Step 4 產出】獨立拆分評估結果包（`eval_pack_01` ~ `03`）：**<br>每包皆含獨立重排的 `escape_ranking_report.csv/.txt` 與對應的 `anomaly_images/`、`amap_npy/`，方便交由不同工程師平行複檢或交叉驗證。 |
| `log.txt` | 本次推論執行的完整終端機日誌與時間統計。 |

---

## 7. 常見問題與預先排錯指南（Troubleshooting）

### Q1：在公司內執行時報錯 `URLError` / `ConnectionError`（嘗試連網下載 GitHub 或 HuggingFace）
* **原因：** 公司內網阻擋連向 GitHub (`torch.hub.load`) 或 HuggingFace。
* **解法：**
  1. 確認已照 **第 1.3 節 Step 1** 將外部機器 `~/.cache/torch/hub/*dino*` 複製到專案下的 `./hub_cache/`。
  2. 確認 `run_1_auto_golden.sh`、`run_2_train_fab_no_gt.sh`、`run_3_infer_fab_no_gt.sh` 內的 `COMPANY_DINO_CKPT` 已正確指向公司伺服器上的 `.pth` 權重檔路徑，且 `FAB_LOCAL_DINO_REPO` 指向 `./hub_cache/facebookresearch_dinov3_main`。

### Q2：如何根據報表中的「盲池正常核心分數分佈」與「過擬合自檢」微調門檻？
* **解法：**
  每次跑完 `run_3_infer_fab_no_gt.sh` 或 `run_outside.sh`，終端機與報表都會印出診斷面板：
  ```text
  📊 盲池正常核心分數分佈 (四合一濾波後) : Max=0.2410 | P99=0.2350 | P95=0.2180 | Median=0.1850 | Mean=0.1880 | Std=0.0160
  📐 盲池正常核心門檻參考值      : mean+1*std=0.2040 | mean+2*std=0.2200 | mean+3*std=0.2360
  🔬 濾波前後正常圖對照 (Ablation): [未濾波 Raw] Max=0.3120, Mean=0.2250  vs  [四合一濾波] Max=0.2410, Mean=0.1880
  🩺 模型過擬合自檢 (Overfit)    : 🟢 良好: 無過擬合現象 (Golden平均=0.1790 vs 盲池正常中位數=0.1850 | Overfit_Ratio=1.03x)
  ```
  * **若出現 `🔴 警告: 偵測到明顯過擬合` 或正常晶圓誤報偏多：** 可將 `run_3_infer_fab_no_gt.sh` 中的 `THRESH_MODE` 設為 `"mean_std"`、將 `SIGMA_K` 調高至 `3.5` 或 `4.0` 直接重跑 Step 3；或在 Step 2 將 `CKPT_STEP` 由 `2000` 調降為 `1000` 並加上 `--force_train` 重訓。
  * **若想抓出更微弱的細小瑕疵：** 將 `run_3_infer_fab_no_gt.sh` 中的 `SIGMA_K` 調小為 `2.5` 或 `3.0`，或設定 `THRESH_MODE="fixed"` 搭配指定的 `THRESHOLD` 重跑 Step 3。
  * **若晶圓最外圈圓弧邊緣反光被誤框：** 確認 `WAFER_EDGE_MODE="on"`，並將 `WAFER_EDGE_MARGIN=0.04` 調大為 `0.06`（代表外圈 6% 半徑平滑抑制）。

### Q3：想強制重新挑選 Golden Image 或強制重新訓練模型該怎麼做？
* **解法：** 為避免浪費 GPU 時間，系統預設開啟防呆快取（有 Golden 不重挑、有權重不重訓）。
  * **強制重挑 Golden：** 在執行 `1_auto_golden.py` 或 `2_train_fab_no_gt.py` 時加上 `--rebuild_golden`（或在 `run_2_train_fab_no_gt.sh` 中設定 `REBUILD_GOLDEN="--rebuild_golden"`）。
  * **強制重新訓練模型：** 在執行 `2_train_fab_no_gt.py` 時加上 `--force_train`。
  * **強制重新切塊測試池：** 在執行 `3_infer_fab_no_gt.py` 時加上 `--force_infer`（或在 `run_3_infer_fab_no_gt.sh` 中設定 `FORCE_INFER="--force_infer"`）。