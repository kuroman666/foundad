# FoundAD 晶圓瑕疵抓漏（Escape 檢測）系統 — 完整建置與操作指南

本專案基於 FoundAD (Few-Shot Anomaly Detection) 架構進行客製化擴充，專為「晶圓廠工程師初篩為正常品、完全無標註（No Ground Truth）之高解析度 Wafer 影像池」設計。系統全面升級為 **四階段模組化無 GT 抓漏、跨批號推論與獨立門檻拆分評估流水線（4-Stage Modular Pipeline）**，支援三大核心驗證與部署場景：

* **模式一：公司內無 GT 晶圓廠盲池四階段抓漏與跨批號推論模式（`run_all_layers.sh` 或單獨執行 `run_1` ~ `run_4` / 廠內即插即用）：**
  1. **Step 1 自動挑選與切塊投票純化 Golden（`1_auto_golden.py` / `run_1_auto_golden.sh`）：** 支援上千張大資料集固定種子快速抽樣（預設上限 250 張），結合「光度對比過濾 + 空間網格對齊 + 跨片局部異常峰值 + 近鄰多樣性去重 (Diversity NMS)」四合一初篩鎖定 Top-16 候選池；接著於 RAM 內切塊進行跨片同座標 PK 投票，選出最終 4 張最純淨 Golden 原圖與 48 張代表性高解析度訓練圖塊（若影像 <= 1280px 自動維持整張原圖訓練），並自動繪製 **48 個訓練圖塊與同心圓環帶空間分佈圖（`golden_tile_layout_vis.png`）**。
  2. **Step 2 無 GT 高解析度流形訓練（`2_train_fab_no_gt.py` / `run_2_train_fab_no_gt.sh`）：** 內建雙重防呆快取（有 Golden 不重挑、有權重不重訓，若訓練集結構升級則自動重訓）、動態分配空閒本地 Port（徹底解決 DDP Port 40112 衝突）、內建 `.offline_hub_patch` 自動攔截 `torch.hub.load` 離線防 504 保護，並自動儲存 `train_meta_<category>.json` 與對接 `pretrained.pth.tar`。
  3. **Step 3 高解析度切塊推論、跨資料集基準對接、微瑕疵增強與自適應左半核心門檻（`3_infer_fab_no_gt.py` / `run_3_infer_fab_no_gt.sh`）：**
     * **跨資料集 / 跨批號直接推論（`--golden_src` & `--trained_ckpt`）：** 支援直接指定舊資料集挑過的 Golden Image 與訓練好的 `.pth.tar` 權重直接推論新批號盲池，自動將外部 Golden 注入推論池計算空間結構容差與動態門檻基準（且不混入新測試集排名報表）。
     * **RAM 即時縫合與 0 碎片殘留：** 針對 > 1280px 大圖自動切塊推論（<= 1280px 直接 1:1 整張推論），於 RAM 內以 2D 餘弦平滑權重（Raised-Cosine Window）無縫拼接回全解析度，並 **立即自動銷毀所有子圖塊 `.npy` 碎片（0 殘留）**。
     * **三種計分與濾波模式：** 支援 `MICRO_DEFECT_SENSITIVE`（預設開啟 `--micro_defect_mode`：微瑕疵白頂帽局部尖峰增強 + Top-24px 防稀釋計分 + 5x5 閉運算微框保護）、`FILTERED_ROBUST`（標準四合一防誤報純化）與 `RAW_UNFILTERED`（`--raw_score_mode` 未濾波原生最大值消融對照）。
     * **自適應未見過正常核心估計器（Adaptive Unseen-Normal Left-Core Estimator）與自動記憶效應補償：** 直接針對非 Golden 盲池左半部（前 55% 低分群之 P75）鎖定未見過正常核心 `mu_core` 與左半標準差 `sigma_left`，自動排除 4-shot Golden 記憶壓低效應與跨批號微偏移，精準框出紅色 Bounding Box 並輸出異常排名報表（正常圖片自動略過不輸出）。
  4. **Step 4 評估結果自動分包與各包獨立門檻重算器（`4_split_eval_results.py` / `run_4_split_eval_results.sh`）：** 支援依「每包指定張數 (`--counts`)」或「比例 (`--ratios`)」搭配分層抽樣（`stratified`）、隨機、排名或檔名拆分為 N 份獨立 Eval 報表與圖片包；內建**拆分總張數嚴格防呆核對**與**各包獨立重算動態門檻（`--recompute_pack_thresh`）**，並自動產出跨包對照總覽表（`all_<N>_evals_comparison.txt/.csv`）。
* **模式二：廠外公開資料集無 GT 模擬抓漏與純正常池 FPR 驗證模式（`run_outside.sh`）：** 支援 MVTec AD 與 MVTec AD 2 多類別一鍵完成「建立獨立模擬盲池（支援 `escape` 抓漏模式與 `pure_normal` 純正常池 FPR 測試，正常圖不足時自動從 `validation/good` 或 `train/good` 補齊） -> Step 1 Auto-Golden -> Step 2 訓練 -> Step 3 推論 -> Step 4 拆分評估」，各資料集與類別獨立命名不覆蓋，並自動匯出跨資料集總表 `latest_multi_dataset_summary.txt/.csv`。
* **模式三：有 GT 公開資料集基準驗證模式（`fab_tools/1_run_public_with_gt.py`）：** 自動完成少樣本抽樣、訓練，並將 `AUROC_i`、`AUPR_i`、`AUROC_p`、`PRO-AUC` 分數與各階段耗時匯出為 `.txt` 與 `.csv` 報表。

---

## 1. 環境、套件安裝（含公司內斷網環境準備）

### 1.1 基礎環境需求
* **作業系統：** Linux (Ubuntu 20.04 / 22.04+)
* **Python 版本：** Python 3.10 ~ 3.12
* **GPU 硬體：** 外部測試機 或 廠內伺服器（如 NVIDIA H100 / A100 / RTX 系列）

### 1.2 情況 A：在「有網路」環境安裝（或進公司前打包環境）
請依序執行以下指令建立 Conda 虛擬環境並安裝套件（亦可直接使用專案根目錄之 `requirements.txt`）：
```bash
# 1. 建立並啟動虛擬環境
conda create -n foundad python=3.10 -y
conda activate foundad

# 2. 安裝 PyTorch (請依機器之 CUDA 版本選擇，例如 CUDA 12.1 / 12.4)
pip install torch torchvision torchaudio

# 3. 安裝專案所需核心套件 (或執行 pip install -r requirements.txt)
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
*註：* 本專案 `2_train_fab_no_gt.py` 與 `3_infer_fab_no_gt.py` 已內建 `.offline_hub_patch/sitecustomize.py` 自動攔截機制，會優先從本地快取載入 DINO 架構，徹底杜絕 GitHub 連線失敗或 HTTP 504 Timeout；若使用 `dinov3`，請保留 `./hub_cache/facebookresearch_dinov3_main` 資料夾。

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

每次執行實驗時，系統皆會依資料集（`DATASET_ID`）、製程層（`LAYER_ID`）、類別與微秒級時間戳記（`<YYYYMMDD_HHMMSS_fff>`）建立獨立子資料夾，**絕不覆蓋歷史實驗結果**：

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
├── fab_tools/                                # [輔助與封存工具目錄]
│   ├── 1_run_public_with_gt.py               # [輔助程式] 有 GT 公開資料集抽樣 + 訓練 + 評估自動化腳本
│   ├── 2_run_fab_no_gt.py                    # (舊版單檔整合流水線備份)
│   └── bbox_visualizer(unuse_now).py         # (舊版輔助視覺化工具模組備份)
├── hub_cache/                                # [離線快取] 存放 facebookresearch_dinov3_main 等本地架構檔
├── .offline_hub_patch/                       # [自動生成] sitecustomize.py 自動攔截 torch.hub 離線防 504 模組
├── datasets/                                 # [資料集與自動生成訓練集/快取專區]
│   ├── mvtec/ & mvtec-2/                     # (Input) 原始 MVTec AD / MVTec AD 2 資料集
│   ├── wafer_<LAYER_ID>_4shot/               # (Step 1 自動生成) 廠內各製程層專屬 Golden 資料集
│   │   ├── train/wafer/                      # 切塊投票純化後之 48 張代表性訓練圖塊 (或 <=1280px 完整原圖)
│   │   ├── golden_full_res_backup/wafer/     # 切塊投票最終勝出的 4 張全解析度 Golden 原圖備份
│   │   ├── golden_tile_layout_vis.png        # [新增] 48 個代表性訓練圖塊與同心圓環帶空間分佈視覺化對照圖
│   │   ├── stage1_candidates_wafer.json      # Stage 1 初篩鎖定之 Top-16 候選晶圓詳細評分紀錄
│   │   ├── golden_tile_purity_report.txt     # Stage 2 候選池切塊投票排行榜與圖塊純化報表
│   │   ├── golden_cache_meta.json            # 盲池 MD5 指紋與 Golden 快取紀錄
│   │   └── train_meta_wafer.json             # Step 2 訓練完成後記錄之實際步數、耗時與權重路徑 Meta
│   ├── wafer_<LAYER_ID>_4shot_tiled_pool/    # (Step 3 自動生成) 測試盲池切分後的重疊推論子圖塊 PNG 快取
│   ├── wafer_<LAYER_ID>_4shot_cached_tile_manifest.json # (Step 3 自動生成) 測試池與外部 Golden 切塊座標映射快取
│   └── sim_fab_<DS>_<類別>_pool/             # (廠外自動生成) 模擬無 GT 盲池 (含 clean_* 與隱藏 escape_* 瑕疵圖)
├── logs/                                     # [模型權重區] 訓練自動產生的 .pth.tar 權重檔與 params.yaml
│   ├── wafer_<LAYER_ID>_4shot/dinov3fab_<LAYER_ID>/         # 廠內各 Layer 獨立訓練權重 (train-step*.pth.tar & pretrained.pth.tar)
│   └── <DS>_sim_<類別>_4shot/dinov3sim_escape_<類別>/       # 廠外公開資料集各類別獨立訓練權重
├── output/                                   # [廠外測試唯一結果出口] 依任務、資料集、類別與時間戳記分類存放
│   ├── 1_public_with_gt/<YYYYMMDD_HHMMSS>/   # Task 1 有 GT 公開資料集評估結果
│   ├── 2_fab_no_gt_<類別>/                   # 廠外無 GT 模擬盲池抓漏結果 (含 train_* 與 infer_* 子目錄)
│   ├── 3_pure_normal_<DS>_<類別>/            # 100% 純正常池誤報率 (FPR) 驗證結果 (POOL_MODE="pure_normal")
│   ├── multi_dataset_batch_summary_<時間戳>.txt # [新增] 廠外多資料集批次測試彙整總覽報表
│   └── latest_multi_dataset_summary.txt/.csv # [新增] 最新一次多資料集批次測試總表 (TXT 與 CSV)
├── wafer_escape_results/                     # [公司內唯一結果出口] 實測高解析度晶圓之輸出根目錄
│   └── <DATASET_ID>/<LAYER_ID>/              # 依資料集 (如 W639) 與製程層 (如 TMQB28) 分層管理
│       ├── golden_reports/                   # Step 1 輸出之 Golden 名單、純化報表與 golden_tile_layout_vis.png
│       ├── train_<DATA_NAME>_<CAT>_<時間戳>/ # Step 2 訓練階段專屬日誌目錄 (train_log.txt & train_meta_*.json)
│       ├── infer_<DATA_NAME>_<CAT>_<時間戳>/ # Step 3 推論階段專屬結果目錄 (含紅框圖、排名報表與 split_eval_<N>packs/)
│       ├── latest_train_log_<CAT>.txt & latest_train_log.txt # 最新一次訓練階段即時日誌
│       └── latest_infer_log_<CAT>.txt & latest_log.txt       # 最新一次推論階段即時日誌
├── 1_auto_golden.py                          # [核心程式 1] Stage 1 大資料集抽樣候選池 + Stage 2 切塊投票選 Golden 與同心圓視覺化
├── 2_train_fab_no_gt.py                      # [核心程式 2] 無 GT Golden 資料集載入防呆 + 離線防 504 + FoundAD 模型訓練
├── 3_infer_fab_no_gt.py                      # [核心程式 3] 跨批號指定 Golden/權重 + RAM 縫合銷毀碎片 + 微瑕疵增強 + 自適應左半核心門檻
├── 4_split_eval_results.py                   # [核心程式 4] 推論結果拆分器 (支援指定包數/張數、總數核對防呆與各包獨立重算門檻)
├── run_1_auto_golden.sh                      # [廠內腳本 1] 單獨執行 Step 1 Auto-Golden 挑選、切塊純化與分佈圖繪製
├── run_2_train_fab_no_gt.sh                  # [廠內腳本 2] 單獨執行 Step 2 無 GT 模型訓練
├── run_3_infer_fab_no_gt.sh                  # [廠內腳本 3] 單獨執行 Step 3 無 GT 推論、跨資料集指定 Golden/權重與重調門檻
├── run_4_split_eval_results.sh               # [廠內腳本 4] 單獨執行 Step 4 評估結果拆分為多包與各包獨立門檻重算
├── run_all_layers.sh                         # [廠內總控腳本] 多製程層 (Multi-Layer) 四階段全自動批次執行腳本
├── run_outside.sh                            # [廠外總控腳本] 公開資料集 (MVTec / MVTec-2) 四階段無 GT 模擬抓漏與跨資料集總表彙整
└── requirements.txt                          # Python 環境依賴套件清單
```

---

## 3. 核心檔案用途、演算法亮點與 Input / Output 路徑對照總表

### 3.1 檔案一覽與 Input / Output 路徑速查表

| 階段 / 檔案名稱 | 核心目的與功能 | Input (輸入資料夾 / 檔案) | 中間產物與權重位置 (`datasets/` & `logs/`) | Output (最終結果輸出資料夾) |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1**<br>`1_auto_golden.py`<br>`run_1_auto_golden.sh` | **兩階段 Golden 自動挑選、切塊投票純化與同心圓視覺化：** Stage 1 抽樣並以四合一指標初篩 Top-16 候選晶圓；Stage 2 於 RAM 內切塊跨片同座標 PK 投票，選出最終 4 張 Golden 原圖與 48 張純淨訓練圖塊，並繪製切塊空間分佈圖。 | `--pool_dir` 無標註盲池圖片資料夾（支援遞迴掃描子目錄與 8/16-bit TIFF） | - 訓練圖塊：`./datasets/<data_name>/train/<category>/`<br>- 原圖備份：`./datasets/<data_name>/golden_full_res_backup/<category>/`<br>- 視覺化圖：`golden_tile_layout_vis.png`<br>- 候選池與快取：`stage1_candidates_*.json`、`golden_cache_meta.json` | `<FAB_RESULT_DIR>/golden_reports/`（含 `selected_4_golden_images.txt`、`golden_tile_purity_report.txt` 與 `golden_tile_layout_vis.png`） |
| **Step 2**<br>`2_train_fab_no_gt.py`<br>`run_2_train_fab_no_gt.sh` | **無 GT 模型訓練與防呆對接：** 自動檢查既有 Golden 與已訓練權重（防呆不重挑、不重訓），動態分配空閒 DDP Port、自動注入離線 Hub Patch 防 504，並對接 `pretrained.pth.tar`。 | `./datasets/<data_name>/train/<category>/` 訓練樣本 | - 訓練權重：`./logs/<data_name>/dinov3<diy_name>/train-step*.pth.tar` & `pretrained.pth.tar`<br>- 訓練紀錄：`./datasets/<data_name>/train_meta_<category>.json` | `<FAB_RESULT_DIR>/train_<data_name>_<cat>_<時間戳>/train_log.txt` |
| **Step 3**<br>`3_infer_fab_no_gt.py`<br>`run_3_infer_fab_no_gt.sh` | **高解析度切塊推論、跨批號基準對接、RAM 縫合與微瑕疵增強畫框：** 支援 `--golden_src` 與 `--trained_ckpt` 跨資料集推論；>1280px 自動切塊推論，於 RAM 內餘弦加權縫合並**立即銷毀碎片 `.npy`**；支援微瑕疵白頂帽增強（`--micro_defect_mode`）、自適應未見過正常左半核心門檻與過擬合自檢。 | - 測試盲池：`--pool_dir`<br>- 預設或自訂權重：`--trained_ckpt`<br>- 預設或自訂 Golden：`--golden_src` | - 測試切塊快取：`./datasets/<data_name>_tiled_pool/`<br>- 切塊座標映射：`./datasets/<data_name>_cached_tile_manifest.json` | `<FAB_RESULT_DIR>/infer_<data_name>_<cat>_<時間戳>/`（含 `anomaly_images/`、`amap_npy/`、`escape_ranking_report.csv/.txt`） |
| **Step 4**<br>`4_split_eval_results.py`<br>`run_4_split_eval_results.sh` | **推論結果多包拆分與各包獨立門檻重算器：** 將 Step 3 產出之報表與圖片依指定張數（`--counts`）或比例（`--ratios`）拆分為 N 包獨立評估集，內建總張數嚴格防呆核對與各包獨立重算動態門檻（`--recompute_pack_thresh`）。 | Step 3 輸出之 `<FAB_RESULT_DIR>`（自動鎖定最新時間戳記推論目錄） | 無（支援 `--symlink` 軟連結零空間負擔分包） | `<推論時間戳目錄>/split_eval_<N>packs/eval_pack_1 ~ N/` 與 `all_<N>_evals_comparison.txt/.csv` |
| **廠內多層總控**<br>`run_all_layers.sh` | **多製程層全自動流水線：** 依序針對多個晶圓 Layer（如 `TMQB28`、`TMTN86`、`TMUN87`）自動串聯執行 `run_1` -> `run_2` -> `run_3` -> `run_4`（支援設定各 Layer 拆分包數與每包張數）。 | `/workspace/dataset/DA_Wafer_use/<DATASET_ID>/<LAYER_ID>/other_status` | 各 Layer 獨立之 `datasets/wafer_<LAYER_ID>_4shot` 與 `logs/` 權重 | `./wafer_escape_results/<DATASET_ID>/<LAYER_ID>/` |
| **廠外多類別總控**<br>`run_outside.sh` | **公開資料集四階段模擬抓漏與跨資料集總表彙整：** 支援 MVTec AD / MVTec AD 2 多類別自動建立獨立模擬盲池（`POOL_MODE="escape"` 或 `"pure_normal"`），跑完 Step 1 ~ Step 4 並匯出跨資料集總表。 | `./datasets/mvtec/<類別>` 或 `./datasets/mvtec-2/<類別>` | `./datasets/sim_fab_<DS>_<類別>_pool/` 與對應 `logs/` 權重 | `./output/2_fab_no_gt_<類別>/` 與 `./output/latest_multi_dataset_summary.txt/.csv` |

---

### 3.2 四階段核心演算法與工程亮點詳解

#### (1) `1_auto_golden.py` — 大資料集抽樣 + Top-16 候選池四合一過濾 + Stage 2 切塊投票與同心圓視覺化
1. **Stage 1 大資料集快速抽樣與四合一純淨度初篩 (`select_golden_images`)：**
   * **大資料集抽樣防護 (`--max_coarse_scan 250`)：** 面對上千張晶圓盲池時，自動以固定亂數種子（`sample_seed=42`）均勻抽樣 250 張進行全域評估，徹底消除全池 I/O 瓶頸。
   * **光度與對比度穩健門檻 (Photometric Gate)：** 計算亮度中位數、標準差與 P5 ~ P95 跨度之 MAD Z-score，自動攔截曝光不足 (`underexposed`) 與過曝 (`overexposed`) 樣本（同時支援 16-bit TIFF P0.1~P99.9 穩健正規化）。
   * **空間網格對齊中心度 (Spatial Grid Alignment)：** 提取 8x8 空間池化特徵與跨片中位數比對，攔截平移或旋轉偏移 (`shift_1` / `shift_2`) 樣本。
   * **強化版跨片同座標局部 Patch 異常峰值與單點 Spike 懲罰：** 強化 16x16 局部座標異常（`patch_anom_z`）與單圖內部 Token Spike（`spike_z`）懲罰權重，嚴格攔截細微裂痕或微粒混入 Golden。
   * **近鄰多樣性去重 (Diversity NMS)：** 防止同一顆工件的多重拍攝分身同時霸佔候選池，鎖定不重複的 **Top-16 候選晶圓池 (`--candidate_pool_size 16`)**。
2. **Stage 2 先切塊後投票選 Golden、二次純化與空間分佈視覺化 (`prepare_tiled_golden_trainset`)：**
   * **智慧切塊觸發門檻 (`--min_tile_trigger 1280`)：** 當原圖尺寸 <= 1280px（如 MVTec 1000x1000）時，自動維持整張原圖訓練以保留完整空間結構對齊；當晶圓大圖 > 1280px 時才啟動高解析度切塊。
   * **跨片同座標切塊 PK 投票：** 將 Stage 1 選出的 16 張候選晶圓於 RAM 內切塊，在每個非背景網格座標 `(y, x)` 結合「零均值標準化結構殘差 + 像素中位數殘差 + 曝光偏離懲罰 + 輕量 DINO 圖塊特徵距離 + Stage 1 先驗正則化」進行同座標對決。
   * **統計勝出票數選出最終 4 張 Golden 與 48 張純淨訓練圖塊：** 統計 16 張候選圖在所有座標獲得前 2 名的勝出票數（`Win_Votes`，第 1 名得 2 票、第 2 名得 1 票），正式選出最終 **Top-4 張最純淨 Golden 原圖** 備份至 `golden_full_res_backup/`，同時跨候選池徑向分層選出 **48 張最純淨訓練圖塊** 輸出至 `train/<category>/`。
   * **自動輸出 48 個訓練圖塊與同心圓環帶視覺化圖 (`golden_tile_layout_vis.png`)：** 左面板繪製所有有效非背景候選網格與 25%、50%、75%、100% 四道同心圓徑向分層環帶；右面板依圓心至外環（`#01` 藍青色 -> `#48` 紅色）以半透明彩虹漸層標示最終選出的 48 個代表性訓練圖塊位置與 64px 重疊區（若觸發防呆沿用既有 Golden 但缺少該圖，亦會自動從備份補畫）。

#### (2) `2_train_fab_no_gt.py` — 智慧防呆快取、離線防 504 保護與無衝突 DDP 訓練
* **雙重防呆略過與結構升級自癒機制：**
  * 只要 `datasets/<data_name>/train/<category>` 已存在挑選好的 Golden 樣本且未指定 `--rebuild_golden`，自動直接沿用，絕不重複挑選；若偵測到舊版誤將 <= 1280px 圖片切成碎片，還會自動從 `golden_full_res_backup` 無損還原完整 Golden 原圖。
  * 嚴格檢查當前 `data_name` 專屬 `./logs/<data_name>/` 下是否已有訓練完成的權重檔；若已存在且訓練集樣本數（`num_train_samples`）未變動，自動跳過訓練階段（若想強制重訓可加 `--force_train`；若訓練集結構升級則自動重新訓練）。
* **動態空閒 Port 分配 (`get_free_port`)：** 每次啟動子程序時自動由作業系統分配保證空閒的 `FAB_MASTER_PORT`，徹底根除多人共用伺服器或多層連跑時的 `Port 40112 Address already in use` 錯誤。
* **內建 `.offline_hub_patch` 自動攔截 `torch.hub.load`：** 自動建立 `./.offline_hub_patch/sitecustomize.py` 並注入 `PYTHONPATH`，當底層呼叫 `torch.hub.load(source='github')` 時自動轉向本地快取目錄載入，徹底杜絕 GitHub HTTP 504 Gateway Time-out。

#### (3) `3_infer_fab_no_gt.py` — 跨批號基準對接、RAM 即時縫合、微瑕疵高靈敏模式與自適應左半核心門檻
1. **跨資料集 / 跨批號 Golden 與權重直接對接 (`--golden_src` & `--trained_ckpt`)：**
   * **靈活指定舊 Golden (`--golden_src` / `--custom_golden_dir`)：** 支援傳入「舊資料集的 `shot_dir`」、「任意含 Golden 原圖的資料夾」、「`selected_4_golden_images.txt` 清單」或「以逗號分隔的多張圖片路徑」。若指定的 Golden 來自外部舊批號（不在當前 `--pool_dir` 內），系統會自動以 `[GOLDEN_REF]` 注入推論圖塊池，同步計算空間結構容差與動態門檻基準，並在輸出最終排名報表時自動排除外部基準圖，不混入新測試集排名。
   * **靈活指定舊訓練權重 (`--trained_ckpt` / `--custom_ckpt`)：** 支援直接指定任意 `.pth.tar` 權重檔案或權重目錄，自動解析訓練步數並對接至當前推論目錄。
   * **智慧無縫回退 (`Golden-AutoRef`)：** 即使未顯式傳入 `--golden_src`，只要沿用舊 `data_name` 推論全新 `--pool_dir`（新池內不含舊 Golden 檔名），系統也會自動從 `golden_full_res_backup/` 載入舊 Golden 原圖作為跨資料集基準。
2. **測試池切塊快取與 RAM 即時縫合銷毀碎片：**
   * 自動驗證盲池 MD5 指紋、外部 Golden 指紋（`ext_golden_sig`）與切塊參數（含 `min_tile_trigger=1280`），若未變動則秒速沿用既有推論圖塊 PNG 快取。
   * 推論完成後，於 RAM 內將單張大圖的所有子圖塊 `.npy` 以 2D 餘弦平滑權重縫合回完整解析度，**並在拼回整張 Wafer 的瞬間立刻刪除該圖所有碎片 `.npy`**，確保硬碟零碎片殘留。
3. **升級版防誤報熱力圖純化器與「微瑕疵高靈敏模式 (`--micro_defect_mode`)」：**
   * **模式 A：`MICRO_DEFECT_SENSITIVE`（預設開啟 `--micro_defect_mode`）：**
     * **形態學白頂帽（White Top-Hat）局部尖峰增強：** 自動依據影像尺寸計算背景開運算核，提取高於局部背景的微小突波並加權增強（`--micro_boost_weight 0.35`），同時將 Golden 結構容差權重溫和調降為 `0.08`，避免吃掉微弱缺陷。
     * **防稀釋局部尖峰計分 (`--micro_peak_pixels 24`)：** 全圖異常分數改取前 24 個最亮像素（或前 0.002% 像素）之平均值，徹底防止微米級針孔（Pinhole）、細小裂痕（Crack）或微刮痕在大圖中被周圍背景稀釋。
     * **5x5 閉運算微框保護：** 針對二值化遮罩改用 5x5 橢圓閉運算連接斷裂微結構（不做開運算削弱），並將最小面積門檻放寬至 `min(min_area_ratio, 0.000004)`（最低 4px），精準鎖定微米級瑕疵。
   * **模式 B：`FILTERED_ROBUST`（一般四合一濾波模式，將 `MICRO_DEFECT_MODE=""` 時啟用）：**
     * 採用 `golden_tol_weight=0.15` 結構複雜區容差補償 + 3x3 輕量橢圓核開運算微塵抑制 + Top-0.015% 抗噪峰值計分。
   * **模式 C：`RAW_UNFILTERED`（`--raw_score_mode` 消融對照模式）：** 完全不經濾波，直接取單點最大值 `np.max`。
   * **矩形邊框抑制 (`--border_ratio 0.02`)** 與 **智慧圓形晶圓倒角抑制 (`--wafer_edge_mode [auto|on|off]`, `--wafer_edge_margin 0.04`)**：內建長寬比（>= 0.85）與圓形度（>= 0.82）雙重幾何檢驗，精準消除圓形晶圓外圈 Bevel 反光，且絕不會誤傷長條形工件（如 Capsule）頭尾。
4. **自適應未見過正常核心估計器 + Golden 基準雙重校準動態門檻 (`estimate_blind_normal_core_stats`)：**
   * 完全不依賴任何檔名標籤，直接從**已排除 Golden 訓練圖之非 Golden 盲池左半部（前 55% 低分群之 P75）**穩健估計未見過正常樣本中心 `mu_core` 與左半單邊標準差 `sigma_left`（結合 RMS、MAD 與下半部 IQR）。
   * **雙向自適應保護：** 既能解決「盲池內異常比例偏高時，傳統 `mean + k*std` 被右側異常高分拉爆導致漏抓」的問題，又能避免「4-shot Golden 整張原圖因記憶效應分數極低或跨批號光照微偏移時，將 `mu_core` 錯誤向下鎖死在 `g_max` 導致正常晶圓大量誤報」。
5. **無監督模型過擬合 / 跨資料集分佈偏移自檢診斷 (`evaluate_overfitting_diagnostics`)：**
   * 自動比對「Golden 基準圖平均分數」與「盲池正常核心中位數」之比值（`Overfit_Ratio`）與差距（`Gap`），於終端機與報表自動標示 `🟢 良好: 無過擬合現象`、`🟢 穩定: 輕微批次偏移傾向` 或 `🟡 注意: 偵測到 Golden 記憶效應或跨批號微偏移 (已由 Unseen-Normal 左半核心自動校準補償)`。

#### (4) `4_split_eval_results.py` — 指定包數/每包張數、總數核對防呆與各包獨立重算動態門檻
* **靈活拆分模式（`--counts` vs `--ratios`）與嚴格數量防呆核對：**
  * 支援透過 `--num_packs` 指定拆分成幾包（預設 3 包），並可透過 `--counts` 指定每包固定張數（如 `"500"`）或各包明細（如 `"700,700,600"`）。
  * **嚴格數量核對防呆（Count Mismatch Error）：** 若設定之各包張數加總不等於資料集內可拆分圖片總數，程式會立即中止並清楚列出「資料集實際總張數」、「各包設定明細」與「多出/少算張數」。
* **各包獨立重算動態門檻（`--recompute_pack_thresh`，預設開啟）：**
  * 每一包僅根據**該包內部的非 Golden 樣本分數**與 Golden 基準錨點，獨立套用與 Step 3 100% 相同的數學公式重算該包專屬門檻 `Pack_Local_Threshold`（當單包樣本數 `< 25` 張時，自動啟動小樣本平滑融合機制與全池門檻加權融合），徹底消除「先推論再拆分是否偷看全池門檻」的疑慮。
  * 同時輸出各包獨立報表（`eval_pack_1` ~ `eval_pack_N`）與跨包並列比較總覽表 `all_<N>_evals_comparison.txt/.csv`。

---

## 4. 執行訓練與測試操作指南（公司內實測 vs 跨批號推論 vs 公司外測試）

執行腳本前，請先確認已啟動環境並位於專案根目錄 `FoundAD/`：
```bash
conda activate foundad
cd /path/to/FoundAD
```

### 4.1 公司內實測：分步執行或一鍵批次跑完多製程層 (Multi-Layer)

#### 方式 A：使用 `run_all_layers.sh` 一鍵自動跑完所有製程層（Step 1 -> Step 4 全自動）
打開 `run_all_layers.sh` 確認要執行的 `DATASET_ID`、`LAYER_ID`、盲池路徑，以及每個 Layer 要拆成幾包（`NUM_PACKS`）與每包張數（`PACK_COUNTS`，留空 `""` 則自動均分）後，直接執行：
```bash
# 前景執行
bash run_all_layers.sh

# 或使用 nohup 掛在背景執行 (推薦，防止 SSH 斷線)
nohup bash run_all_layers.sh > train_all_layers.log 2>&1 &
```

#### 方式 B：針對單一製程層（Layer）分階段獨立執行與調參
四支 `.sh` 腳本皆支援直接執行（使用腳本內預設值）或透過命令列傳入參數：
```bash
# Step 1: 自動從盲池初篩 16 張候選圖、RAM 切塊投票選出 4 張 Golden 與 48 張訓練圖塊，並輸出同心圓分佈圖
bash run_1_auto_golden.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 2: 載入 Golden 訓練集並執行 FoundAD 2000 步訓練 (若已訓練過會自動防呆跳過)
bash run_2_train_fab_no_gt.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 3: 執行測試池切塊推論、RAM 即時縫合銷毀碎片、微瑕疵高靈敏防誤報濾波與紅框輸出
bash run_3_infer_fab_no_gt.sh W639 TMQB28 /workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status

# Step 4: 將最新一次推論結果依分層抽樣 (stratified) 自動拆分為多包獨立 Eval 報表與圖片包 (預設各包獨立重算門檻)
# 用法 1 (預設均分拆為 3 包):
bash run_4_split_eval_results.sh W639 TMQB28
# 用法 2 (指定拆成 3 包，且每包張數分別為 700, 700, 600 張 — 加總須等於總張數):
bash run_4_split_eval_results.sh W639 TMQB28 "" 3 "700,700,600"
# 用法 3 (指定拆成 4 包，每包固定 500 張):
bash run_4_split_eval_results.sh W639 TMQB28 "" 4 "500"
```

### 4.2 跨資料集 / 跨批號推論：直接用舊批號的 Golden 與訓練權重推論新批號晶圓池
當同一製程層（或相近製程）有新一批晶圓圖片池需要檢測，且你想**直接沿用舊資料集挑過的 Golden Image 與已訓練好的模型權重（0 訓練成本、不重挑 Golden）**時，`run_3_infer_fab_no_gt.sh` 與 `3_infer_fab_no_gt.py` 支援透過命令列第 4、第 5 個參數（或腳本內變數 `CUSTOM_GOLDEN_SRC`、`CUSTOM_TRAINED_CKPT`）直接指定：

```bash
# 語法: bash run_3_infer_fab_no_gt.sh <新DATASET_ID> <新LAYER_ID> <新晶圓池路徑> <舊Golden來源> <舊訓練權重路徑>
bash run_3_infer_fab_no_gt.sh \
    W640 \
    TMQB28 \
    /workspace/dataset/DA_Wafer_use/W640/TMQB28/other_status \
    ./datasets/wafer_TMQB28_4shot/golden_full_res_backup/wafer \
    ./logs/wafer_TMQB28_4shot/dinov3fab_TMQB28/train-step2000.pth.tar
```
*支援的 `--golden_src`（或 `--custom_golden_dir`）四種輸入格式：*
1. 舊資料集的 `shot_dir` 根目錄（如 `./datasets/wafer_TMQB28_4shot`）
2. 含有 Golden 原圖的任意資料夾路徑（如 `./datasets/wafer_TMQB28_4shot/golden_full_res_backup/wafer`）
3. `selected_4_golden_images.txt` 紀錄檔或純文字路徑清單 (`.txt`)
4. 以逗號 (`,`) 分隔的多張圖片路徑字串

### 4.3 已訓練完成後，如何「零訓練成本」秒速重調門檻、切換微瑕疵模式或防誤報參數？
由於訓練（`2_train_fab_no_gt.py`）與推論（`3_infer_fab_no_gt.py`）已完全解耦，當你想嘗試不同的門檻模式（如 `hybrid`、`mean_std`、`fixed`）、切換微瑕疵高靈敏模式（`--micro_defect_mode`）、調整標準差倍數 `SIGMA_K`、晶圓邊緣抑制寬度 `WAFER_EDGE_MARGIN`，或切換 `--raw_score_mode` 進行消融實驗時，**完全不需要動到 Step 1 與 Step 2**！

只需打開 `run_3_infer_fab_no_gt.sh` 修改第 2、3 區參數：
```bash
THRESH_MODE="hybrid"                       # 支援: "hybrid"、"mean_std"、"percentile"、"top_k"、"fixed"
SIGMA_K=3.5                                # 調整標準差倍數 k (調小抓更細微瑕疵，調大更嚴格防誤報)
MICRO_DEFECT_MODE="--micro_defect_mode"    # 設為 "--micro_defect_mode" 啟動微瑕疵高靈敏模式；設為 "" 則為一般四合一濾波模式
RAW_SCORE_MODE=""                          # 若想對照未濾波原生分數可設為 "--raw_score_mode"
WAFER_EDGE_MODE="on"                       # 圓形晶圓設 "on" (或 "auto")
WAFER_EDGE_MARGIN=0.04                     # 晶圓最外圈 4% 半徑平滑衰減
```
接著直接重跑 Step 3（與 Step 4）即可（測試池切塊 PNG 會自動命中快取，0 秒完成切塊準備）：
```bash
bash run_3_infer_fab_no_gt.sh W639 TMQB28
bash run_4_split_eval_results.sh W639 TMQB28
```

### 4.4 公司外公開資料集測試（`run_outside.sh`）
在外部環境可一鍵測試預設的 7 個公開資料集類別（或直接由命令列傳入指定類別路徑），自動串聯 Step 1 ~ Step 4，並在 `./output/` 產出跨資料集彙整總表 `latest_multi_dataset_summary.txt/.csv`（若想測試 100% 純正常池誤報率，只需將 `run_outside.sh` 內的 `POOL_MODE` 改為 `"pure_normal"`）：
```bash
# 執行預設類別清單 (mvtec/capsule, mvtec-2/can, bottle, cable, grid, tile, transistor)
bash run_outside.sh

# 或指定單一/多個類別資料夾快速測試
bash run_outside.sh ./datasets/mvtec/capsule ./datasets/mvtec-2/can
```

#### 常用進度監控指令：
```bash
# 廠內即時查看最新訓練或推論進度
tail -f ./wafer_escape_results/W639/TMQB28/latest_train_log.txt
tail -f ./wafer_escape_results/W639/TMQB28/latest_log.txt

# 廠外即時查看最新進度與跨資料集彙整總表
tail -f ./output/2_fab_no_gt_capsule/latest_log.txt
cat ./output/latest_multi_dataset_summary.txt
```

---

## 5. 確認訓練完的權重檔與 Golden 資料集位置

* **Golden 訓練集、候選池投票紀錄與切塊分佈視覺化圖（`./datasets/<DATA_NAME>/`）：**
  * 廠內路徑：`./datasets/wafer_<LAYER_ID>_4shot/`
    * `train/wafer/`：最終用於訓練的 48 張代表性純淨子圖塊（或完整原圖）。
    * `golden_full_res_backup/wafer/`：切塊投票勝出的 4 張全解析度 Golden 原圖備份。
    * `golden_tile_layout_vis.png`：48 個代表性訓練圖塊與同心圓徑向環帶空間分佈視覺化圖。
    * `stage1_candidates_wafer.json`：Stage 1 初篩 Top-16 候選晶圓詳細指標。
    * `golden_tile_purity_report.txt`：Stage 2 候選池切塊投票排行榜與各訓練圖塊座標清單。
    * `train_meta_wafer.json`：記錄訓練完成之步數、耗時與權重檔絕對路徑。
* **模型權重檔位置（`./logs/<DATA_NAME>/`）：**
  * 廠內路徑：`./logs/wafer_<LAYER_ID>_4shot/dinov3fab_<LAYER_ID>/train-step2000.pth.tar`（及自動對接之 `pretrained.pth.tar`）
  * 廠外路徑：`./logs/<DS>_sim_<類別>_4shot/dinov3sim_escape_<類別>/train-step*.pth.tar`

可使用以下指令快速確認所有已生成的權重檔：
```bash
ls -lh ./logs/*/*/*.pth.tar
```

---

## 6. 檢驗測試結果與輸出報表說明 (`./wafer_escape_results/` 與 `./output/`)

每次執行 `3_infer_fab_no_gt.py` 與 `4_split_eval_results.py` 後，會在 `<FAB_RESULT_DIR>/infer_<DATA_NAME>_<CAT>_<時間戳>/` 下生成完整產出：

| 檔案 / 資料夾名稱 | 內容說明 |
| :--- | :--- |
| `anomaly_images/` | **【核心產出 1】異常紅框三合一對照圖（僅輸出超過有效門檻之異常圖）：**<br>檔名格式為 `Rank001_Score0.3450_Box01_<原檔名>.png`（依異常分數由高至低排序，第一張即為最可疑的 Escape 晶圓）。<br>頂部橫幅顯示 `[MICRO/FILT/RAW]`、原檔名、最終分數、未濾波原始分、濾波後分數、有效門檻與紅框數；下方為 `[左：原始圖 -> 中：純化後異常熱力疊圖 -> 右：紅色 Bounding Box 瑕疵定位框]`。 |
| `anomaly_images/full_res_bbox_only/` | **【核心產出 2】1:1 原始全解析度紅框定位圖：**<br>維持原圖 100% 原始解析度（例如 5000x5000 不縮放），僅在原圖畫上紅色瑕疵框與局部分數標籤，供工程師放大檢視微米級缺陷細節。 |
| `escape_ranking_report.txt` | **無 GT 異常抓漏總排名、消融對照與過擬合診斷報表（純文字版）：**<br>1. 標頭完整記錄計分模式（`MICRO_DEFECT_SENSITIVE` / `FILTERED_ROBUST` / `RAW_UNFILTERED`）、有效門檻、盲池正常核心分數分佈、**濾波前後正常圖對照 (Ablation)**、`mean + 1*std / 2*std / 3*std` 門檻參考值、**Overfit Diagnostic 過擬合/分佈偏移自檢結果**，以及跨資料集外部 Golden 基準分數（`External Golden Ref Scores`）。<br>2. 表格依序列出所有圖片之 `Rank`、`Original_Filename`、`Status`（`ANOMALY` / `GOLDEN_PASS` / `PASS_SKIPPED` / `PASS_NOISE_FILTERED`）、`Max_Score`、`Raw_Score`、`Filt_Score`、`Delta`、`BBoxes` 數量與對應輸出檔名。<br>3. 末尾自動附加訓練與推論各階段詳細耗時總結。 |
| `escape_ranking_report.csv` | 同上之完整 CSV 報表（含 `Max_Score`、`Raw_Max_Score`、`Filtered_Max_Score`、`Filter_Delta`、`TopK_Mean_Score`、`Num_BBoxes`、`Resolution`），可直接用 Excel 開啟篩選。 |
| `golden_reports/golden_tile_layout_vis.png` | **【Step 1 產出】48 個代表性訓練圖塊與同心圓環帶視覺化對照圖：**<br>左圖顯示所有非背景有效切塊網格與 4 道同心圓徑向分層環帶；右圖以彩虹漸層色塊與編號 `#01`~`#48` 標示最終選出的訓練圖塊空間分佈，方便直接放入報告與簡報。 |
| `golden_tile_purity_report.txt` | **Stage 2 候選池切塊投票排行榜與圖塊純化報表：**<br>完整列出 16 張候選晶圓的切塊得票數（`Win_Votes`）、平均圖塊殘差成本，以及最終選出的 4 張 Golden 與 48 張訓練圖塊來源。 |
| `selected_4_golden_images.txt` / `selected_custom_golden_images.txt` | 最終勝出之 4 張 Golden Image 原檔名與評分紀錄；若使用 `--golden_src` 指定自訂 Golden，則會輸出 `selected_custom_golden_images.txt` 記錄對應的 `Manifest_Key`。 |
| `amap_npy/` | **異常樣本之全解析度熱力圖矩陣（`.npy`）：**<br>僅儲存超過門檻之異常圖片縫合且濾波後的整張 Wafer 2D 矩陣（子圖塊碎片已於 RAM 拼接後立即自動銷毀）。 |
| `split_eval_<N>packs/` | **【Step 4 產出】獨立拆分評估結果包（`eval_pack_1` ~ `eval_pack_N`）與跨包比較總表：**<br>每包皆含獨立重排與獨立門檻對照的 `escape_ranking_report.csv/.txt`（包含 `Pack_Rank`、`Global_Rank`、`Effective_Threshold_Used`、`Pack_Local_Threshold`、`Global_Pool_Threshold`）及對應的 `anomaly_images/`、`amap_npy/`；根目錄並附有 `all_<N>_evals_comparison.txt/.csv`（與 `all_3_evals_comparison.txt/.csv`）並列總覽表。 |
| `log.txt` | 本次推論執行的完整終端機日誌與時間統計。 |

---

## 7. 常見問題與預先排錯指南（Troubleshooting）

### Q1：在公司內執行時報錯 `URLError` / `ConnectionError` / `HTTP 504`（嘗試連網下載 GitHub 或 HuggingFace）
* **原因：** 公司內網阻擋連向 GitHub (`torch.hub.load`) 或 HuggingFace。
* **解法：**
  1. 本系統已在 `2_train_fab_no_gt.py` 與 `3_infer_fab_no_gt.py` 內建 `.offline_hub_patch/sitecustomize.py` 自動攔截器，只要本地快取有 DINO repo 即可自動離線載入。
  2. 請確認已照 **第 1.3 節 Step 1** 將外部機器 `~/.cache/torch/hub/*dino*` 複製到專案下的 `./hub_cache/`（或 `~/.cache/torch/hub/`）。
  3. 確認 `run_1_auto_golden.sh`、`run_2_train_fab_no_gt.sh`、`run_3_infer_fab_no_gt.sh` 內的 `COMPANY_DINO_CKPT` 已正確指向公司伺服器上的 `.pth` 權重檔路徑，且 `FAB_LOCAL_DINO_REPO` 指向 `./hub_cache/facebookresearch_dinov3_main`。

### Q2：如何根據報表中的「盲池正常核心分數分佈」與「過擬合自檢」微調門檻？
* **解法：** 每次跑完 `run_3_infer_fab_no_gt.sh` 或 `run_outside.sh`，終端機與報表都會印出診斷面板：
  ```text
  📊 盲池正常核心分數分佈 (MICRO_DEFECT_SENSITIVE) : Max=0.2410 | P99=0.2350 | P95=0.2180 | Median=0.1850 | Mean=0.1880 | Std=0.0160
  📐 盲池正常核心門檻參考值      : mean+1*std=0.2040 | mean+2*std=0.2200 | mean+3*std=0.2360
  🔬 濾波前後正常圖對照 (Ablation): [未濾波 Raw] Max=0.3120, Mean=0.2250  vs  [當前模式] Max=0.2410, Mean=0.1880
  🩺 模型過擬合自檢 (Overfit)    : 🟢 良好: 無過擬合現象 (Healthy Generalization — Golden基準與未見過正常分佈一致)
  ```
  * **若出現 `🟡 注意: 偵測到 Golden 記憶效應或跨批號微偏移`：** 系統的「自適應未見過正常左半核心估計器（Unseen-Normal Left-Core Estimator）」已經自動將門檻錨點切換至非 Golden 正常核心 `mu_core` 進行補償；若正常晶圓誤報仍偏多，可將 `run_3_infer_fab_no_gt.sh` 中的 `SIGMA_K` 調高（例如由 `3.5` 調至 `4.5` 或 `5.0`）或關閉 `MICRO_DEFECT_MODE=""` 直接重跑 Step 3。
  * **若想抓出更微弱的細小針孔或微裂痕：** 確認 `run_3_infer_fab_no_gt.sh` 中已啟用 `MICRO_DEFECT_MODE="--micro_defect_mode"`，並可將 `SIGMA_K` 調小為 `2.5` 或 `3.0`（或調高 `--micro_boost_weight` 至 `0.45`）重跑 Step 3。
  * **若晶圓最外圈圓弧邊緣反光被誤框：** 確認 `WAFER_EDGE_MODE="on"`，並將 `WAFER_EDGE_MARGIN=0.04` 調大為 `0.06`（代表外圈 6% 半徑平滑抑制）。

### Q3：執行 Step 4 分包時報錯 `❌ [拆分圖片總數量不符錯誤 (Count Mismatch Error)]` 怎麼辦？
* **原因：** 當你在 `run_4_split_eval_results.sh`（或 `--counts`）指定了每包具體張數（例如 `700,700,600` 加總為 2000 張），但實際推論報表中的可拆分圖片總數不等於 2000 張時，防呆機制會主動攔截。
* **解法：**
  1. 根據終端機報錯訊息中顯示的「原選擇資料集內的可拆分圖片總數量」，修正 `COUNTS` 使其加總完全吻合。
  2. 或者將 `COUNTS=""` 留空，改用預設比例均分（`RATIOS="1:1:1"`），系統會自動將任意數量的圖片精準均分為 `NUM_PACKS` 包並分配餘數。

### Q4：想強制重新挑選 Golden Image 或強制重新訓練模型該怎麼做？
* **解法：** 為避免浪費 GPU 時間，系統預設開啟防呆快取（有 Golden 不重挑、有權重不重訓）。
  * **強制重挑 Golden：** 在執行 `1_auto_golden.py` 或 `2_train_fab_no_gt.py` 時加上 `--rebuild_golden`（或在 `run_2_train_fab_no_gt.sh` 中設定 `REBUILD_GOLDEN="--rebuild_golden"`）。
  * **強制重新訓練模型：** 在執行 `2_train_fab_no_gt.py` 時加上 `--force_train`。
  * **強制重新切塊測試池：** 在執行 `3_infer_fab_no_gt.py` 時加上 `--force_infer`（或在 `run_3_infer_fab_no_gt.sh` 中設定 `FORCE_INFER="--force_infer"`）。