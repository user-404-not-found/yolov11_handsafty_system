# 手部安全監控系統 (YOLOv11 + ONNX + PLC)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-GPU_Accelerated-green.svg)](https://onnxruntime.ai/)
[![Framework](https://img.shields.io/badge/Framework-YOLOv11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本專案主要利用相機鏡頭加上 AI 辨識，當發現作業員的手或手套不小心伸進「危險區域」時，系統發送訊號給 PLC 控制器，把機台強制停機，保護人員安全

---

##  簡介

一般的機械防護網很笨重，而傳統的安全光幕（紅外線）又很容易因為機台震動、反光或安裝角度而有死角。
本系統用 AI 來改善這些問題：
1. 支援兩台相機同時看，正面、側面都不漏抓。
2. 可以自訂危險區域，比傳統的死板矩形更靈活。
3. 手伸進去的一瞬間只會通知 PLC 一次，不會因為 AI 一秒跑好幾十張圖就對 PLC 持續發送停機訊息導致連線卡死。

---

##  系統需求 
本系統開發與測試基於高效能邊緣運算平台，確保在雙路 30+ FPS 下能維持極低的運算遲滯與高幀率。

* **中央處理器 (CPU)**: Intel Core i9-14900HX (24 核心 / 32 執行緒) 
* **圖形處理器 (GPU)**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB GDDR6 VRAM) 
* **系統記憶體 (RAM)**: 32GB DDR5 5600MHz。
* **工業控制端 (PLC)**: 支援 Modbus TCP 協議之控制器（如：三菱 FX/Q 系列、台達 DVP、西門子 S7 系列等）。

---

##  環境配置與依賴安裝 

本專案推薦使用 Anaconda 進行環境隔離。請在終端機 (Terminal) 中依序執行以下指令建立環境：

### 1. 建立 Conda 環境
```bash
conda create -n yolov11_safety python=3.11 -y
conda activate yolov11_safety
```

### 2. 安裝核心依賴庫
```bash
# 安裝影像處理與數值計算庫
pip install opencv-python numpy

# 安裝工業 Modbus TCP 通訊庫
pip install pyModbusTCP
```

### 3. GPU 推理引擎配置 
為使 ONNX 引擎順利調用 RTX 4070 的 Tensor Core，必須精確匹配 CUDA 與 cuDNN 版本：
```bash
# 安裝對應 CUDA 12.x 的 ONNX Runtime GPU 版本
pip install onnxruntime-gpu
```
>  **提示**：需預先安裝 NVIDIA 顯示卡驅動、**CUDA Toolkit 12.x** 以及 **cuDNN 9.x**，並將相關 DLL 路径加入系統環境變數中。若無 GPU 環境，可選擇安裝純 CPU 版本：`pip install onnxruntime`。



##  模型介紹 

系統核心使用經由工業手部、防護手套數據集（Hand & Gloves Pro dataset）訓練優化的 **YOLOv11** 模型。

### 1. 模型效能評估 
本模型各項指標如下：

* **最高 F1 分數**: `0.88`
* **最佳信心門檻**: `0.351`


##  檔案結構

```text
yolo_hand/
├── model/                          # 核心執行程式與主模型資料夾
│   ├── hand_detect_just_onnx.py     # 乾淨的 ONNX 測試腳本（單台相機快速測試用）
│   ├── hand_detect_onnx_PLC.py      # 最終完成版！雙路攝影機 + 畫梯形 ROI + 控制 PLC 停機
│   ├── hand_detect_v4.py            # .pt 模型的測試檔案（需要安裝 ultralytics 環境）
│   ├── v4.onnx                      # 翻譯成通用格式的模型（跑部署、衝辨識速度用這個）
│   └── v4.pt                        # PyTorch 原始模型權重
│
└── YOLO_gloves_pro_v4/              # AI 模型訓練的原始紀錄與數據資料夾
    ├── weights/                     # 訓練過程中自動儲存的權重檔案庫
    │   ├── best.onnx                # 訓練出來「表現最好」的 ONNX 轉檔模型
    │   ├── best.pt                  # 訓練出來「表現最好」的 PyTorch 原始權重
    │   └── last.pt                  # 訓練到「最後一輪」的 PyTorch 原始權重
    ├── args.yaml                    # 當時訓練模型時，系統記錄下來的參數設定檔
    ├── BoxF1_curve.png              # F1 分數平衡曲線（評估模型穩不穩定的核心指標）
    ├── BoxP_curve.png               # 精準率（Precision）變化曲線圖
    ├── BoxPR_curve.png              # 精準率與召回率（PR）精確度對應圖
    ├── BoxR_curve.png               # 召回率（Recall）變化曲線圖
    ├── confusion_matrix.png         # 混淆矩陣圖（看 AI 認錯哪些東西）
    ├── confusion_matrix_normalized.png # 歸一化混淆矩陣（對角線代表預測正確的比例）
    ├── labels.jpg                   # 資料集標籤與框線的分佈狀況圖
    ├── results.csv                  # 訓練過程所有數值的數據報表（可用 Excel 打開）
    ├── results.png                  # 訓練 Loss 暴跌與 mAP 戰力飆升的視覺化曲線圖
    ├── train_batch0.jpg             # 訓練集圖片範例與標籤（批次 0）
    ├── train_batch1.jpg             # 訓練集圖片範例與標籤（批次 1）
    ├── train_batch2.jpg             # 訓練集圖片範例與標籤（批次 2）
    ├── train_batch143680.jpg        # 訓練後期（看模型精修狀況）的圖片範例 A
    ├── train_batch143681.jpg        # 訓練後期（看模型精修狀況）的圖片範例 B
    ├── train_batch143682.jpg        # 訓練後期（看模型精修狀況）的圖片範例 C
    ├── val_batch0_labels.jpg        # 驗證集 - 人類標註的正確答案（批次 0）
    ├── val_batch0_pred.jpg          # 驗證集 - AI 實際上預測出來的結果（批次 0）
    ├── val_batch1_labels.jpg        # 驗證集 - 人類標註的正確答案（批次 1）
    ├── val_batch1_pred.jpg          # 驗證集 - AI 實際上預測出來的結果（批次 1）
    ├── val_batch2_labels.jpg        # 驗證集 - 人類標註的正確答案（批次 2）
    └── val_batch2_pred.jpg          # 驗證集 - AI 實際上預測出來的結果（批次 2）
```

---

##  核心程式碼邏輯 

### 1. 直梯形 ROI 多邊形判定
利用 `cv2.pointPolygonTest` 方法，將手部偵測框的**底邊中點** $P(X_p, Y_p)$ 帶入梯形座標數組中進行幾何內點判定：

```python
# 垂直範圍：上面 1/3 (h/3) 到 下面 1/2 (h/2)
y_top = int(h / 3)
y_bottom = int(h / 2)

# 左攝影機 (Cam 0) 直梯形定義
roi_points_left = np.array([
    [0, y_top],          # 左上
    [int(w/4), y_top],   # 右上 (向內收縮)
    [int(w/2), y_bottom],# 右下 (延伸到中線)
    [0, y_bottom]        # 左下
], np.int32)

# 測試點：手部邊界框底邊中點
test_point = (int((x1 + x2) / 2), y2)

# 判斷點是否在多邊形內部 (dist >= 0 代表在梯形內或邊界上)
dist = cv2.pointPolygonTest(roi_points_left, test_point, False)
if dist >= 0:
    is_intrusion = True # 觸發安全連鎖
```

### 2. 邊緣觸發 PLC 連鎖
採用狀態鎖定機制，避免對 PLC 線圈進行高頻重複寫入，延長通訊模組壽命：

```python
if is_intrusion:
    # [邊緣觸發] 如果先前未報警，則發送單次停機訊號
    if not already_triggered:
        plc.write_single_coil(STOP_COIL_ADDR, True)
        print("[PLC 訊號] 偵測到手部侵入！已發送緊急停機指令")
        already_triggered = True # 狀態鎖定
else:
    # [復歸機制] 區域安全且先前處於報警狀態，發送復歸訊號
    if already_triggered:
        plc.write_single_coil(STOP_COIL_ADDR, False)
        print("[PLC 訊號] 危險解除，控制系統復歸。")
        already_triggered = False # 狀態解鎖
```

---

##  開源狀態 

本專案目前處於**主動維護與實地部署測試階段**。
歡迎大家提出 Issue 與 Pull Request 共同完善。

---

##  MIT 協議聲明 

本專案採用 **MIT 授權條款**。您可以自由複製、修改、散布或商用，唯須在衍生作品中標註原作者姓名。

```text
Copyright (c) 2026 Willy Tu (凃瑋禮)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```