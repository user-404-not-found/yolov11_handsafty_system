# ==========================================
# Project: 手部安全監控系統 (Hand Safety system)
# Author:  凃瑋禮
# Date:    2026-02-18
# Version: v4.0 (ONNX GPU Accelerated)
# Description: 針對加工機床開發的實時手部偵測與 ROI 入侵報警系統
# ==========================================
import cv2
import numpy as np
import onnxruntime as ort
import time
from pyModbusTCP.client import ModbusClient  # <--- [新增] 匯入通訊套件

# ==========================================
# 啟動 PLC 連線設定
# ==========================================
PLC_IP = "192.168.1.50"  # 請換成 PLC 的真實 IP
PLC_PORT = 502           # Modbus TCP 預設都是 502
STOP_COIL_ADDR = 0       # 觸發停機的記憶體位置 (要問寫 PLC 的人)

# 建立連線物件 (auto_open=True 代表斷線會自動重連，很方便)
plc = ModbusClient(host=PLC_IP, port=PLC_PORT, auto_open=True)
print(f"🔌 嘗試連線至 PLC ({PLC_IP})...")

# ==========================================
# 1. 啟動 ONNX 引擎
# ==========================================
session = ort.InferenceSession(
    "D:/yolov11/ultralytics-8.3.39/v4.onnx", 
    providers=['CUDAExecutionProvider']
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # ... (你的 letterbox 函數保持不變) ...
    pass # 這裡省略，照你原本的寫

cap = cv2.VideoCapture(0)
prev_time = 0

# --- [新增] 用來防止「機關槍狂掃」的狀態變數 ---
already_triggered = False 

print("AI安全辨識啟動中.........")
print("PLC連線成功")

while True:
    ret, frame = cap.read()
    if not ret: break
    h_orig, w_orig = frame.shape[:2]

    img_for_detection = frame.copy()

    # 2. 預處理
    input_img, ratio, (dw, dh) = letterbox(img_for_detection)
    blob = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, :]

    # 3. 推論
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    predictions = np.squeeze(outputs[0]).T
    
    # 4. 畫危險區
    roi_limit_y = int(h_orig * 2 / 3) 
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, roi_limit_y), (w_orig, h_orig), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame) 
    cv2.line(frame, (0, roi_limit_y), (w_orig, roi_limit_y), (0, 0, 255), 3)

    # 5. 解析結果
    mask = predictions[:, 4] > 0.35
    valid_preds = predictions[mask]

    is_intrusion = False 

    for pred in valid_preds:
        cx, cy, w, h, conf = pred
        x1 = int((cx - w/2 - dw) / ratio)
        y1 = int((cy - h/2 - dh) / ratio)
        x2 = int((cx + w/2 - dw) / ratio)
        y2 = int((cy + h/2 - dh) / ratio)

        if y2 >= roi_limit_y:
            is_intrusion = True
            color = (0, 0, 255) 
            thickness = 3
        else:
            color = (0, 255, 0) 
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ==========================================
    # 核心邏輯：控制 PLC 停機與復歸
    # ==========================================
    if is_intrusion:
        # 畫面閃紅框與警告
        cv2.rectangle(frame, (0,0), (w_orig, h_orig), (0, 0, 255), 10)
        cv2.putText(frame, "!!! STOP MACHINE !!!", (int(w_orig*0.15), int(h_orig*0.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # [通訊] 如果之前沒報警，現在報警
        if not already_triggered:
            # 對 PLC 寫入 True (通常代表觸發某個繼電器或停機接點)
            plc.write_single_coil(STOP_COIL_ADDR, True)
            print("[PLC 訊號] 已發送停機指令")
            already_triggered = True # 鎖住，不再發

    else:
        # [通訊] 手離開了，解除鎖定
        if already_triggered:
            # 視機台設定，有些機台需要傳 False 給它解除，有些需要人工去按復原鈕
            plc.write_single_coil(STOP_COIL_ADDR, False)
            print("[PLC 訊號] 危險解除，訊號復歸。")
            already_triggered = False

    # 6. FPS 顯示
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"GPU FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Hand Safety Guard - PLC Edition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()