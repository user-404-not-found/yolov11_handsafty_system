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

# 1. 啟動 ONNX 引擎
session = ort.InferenceSession(
    "D:/yolov11/ultralytics-8.3.39/v4.onnx",  #不是作者電腦文件路徑要改
    providers=['CUDAExecutionProvider']
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """等比例縮放影像，不變形，空位補灰色邊"""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

cap = cv2.VideoCapture(0)
prev_time = 0

print("AI安全辨識啟動中......")

while True:
    ret, frame = cap.read()
    if not ret: break
    h_orig, w_orig = frame.shape[:2]

    # --- 複製一份「乾淨」的影像給 AI 看 ---
    img_for_detection = frame.copy()

    # 2. 預處理 (使用乾淨的影像)
    input_img, ratio, (dw, dh) = letterbox(img_for_detection)
    blob = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, :]

    # 3. 推論 (AI 會看到完整的手)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    predictions = np.squeeze(outputs[0]).T
    
    # 4. 準備顯示畫面：在 frame 上畫出 ROI 警示區域
    roi_limit_y = int(h_orig * 2 / 3) 
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, roi_limit_y), (w_orig, h_orig), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame) 
    cv2.line(frame, (0, roi_limit_y), (w_orig, roi_limit_y), (0, 0, 255), 3)
    cv2.putText(frame, "DANGER ZONE (DO NOT TOUCH)", (10, roi_limit_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 5. 解析結果並畫在 frame 上
    mask = predictions[:, 4] > 0.35
    valid_preds = predictions[mask]

    is_intrusion = False 

    for pred in valid_preds:
        cx, cy, w, h, conf = pred
        x1 = int((cx - w/2 - dw) / ratio)
        y1 = int((cy - h/2 - dh) / ratio)
        x2 = int((cx + w/2 - dw) / ratio)
        y2 = int((cy + h/2 - dh) / ratio)

        # 核心邏輯：偵測框底邊 (y2) 觸碰界線
        if y2 >= roi_limit_y:
            is_intrusion = True
            color = (0, 0, 255) # 警告變紅色
            thickness = 3
        else:
            color = (0, 255, 0) # 安全為綠色
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"Hand {conf:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 畫面中央的紅色大警告
    if is_intrusion:
        # 1. 畫出全畫面閃爍紅框
        cv2.rectangle(frame, (0,0), (w_orig, h_orig), (0, 0, 255), 10) 
        
        # 2. 設定文字內容與樣式
        text = "!!! STOP MACHINE !!!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5  # <--- 從 2.5 調小到 1.5
        thickness = 4     # <--- 線條粗細也隨之調小 (原本是 8)
        
        # 3. 自動計算文字大小，好讓它完美置中
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w_orig - text_size[0]) // 2  # 水平置中
        text_y = (h_orig + text_size[1]) // 2  # 垂直置中
        
        # 4. 畫出帶有黑色外框的文字 (這樣在雜亂背景下更清楚)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

    # 6. FPS 顯示
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"GPU FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Hand Safety Guard - V4 Pro ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
