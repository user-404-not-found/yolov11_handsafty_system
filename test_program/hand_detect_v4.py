import cv2
from ultralytics import YOLO

# 1. 載入 v4 模型
model = YOLO("D:/yolov11/ultralytics-8.3.39/hand_gloves_pro_v4.pt") 
#D:/yolov11/ultralytics-8.3.39/hand_gloves_pro_v4.pt
#F:\yolo_hand\model\v4.pt
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("打不開鏡頭")
    exit()

print("偵測啟動中... ")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. 根據 v2 數據微調後的預測參數
    results = model.predict(
        source=frame,
        conf=0.35,         # 根據 F1 曲線調整的最佳信心門檻
        iou=0.7,           # 提高 IOU，防止多手重疊時框框消失
        agnostic_nms=False, # 單一類別模型不需開啟，提升效能
        augment=True,      # 開啟測試增強，應對機床邊緣的刁鑽角度！
        stream=True,
        show=False
    )

    for r in results:
        # 取得畫好框框後的畫面
        annotated_frame = r.plot()

        # 3. 顯示畫面
        cv2.imshow("YOLOv11 Hand Safety - V4 Pro Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()