import cv2
from ultralytics import YOLO

model = YOLO("D:/yolov11/ultralytics-8.3.39/hand_gloves_pro_v4.pt") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("打不開鏡頭")
    exit()

print("偵測啟動中... ")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(
        source=frame,
        conf=0.35,         
        iou=0.7,           
        agnostic_nms=False, 
        augment=True,     
        stream=True,
        show=False
    )

    for r in results:
        annotated_frame = r.plot()

        cv2.imshow("YOLOv11 Hand Safety - V4 Pro Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()