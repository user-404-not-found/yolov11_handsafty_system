# ==========================================
# Project: 手部安全監控系統 (Hand Safety system)
# Author:  凃瑋禮
# Date:    2026-09-25
# Version: v4.0 (ONNX GPU Accelerated)
# Description: 支援同時雙鏡頭的實時手部偵測與 ROI 入侵報警系統
# 操作說明：
# e：進入 ROI 編輯模式
# 左鍵：新增 ROI 點
# 右鍵：刪除最後一點
# s：儲存 ROI
# c：清除目前點
# r：重設 ROI
# q：離開
# ==========================================

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "model" / "v4.onnx"
DEFAULT_CONFIG = PROJECT_ROOT / "test_program" / "roi_config.json"
WINDOW_NAMES = ("Hand Safety - Camera 0", "Hand Safety - Camera 1")


class RTSPCamera:
    """Read an RTSP stream in the background and expose only its newest frame."""

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                with self.lock:
                    self.ret = False
                time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()
        self.thread.join(timeout=1.0)


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpadded = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    padding_x = (new_shape[1] - new_unpadded[0]) / 2
    padding_y = (new_shape[0] - new_unpadded[1]) / 2

    if shape[::-1] != new_unpadded:
        image = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)

    top = int(round(padding_y - 0.1))
    bottom = int(round(padding_y + 0.1))
    left = int(round(padding_x - 0.1))
    right = int(round(padding_x + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, ratio, (padding_x, padding_y)


def default_roi():
    return [[0.0, 0.60], [1.0, 0.60], [1.0, 1.0], [0.0, 1.0]]


def load_rois(config_path):
    if not config_path.exists():
        return [default_roi(), default_roi()]

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        rois = config.get("rois", [])
        if len(rois) == 2 and all(len(roi) >= 3 for roi in rois):
            return rois
    except (OSError, json.JSONDecodeError):
        pass

    print(f"無法讀取 ROI 設定，改用預設區域：{config_path}")
    return [default_roi(), default_roi()]


def save_rois(config_path, rois):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump({"rois": rois}, config_file, indent=2)
    print(f"ROI 已儲存：{config_path}")


def normalized_to_pixels(roi, width, height):
    return np.array(
        [[int(x * width), int(y * height)] for x, y in roi], dtype=np.int32
    )


def pixels_to_normalized(points, width, height):
    return [
        [round(max(0, min(width, x)) / width, 6), round(max(0, min(height, y)) / height, 6)]
        for x, y in points
    ]


def create_session(model_path):
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [provider for provider in providers if provider in available]
    if not providers:
        raise RuntimeError("找不到 ONNX Runtime 可用的推理 provider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"ONNX 模型：{model_path}")
    print(f"推理 provider：{session.get_providers()[0]}")
    return session


def run_inference(session, frame, confidence_threshold):
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
    input_width = input_shape[3] if isinstance(input_shape[3], int) else 640
    input_image, ratio, (padding_x, padding_y) = letterbox(
        frame, (input_height, input_width)
    )
    blob = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, :]
    output = session.run(None, {input_info.name: blob})[0]
    predictions = np.squeeze(output)
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []
    for prediction in predictions:
        if len(prediction) < 5:
            continue
        cx, cy, box_width, box_height = prediction[:4]
        confidence = float(prediction[4])
        if len(prediction) > 5:
            confidence *= float(np.max(prediction[5:]))
        if confidence < confidence_threshold:
            continue

        x1 = int((cx - box_width / 2 - padding_x) / ratio)
        y1 = int((cy - box_height / 2 - padding_y) / ratio)
        x2 = int((cx + box_width / 2 - padding_x) / ratio)
        y2 = int((cy + box_height / 2 - padding_y) / ratio)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)

    selected = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
    return [
        (boxes[index][0], boxes[index][1], boxes[index][0] + boxes[index][2],
         boxes[index][1] + boxes[index][3], scores[index])
        for index in np.array(selected).reshape(-1)
    ]


def draw_roi(frame, roi, intrusion):
    height, width = frame.shape[:2]
    points = normalized_to_pixels(roi, width, height)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 0, 255))
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, [points], True, (0, 0, 255), 3)
    label = "DANGER ZONE" if not intrusion else "INTRUSION"
    cv2.putText(frame, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def annotate_frame(frame, detections, roi, camera_index, fps, editing):
    height, width = frame.shape[:2]
    roi_points = normalized_to_pixels(roi, width, height)
    intrusion = False
    for x1, y1, x2, y2, confidence in detections:
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        hand_point = (int((x1 + x2) / 2), y2)
        inside = cv2.pointPolygonTest(roi_points, hand_point, False) >= 0
        intrusion = intrusion or inside
        color = (0, 0, 255) if inside else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if inside else 2)
        cv2.putText(frame, f"Hand {confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    draw_roi(frame, roi, intrusion)
    if intrusion:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
        cv2.putText(frame, "STOP", (width // 2 - 55, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
    cv2.putText(frame, f"CAM {camera_index}  FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if editing:
        cv2.putText(frame, "ROI EDITING", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    return intrusion


def main():
    parser = argparse.ArgumentParser(description="雙鏡頭 ONNX 手部安全辨識")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera0", required=True, help="第一顆 IP Camera 的 RTSP URL")
    parser.add_argument("--camera1", required=True, help="第二顆 IP Camera 的 RTSP URL")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    session = create_session(args.model)
    cameras = [RTSPCamera(url) for url in (args.camera0, args.camera1)]
    if not all(camera.is_opened() for camera in cameras):
        for camera in cameras:
            camera.release()
        raise RuntimeError("無法開啟 RTSP 串流，請確認網址、帳號密碼與網路連線")

    rois = load_rois(args.config)
    edit_points = [[], []]
    frame_sizes = [(640, 480), (640, 480)]
    editing = False

    def on_mouse(event, x, y, _flags, camera_index):
        if not editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            edit_points[camera_index].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and edit_points[camera_index]:
            edit_points[camera_index].pop()

    for camera_index, window_name in enumerate(WINDOW_NAMES):
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse, camera_index)

    previous_time = time.perf_counter()
    try:
        while True:
            frames = []
            for camera in cameras:
                success, frame = camera.read()
                if not success:
                    frames = []
                    break
                frames.append(frame)

            if len(frames) != len(cameras):
                cv2.waitKey(1)
                continue

            now = time.perf_counter()
            fps = 2 / max(now - previous_time, 1e-6)
            previous_time = now
            intrusions = []
            for index, frame in enumerate(frames):
                frame_sizes[index] = (frame.shape[1], frame.shape[0])
                detections = run_inference(session, frame, args.confidence)
                if editing and len(edit_points[index]) >= 3:
                    active_roi = pixels_to_normalized(edit_points[index], *frame_sizes[index])
                else:
                    active_roi = rois[index]
                intrusions.append(annotate_frame(frame, detections, active_roi, index, fps, editing))
                if editing:
                    for point in edit_points[index]:
                        cv2.circle(frame, point, 5, (0, 255, 255), -1)
                cv2.imshow(WINDOW_NAMES[index], frame)

            if any(intrusions):
                print("\r[警告] 偵測到手部進入危險區域       ", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                editing = not editing
                edit_points = [[], []]
                print("\nROI 編輯模式：" + ("開啟" if editing else "關閉"))
            elif key == ord("c") and editing:
                edit_points = [[], []]
            elif key == ord("s") and editing:
                if all(len(points) >= 3 for points in edit_points):
                    rois = [pixels_to_normalized(points, *frame_sizes[index])
                            for index, points in enumerate(edit_points)]
                    save_rois(args.config, rois)
                    editing = False
                    edit_points = [[], []]
                else:
                    print("\n每個鏡頭至少需要 3 個 ROI 點")
            elif key == ord("r"):
                rois = [default_roi(), default_roi()]
                save_rois(args.config, rois)
                edit_points = [[], []]
    finally:
        for camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()