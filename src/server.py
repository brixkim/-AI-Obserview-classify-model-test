import asyncio
import time
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(APP_DIR, "index.html")

app = FastAPI()
det_model = YOLO("./models/yolov12n-face.onnx", task="detect")
cls_model = YOLO("./models/best_3cls.pt", task="classify")

# class_colors = {
#     "real": (0, 255, 0), # Green
#     "print": (255, 0, 0), # Blue
#     "mask": (0, 165, 255), # Orange
#     "replay": (0, 0, 255) # Red
# }

class_colors = {
    "real": (0, 255, 0),
    "print": (255, 0, 0),
    "screen": (0, 0, 255)
}

@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()

def bytes_to_bgrimage(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def bgr_to_jpeg_bytes(img):
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None

def predict_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    results = det_model.predict(source=frame_bgr, verbose=False)
    r = results[0]

    boxes = r.boxes.xyxy
    if boxes is None or len(boxes) == 0:
        return frame_bgr

    boxes_np = boxes.cpu().numpy().astype(int)
    boxes_np = sorted(boxes_np, key=lambda b: b[0])  # x1 기준 정렬

    faces = []
    coords = []
    H, W = frame_bgr.shape[:2]

    for (x1, y1, x2, y2) in boxes_np:

        # clip
        x1 = max(0, min(W, x1))
        y1 = max(0, min(H, y1))
        x2 = max(1, min(W, x2))
        y2 = max(1, min(H, y2))

        # 너무 작은 박스 skip
        # if (x2 - x1) < 40 or (y2 - y1) < 40:
        #     continue

        crop = frame_bgr[y1:y2, x1:x2].copy()

        # crop이 이상하면 skip
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (224, 224))

        faces.append(crop_resized)
        coords.append((x1, y1, x2, y2))

    if not faces:
        return frame_bgr

    cls_results = []
    for face in faces:
        out = cls_model.predict(source=face, imgsz=224, verbose=False)[0]
        cls_results.append(out)

    annotated = frame_bgr.copy()

    for (x1, y1, x2, y2), cr in zip(coords, cls_results):
        probs = cr.probs
        cls_id = int(probs.top1)
        cls_name = cr.names[cls_id]

        if cls_name == "live":
            cls_name = "real"

        conf = float(probs.top1conf)
        color = class_colors.get(cls_name, (255,255,255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name.upper()} {conf:.2f}"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1-th-5), (x1+tw+5, y1), color, -1)

        cv2.putText(
            annotated, label, (x1+2, y1-2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,0,0) if sum(color)>255 else (255,255,255),
            1, cv2.LINE_AA
        )

    return annotated



@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    print("[WS] connected")
    try:
        while True:
            data = await ws.receive_bytes()
            # print(f"[WS] recv {len(data)} bytes")
            frame = bytes_to_bgrimage(data)
            loop = asyncio.get_event_loop()
            annotated = await loop.run_in_executor(None, predict_bgr, frame)
            out = bgr_to_jpeg_bytes(annotated)
            # print(f"[WS] send {len(out)} bytes")
            await ws.send_bytes(out)
    except WebSocketDisconnect:
        print("[WS] disconnected")
