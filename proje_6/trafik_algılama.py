import cv2
import time
from collections import deque, defaultdict
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"

video_path = r"C:\trafik_okuma\trafikVideo.mp4"

TARGET_CLASSES = ["car", "bus", "truck", "motorcycle"]

CLASS_COLORS = {
    "car": (60, 220, 60),
    "bus": (0, 165, 255),
    "truck": (0, 0, 255),
    "motorcycle": (255, 0, 0),
}

DENSITY_THRESHOLDS = {
    "low": 8,
    "medium": 15
}

CONF_THRES = 0.25
IOU_THRES = 0.5


def put_box_with_label(img, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def draw_info_panel(frame, counts, density_text, fps):
    panel_w = 270
    panel_h = 25 * (len(counts) + 3)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    cv2.putText(frame, "Sayac (Line-Cross):", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    y = 65
    total = 0
    for cname in ["car", "bus", "truck", "motorcycle"]:
        c = counts[cname]
        total += c
        cv2.putText(frame, f"{cname:<11}: {c}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLASS_COLORS[cname], 2, cv2.LINE_AA)
        y += 25

    cv2.putText(frame, f"TOPLAM: {total}", (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    text = f"Yogunluk: {density_text}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    color = (0, 220, 0) if density_text == "DUSUK" else (0, 215, 255) if density_text == "ORTA" else (0, 0, 255)
    pad = 10
    X = frame.shape[1] - tw - 2 * pad - 10
    cv2.rectangle(frame, (X, 10), (X + tw + 2 * pad, 10 + th + 2 * pad), (30, 30, 30), -1)
    cv2.putText(frame, text, (X + pad, 10 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (X + pad, 10 + th + pad + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

def draw_density_bar(frame, density_text):

    h, w = frame.shape[:2]
    bar_h = max(30, h // 20)
    margin = 8
    y1 = h - bar_h - margin
    y2 = h - margin

    if density_text == "DUSUK":
        color = (0, 220, 0)
        label = "DUSUK YOGUNLUK"
    elif density_text == "ORTA":
        color = (0, 215, 255)
        label = "ORTA YOGUNLUK"
    else:
        color = (0, 0, 255)
        label = "YUKSEK YOGUNLUK"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = max(10, (w - tw) // 2)
    y = y1 + (bar_h + th) // 2
    cv2.putText(frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

def density_label(avg_active, thr=DENSITY_THRESHOLDS):
    if avg_active < thr["low"]:
        return "DUSUK"
    elif avg_active <= thr["medium"]:
        return "ORTA"
    else:
        return "YUKSEK"

def main():
    model = YOLO(MODEL_NAME)

    name_to_id = {v: k for k, v in model.names.items()}
    target_ids = [name_to_id[c] for c in TARGET_CLASSES if c in name_to_id]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video acilamadi. Dosya yolunu kontrol et.")

    ret, sample = cap.read()
    if not ret:
        raise RuntimeError("Video ilk frame okunamadi.")
    h, w = sample.shape[:2]
    line_y = int(h * 0.6)
    line_color = (255, 255, 0)
    line_thickness = 2

    counts = defaultdict(int)
    prev_centers_y = dict()
    passed_ids = set()

    active_deque = deque(maxlen=30)

    prev_time = time.time()
    fps = 0.0

    print("q -> cikis, p -> duraklat/devam, r -> sayaci sifirla")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                conf=CONF_THRES,
                iou=IOU_THRES,
                classes=target_ids,
                tracker="bytetrack.yaml",
                verbose=False
            )

            vis = frame.copy()
            active_ids_now = set()

            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    for i in range(len(boxes)):
                        b = boxes[i]
                        cls_id = int(b.cls[0].item())
                        track_id = int(b.id[0].item()) if b.id is not None else None

                        if cls_id not in target_ids or track_id is None:
                            continue

                        cls_name = model.names[cls_id]
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        active_ids_now.add(track_id)

                        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                        label = f"{cls_name}#{track_id}"
                        put_box_with_label(vis, x1, y1, x2, y2, label, color, thickness=2)
                        cv2.circle(vis, (cx, cy), 3, color, -1)

                        if track_id not in passed_ids:
                            prev_y = prev_centers_y.get(track_id, None)
                            if prev_y is not None:
                                crossed_down = prev_y < line_y <= cy
                                crossed_up = prev_y > line_y >= cy
                                if crossed_down or crossed_up:
                                    counts[cls_name] += 1
                                    passed_ids.add(track_id)
                            prev_centers_y[track_id] = cy
                        else:
                            prev_centers_y[track_id] = cy

            cv2.line(vis, (0, line_y), (w, line_y), line_color, line_thickness)

            active_deque.append(len(active_ids_now))
            avg_active = sum(active_deque) / max(1, len(active_deque))
            dens_text = density_label(avg_active)

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            draw_info_panel(vis, counts, dens_text, fps)

            draw_density_bar(vis, dens_text)

            cv2.imshow("Trafik Analizi", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            counts = defaultdict(int)
            passed_ids.clear()
            prev_centers_y.clear()
            active_deque.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
