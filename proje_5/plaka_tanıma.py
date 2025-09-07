import cv2
from ultralytics import YOLO
import easyocr

# Model ve video yolu
model_path = r"C:\plaka_tanima_dosyalari\archive\platebest.pt"
video_path = r"C:\plaka_tanima_dosyalari\plakaVideo.MOV"

# YOLO ve OCR
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=True)

# Video aç
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Video açılamadı. Yol doğru mu kontrol et.")
    exit()

frame_count = 0
ocr_results_cache = []

# Videonun orijinal FPS'si
video_fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / video_fps)

# Videonun orijinal çözünürlüğünü al
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Pencereyi orijinal çözünürlüğe göre aç
cv2.namedWindow("Plaka Tanima", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plaka Tanima", orig_w, orig_h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, verbose=False)[0]
    annotated_frame = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]

        plate = ""
        if roi.size > 0:
            if frame_count % 30 == 0:  # OCR her 30 karede
                ocr_result = reader.readtext(roi)
                if ocr_result:
                    plate = ocr_result[0][1]
                    ocr_results_cache.append(plate)
            elif len(ocr_results_cache) > 0:
                plate = ocr_results_cache[-1]

        if plate:
            cv2.putText(annotated_frame, f"Plaka: {plate}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    # Orijinal boyutla göster
    cv2.imshow("Plaka Tanima", annotated_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
