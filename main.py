# main.py - ByteTrack + ROI 좌석 체크

from ultralytics import YOLO
import cv2
import json
from collections import defaultdict
import time
import torch

# ----------------------
# 설정
# ----------------------
MODEL_PATH = "yolov8n.pt"
SEATS_PATH = "seats.json"
ABSENCE_THRESHOLD = 10  # 자리 비움 판정까지 걸리는 시간 (초)

# ----------------------
# 좌석 로드
# ----------------------
def load_seats(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items()}

# ----------------------
# 유틸 함수
# ----------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def is_in_seat(center, seat):
    x, y = center
    x1, y1, x2, y2 = seat
    return x1 <= x <= x2 and y1 <= y <= y2

# ----------------------
# 메인
# ----------------------
def main():
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    seats = load_seats(SEATS_PATH)

    # 좌석별 마지막으로 사람이 있었던 시간
    last_seen = {seat_id: time.time() for seat_id in seats}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ByteTrack 추적 (tracker="bytetrack.yaml" 이 핵심)
        results = model.track(
            frame,
            persist=True,           # 프레임 간 ID 유지
            tracker="bytetrack.yaml",
            classes=[0],            # person only
            verbose=False
        )[0]

        # 좌석 점유 초기화
        seat_occupied = {seat_id: False for seat_id in seats}

        # 탐지 결과 처리
        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(track_id)
                cx, cy = get_center((x1, y1, x2, y2))

                # bbox + track ID 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 좌석 매핑
                for seat_id, seat in seats.items():
                    if is_in_seat((cx, cy), seat):
                        seat_occupied[seat_id] = True
                        last_seen[seat_id] = time.time()

        # 좌석 ROI 그리기 + 자리 비움 판정
        now = time.time()
        for seat_id, (x1, y1, x2, y2) in seats.items():
            elapsed = now - last_seen[seat_id]
            is_absent = elapsed > ABSENCE_THRESHOLD

            if seat_occupied[seat_id]:
                color = (0, 255, 0)   # 초록: 착석
                label = f"{seat_id} O"
            elif is_absent:
                color = (0, 0, 255)   # 빨강: 자리비움
                label = f"{seat_id} X ({int(elapsed)}s)"
            else:
                color = (0, 165, 255) # 주황: 잠깐 미탐지 (버퍼)
                label = f"{seat_id} ?"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 자리 비움 알림 (콘솔)
            if is_absent and not seat_occupied[seat_id]:
                print(f"[알림] {seat_id} 자리 비움 ({int(elapsed)}초 경과)")

        cv2.imshow("Seat Check", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()