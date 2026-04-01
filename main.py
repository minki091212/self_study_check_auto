from ultralytics import YOLO
import cv2

# ----------------------
# 설정
# ----------------------
MODEL_PATH = "yolov8n.pt"

# 좌석 ROI (x1, y1, x2, y2)
seats = {
    "A": (100, 100, 200, 200),
    "B": (250, 100, 350, 200),
}

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
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 탐지
        results = model(frame)[0]

        # 좌석 상태 초기화
        seat_status = {key: False for key in seats}

        # 사람 탐지 결과 처리
        for box in results.boxes:
            cls = int(box.cls[0])

            # person만 사용 (class 0)
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = get_center((x1, y1, x2, y2))

            # 좌석 매핑
            for seat_id, seat in seats.items():
                if is_in_seat((cx, cy), seat):
                    seat_status[seat_id] = True

            # 디버깅: 사람 bbox + 중심점 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 좌석 ROI 그리기
        for seat_id, (x1, y1, x2, y2) in seats.items():
            color = (0, 255, 0) if seat_status[seat_id] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{seat_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 상태 출력
        print(seat_status)

        # 화면 출력
        cv2.imshow("Seat Check MVP", frame)

        # ESC로 종료
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()