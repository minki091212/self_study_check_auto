# roi_setup.py - 좌석 ROI를 마우스로 설정하고 JSON으로 저장

import cv2
import json

seats = {}
drawing = False
start_x, start_y = -1, -1
seat_counter = 0
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, seat_counter, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = frame_copy.copy()
            cv2.rectangle(temp, (start_x, start_y), (x, y), (255, 255, 0), 2)
            cv2.imshow("ROI Setup", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        seat_id = f"seat_{seat_counter}"
        seats[seat_id] = (
            min(start_x, x), min(start_y, y),
            max(start_x, x), max(start_y, y)
        )
        seat_counter += 1
        cv2.rectangle(frame_copy, (min(start_x,x), min(start_y,y)),
                      (max(start_x,x), max(start_y,y)), (0, 255, 255), 2)
        cv2.putText(frame_copy, seat_id,
                    (min(start_x,x), min(start_y,y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("ROI Setup", frame_copy)
        print(f"[추가] {seat_id}: {seats[seat_id]}")

def main():
    global frame_copy

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("카메라 열기 실패")
        return

    frame_copy = frame.copy()
    cv2.imshow("ROI Setup", frame_copy)
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    print("드래그로 좌석 ROI 설정 | 'z': 마지막 취소 | 's': 저장 후 종료")

    while True:
        key = cv2.waitKey(1)

        if key == ord('s'):
            # 튜플 → 리스트로 변환해서 JSON 저장
            save_data = {k: list(v) for k, v in seats.items()}
            with open("seats.json", "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"저장 완료: seats.json ({len(seats)}개 좌석)")
            break

        elif key == ord('z') and seats:
            # 마지막 좌석 취소
            last_key = list(seats.keys())[-1]
            del seats[last_key]
            print(f"[취소] {last_key} 삭제")

        elif key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()