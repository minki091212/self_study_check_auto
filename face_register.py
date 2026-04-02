import cv2
import numpy as np
import pickle
import os
import insightface
from insightface.app import FaceAnalysis

SAVE_PATH = "face_db.pkl"

def load_db():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}

def save_db(db):
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(db, f)

def main():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    db = load_db()
    cap = cv2.VideoCapture(0)

    while True:
        name = input("등록할 학생 이름 (종료: q): ").strip()
        if name == 'q':
            break

        print(f"{name} - 얼굴을 카메라에 맞추고 'c' 눌러 촬영")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Register", frame)
            key = cv2.waitKey(1)

            if key == ord('c'):
                faces = app.get(frame)

                if not faces:
                    print("얼굴을 찾을 수 없어요. 다시 시도하세요.")
                    continue

                encoding = faces[0].embedding
                db["names"].append(name)
                db["encodings"].append(encoding)
                save_db(db)
                print(f"[등록 완료] {name}")
                break

            elif key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()