import cv2
import pickle
import time
import os
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from collections import defaultdict

# ----------------------
# 설정
# ----------------------
MODEL_PATH = "yolov8n.pt"
FACE_DB_PATH = "face_db.pkl"
ABSENCE_THRESHOLD = 10
RECHECK_INTERVAL = 5
FACE_THRESHOLD = 0.5      # 코사인 유사도 기준 (높을수록 엄격)
UNKNOWN_TIMEOUT = 30

# ----------------------
# 얼굴 DB 로드
# ----------------------
def load_face_db(path):
    if not os.path.exists(path):
        print("face_db.pkl 없음. face_register.py 먼저 실행하세요.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------
# 코사인 유사도
# ----------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------
# 얼굴 인식
# ----------------------
def recognize_face(face_embedding, db, threshold=FACE_THRESHOLD):
    best_name = None
    best_score = -1

    for name, enc in zip(db["names"], db["encodings"]):
        score = cosine_similarity(face_embedding, enc)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score

# ----------------------
# 상태 관리
# ----------------------
class StudentTracker:
    def __init__(self):
        self.track_map = {}
        self.last_seen = defaultdict(float)
        self.last_recheck = defaultdict(float)
        self.unknown_since = {}
        self.attendance = defaultdict(lambda: "absent")

    def update(self, track_id, name):
        self.track_map[track_id] = name
        self.last_seen[track_id] = time.time()
        if name != "unknown":
            self.attendance[name] = "present"
            self.unknown_since.pop(track_id, None)

    def mark_seen(self, track_id):
        self.last_seen[track_id] = time.time()

    def need_recheck(self, track_id):
        now = time.time()
        if track_id not in self.track_map:
            return True

        name = self.track_map.get(track_id)
        if name == "unknown":
            if track_id not in self.unknown_since:
                self.unknown_since[track_id] = now
            if now - self.unknown_since[track_id] > UNKNOWN_TIMEOUT:
                return False
            return now - self.last_recheck[track_id] > RECHECK_INTERVAL

        return False

    def get_absence_status(self, track_id):
        elapsed = time.time() - self.last_seen.get(track_id, time.time())
        return elapsed > ABSENCE_THRESHOLD, int(elapsed)

    def cleanup_old_tracks(self, active_ids, timeout=30):
        now = time.time()
        dead = [
            tid for tid, t in self.last_seen.items()
            if tid not in active_ids and now - t > timeout
        ]
        for tid in dead:
            name = self.track_map.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_recheck.pop(tid, None)
            self.unknown_since.pop(tid, None)
            if name and name != "unknown":
                if name not in self.track_map.values():
                    self.attendance[name] = "absent"
                    print(f"[이탈] {name} absent 처리")

# ----------------------
# 메인
# ----------------------
def main():
    face_db = load_face_db(FACE_DB_PATH)
    if face_db is None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device)

    # InsightFace 초기화
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    tracker = StudentTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    print(f"[시작] device: {device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ByteTrack 추적
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )[0]

        active_ids = set()

        if results.boxes.id is not None:
            # 얼굴 인식이 필요한 track_id가 있으면 한 번만 호출
            need_face = any(
                tracker.need_recheck(int(tid))
                for tid in results.boxes.id
            )
            faces = face_app.get(frame) if need_face else []

            for box, track_id in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(track_id)
                active_ids.add(tid)
                tracker.mark_seen(tid)

                if tracker.need_recheck(tid):
                    tracker.last_recheck[tid] = time.time()

                    # bbox 안에 있는 얼굴 찾기
                    matched_face = None
                    for face in faces:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        # 얼굴 중심이 사람 bbox 안에 있는지 확인
                        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                        if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                            matched_face = face
                            break

                    if matched_face is not None:
                        name, score = recognize_face(matched_face.embedding, face_db)
                        if name:
                            tracker.update(tid, name)
                            print(f"[인식] ID:{tid} → {name} (score:{score:.2f})")
                        else:
                            tracker.update(tid, "unknown")
                            print(f"[미인식] ID:{tid} score:{score:.2f}")
                    else:
                        tracker.update(tid, "unknown")

                name = tracker.track_map.get(tid, "unknown")
                is_absent, elapsed = tracker.get_absence_status(tid)

                # 색상
                if name == "unknown":
                    color = (128, 128, 128)
                elif is_absent:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} (ID:{tid})",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        tracker.cleanup_old_tracks(active_ids)

        # 출석 현황
        y_offset = 20
        cv2.putText(frame, "=== 출석 현황 ===", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for student, status in tracker.attendance.items():
            y_offset += 25
            color = (0, 255, 0) if status == "present" else (0, 0, 255)
            cv2.putText(frame, f"{student}: {status}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Self Study Check", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()