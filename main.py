import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.spatial import distance as dist
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0
ALERT_ON = False
last_emotion_time = 0
emotion_interval = 10  

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices):
    points = [landmarks[i] for i in eye_indices]
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALERT_ON:
                    ALERT_ON = True
                    print("[ALERT] Drowsiness detected!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALERT_ON = False

        if time.time() - last_emotion_time > emotion_interval:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                print(f"[EMOTION] {emotion}")
                last_emotion_time = time.time()
            except Exception as e:
                print(f"[ERROR] Emotion detection failed: {e}")

    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
