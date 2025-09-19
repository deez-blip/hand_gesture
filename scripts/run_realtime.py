#!/usr/bin/env python3
import argparse
import os
import collections
from pathlib import Path

import cv2
import numpy as np
import joblib
import mediapipe as mp

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------- Config & parsing ----------------
LABELS = ["Stop", "Avance", "Recul", "Droite", "Gauche"]
VOTE_N = 7
CONF_THRESH = 0.6  # seuil pour tenir compte de la confiance

parser = argparse.ArgumentParser(description="Realtime hand gesture inference (OpenCV + MediaPipe)")
parser.add_argument("--device", type=int, default=2, help="Index VideoCapture OpenCV (correspond à /dev/videoN)")
parser.add_argument("--width", type=int, default=640, help="Largeur capture")
parser.add_argument("--height", type=int, default=480, help="Hauteur capture")
parser.add_argument("--fps", type=int, default=30, help="Framerate capture")
parser.add_argument("--model", type=str, default=None, help="Chemin vers gesture_svm.joblib (override)")
parser.add_argument("--device-path", type=str, default=None, help="Chemin /dev/videoN; prioritaire sur --device")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]  # .../src
MODEL_PATH = BASE_DIR / "models" / "gesture_svm.joblib"

env_model = os.getenv("HAND_GESTURE_MODEL")
if args.model:
    MODEL_PATH = Path(args.model)
elif env_model:
    MODEL_PATH = Path(env_model)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}\n"
                            f"Astuce: --model /chemin/vers/gesture_svm.joblib")

# ---------------- ML / MediaPipe init ----------------
clf = joblib.load(str(MODEL_PATH))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_features(hl):
    """21 points (x,y), centrés & normalisés par l’ampleur max, flatten -> (42,)"""
    pts = np.array([(lm.x, lm.y) for lm in hl.landmark], dtype=np.float32)
    origin = pts[0]
    pts -= origin
    scale = float(np.linalg.norm(pts, axis=1).max()) or 1.0
    pts /= scale
    return pts.flatten()

# ---------------- ROS init ----------------
rclpy.init(args=None)
ros_node = Node('gesture_client')
gesture_pub = ros_node.create_publisher(String, 'gesture_cmd', 10)
last_sent = None

# ---------------- Camera init ----------------
# Try to open by explicit device path first, then by index; try CAP_V4L2 then CAP_ANY

def _open_capture():
    backends = []
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(0)  # CAP_ANY

    tried = []

    # 1) Try path
    if args.device_path:
        for b in backends:
            cap = cv2.VideoCapture(args.device_path, b)
            if cap.isOpened():
                return cap, f"path={args.device_path} backend={b}"
            cap.release()
            tried.append(f"path {args.device_path} (b={b})")

    # 2) Try index
    for b in backends:
        cap = cv2.VideoCapture(args.device, b)
        if cap.isOpened():
            return cap, f"index={args.device} backend={b}"
        cap.release()
        tried.append(f"index {args.device} (b={b})")

    raise RuntimeError("Impossible d’ouvrir la caméra. Tentatives: " + ", ".join(tried))

cap, how = _open_capture()
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
except Exception:
    pass

try:
    ros_node.get_logger().info(f"OpenCV capture -> {how}")
except Exception:
    pass

votes = collections.deque(maxlen=VOTE_N)

# ---------------- Main loop ----------------
try:
    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pred_label = "Stop"
            conf = 0.0

            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                feats = landmarks_to_features(hl).reshape(1, -1)
                probs = clf.predict_proba(feats)[0]
                idx = int(np.argmax(probs))
                pred = str(clf.classes_[idx])
                conf = float(probs[idx])

                votes.append(pred if conf >= CONF_THRESH else "Stop")
                pred_label = collections.Counter(votes).most_common(1)[0][0]

                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            else:
                votes.append("Stop")

            # Publication ROS si changement
            gesture = pred_label.lower()
            if gesture != last_sent:
                gesture_pub.publish(String(data=gesture))
                last_sent = gesture
                try:
                    ros_node.get_logger().info(f"gesture_cmd -> {gesture}")
                except Exception:
                    pass

            # Affichage
            cv2.putText(frame, f"Pred: {pred_label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Inference", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        ros_node.destroy_node()
    finally:
        rclpy.shutdown()