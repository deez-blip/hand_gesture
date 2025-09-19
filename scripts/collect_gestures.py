import cv2, csv, time
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import defaultdict

KEY_BINDINGS = {
    "Stop": ['1', '&'],
    "Avance": ['2', 'é'],
    "Recul": ['3', '"'],
    "Droite": ['4', "'"],
    "Gauche": ['5', '('],
}
LABELS = {
    ord(key): label
    for label, keys in KEY_BINDINGS.items()
    for key in keys
}
CAPTURE_INTERVAL = 0.1  # seconds
CAPTURE_LIMIT = 250
BASE_DIR = Path(__file__).resolve().parents[1]
outfile = BASE_DIR / "data" / "gesture.csv"
outfile.parent.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_features(hand_landmarks):
    # 21 points (x,y) normalisés par translation + mise à l'échelle
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = pts[0]  # poignet
    pts -= origin
    scale = np.linalg.norm(pts, axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()  # 42 dims

existing_counts = defaultdict(int)
if outfile.exists():
    with open(outfile, newline="") as existing_file:
        for row in csv.reader(existing_file):
            if row:
                existing_counts[row[0]] += 1

cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     open(outfile, "a", newline="") as f:
    writer = csv.writer(f)

    binding_text = ", ".join(f"{'/'.join(keys)} -> {label}" for label, keys in KEY_BINDINGS.items())
    print("Touches :", binding_text)
    print(f"Chaque geste se coupe automatiquement à {CAPTURE_LIMIT} captures.")
    print("Appuie sur une touche pour démarrer/arrêter la capture auto (0.5s). 'q' pour quitter.")
    label_counts = {label: existing_counts[label] for label in KEY_BINDINGS}
    completed_labels = {label for label, count in label_counts.items() if count >= CAPTURE_LIMIT}
    if completed_labels:
        print("Limite déjà atteinte pour :", ", ".join(sorted(completed_labels)))

    active_label = None
    last_capture = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        counts_text = " ".join(
            f"{label}:{label_counts[label]}/{CAPTURE_LIMIT}"
            for label in KEY_BINDINGS
        )
        cv2.putText(frame, binding_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, counts_text, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"Auto: {active_label or 'off'} (q pour quitter)", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Collecte", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if k in LABELS:
            label = LABELS[k]
            if label in completed_labels:
                print(f"{label} est déjà complet ({CAPTURE_LIMIT}).")
                continue
            if active_label == label:
                active_label = None
                print(f"Capture auto arrêtée pour {label}.")
            else:
                active_label = label
                last_capture = 0.0  # force immediate capture
                print(f"Capture auto activée pour {label} ({label_counts[label]}/{CAPTURE_LIMIT}).")
                if res.multi_hand_landmarks:
                    feats = landmarks_to_features(res.multi_hand_landmarks[0])
                    writer.writerow([label] + feats.tolist())
                    f.flush()
                    last_capture = time.time()
                    label_counts[label] += 1
                    print(f"Saved: {label} ({label_counts[label]}/{CAPTURE_LIMIT})")
                    if label_counts[label] >= CAPTURE_LIMIT:
                        active_label = None
                        completed_labels.add(label)
                        print(f"Limite atteinte pour {label}. Capture arrêtée automatiquement.")

        if active_label and res.multi_hand_landmarks:
            now = time.time()
            if now - last_capture >= CAPTURE_INTERVAL:
                feats = landmarks_to_features(res.multi_hand_landmarks[0])
                writer.writerow([active_label] + feats.tolist())
                f.flush()
                last_capture = now
                label_counts[active_label] += 1
                print(f"Saved: {active_label} ({label_counts[active_label]}/{CAPTURE_LIMIT})")
                if label_counts[active_label] >= CAPTURE_LIMIT:
                    completed_labels.add(active_label)
                    print(f"Limite atteinte pour {active_label}. Capture arrêtée automatiquement.")
                    active_label = None

cap.release()
cv2.destroyAllWindows()
