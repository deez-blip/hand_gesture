import csv, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "gesture.csv"
MODEL_PATH = BASE_DIR / "models" / "gesture_svm.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset introuvable: {DATA_FILE}. Lance collect_gestures.py avant l'entraînement.")

X, y = [], []
with open(DATA_FILE) as f:
    r = csv.reader(f)
    for row in r:
        y.append(row[0])
        X.append([float(v) for v in row[1:]])
X, y = np.array(X, dtype=np.float32), np.array(y)

if len(X) == 0:
    raise ValueError("Le dataset est vide. Collecte au moins un exemple avant l'entraînement.")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True))
clf.fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
