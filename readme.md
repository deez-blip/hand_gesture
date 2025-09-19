# Hand Gesture Scripts

## Environnement & installation
- **Python** : 3.10+ recommandé. Créez un environnement dédié (`python3 -m venv .venv && source .venv/bin/activate`).
- **Pip** : `pip install --upgrade pip` avant d'ajouter les dépendances.
- **Dépendances main** : `pip install numpy opencv-python mediapipe scikit-learn joblib`.
- **Intégration ROS 2** : assurez-vous d'avoir une distribution ROS 2 fonctionnelle (rclpy, std_msgs). Dans un workspace ROS, sourcez l'environnement (`source /opt/ros/<distro>/setup.bash` puis `source install/setup.bash`) avant d'exécuter `run_realtime.py`.

Ce dossier regroupe les outils utilisés pour collecter les données de gestes, entraîner le classifieur et tester l'inférence temps réel (hors ROS). Le pipeline typique est : 1) collecter les exemples, 2) entraîner le modèle, 3) lancer l'inférence.

## collect_gestures.py
- **But** : enregistrer des exemples de main via webcam et sauvegarder les caractéristiques dans `../data/gesture.csv`.
- **Fonctionnement** : MediaPipe extrait les 21 points de la main, les coordonnées sont centrées et normalisées avant d'être stockées avec l'étiquette du geste. Les touches `1`–`5` (et leurs variantes AZERTY) servent à démarrer/arrêter une capture automatique (10 fps max) pour chaque classe jusqu'à 250 exemples. L'aperçu affiche le rappel des touches, le compteur par classe et permet de quitter avec `q`.
- **Quand l'utiliser** : relancer dès que vous souhaitez enrichir le dataset avec de nouvelles prises ou équilibrer les classes.

## train_classifier.py
- **But** : entraîner un SVM probabiliste à partir des données normalisées et produire `../models/gesture_svm.joblib`.
- **Fonctionnement** : charge `gesture.csv`, sépare le jeu de données (80/20), applique un `StandardScaler` + `SVC` (RBF, `C=10`, `gamma=scale`, `probability=True`), affiche un rapport de classification sur le set de test puis sérialise le pipeline avec joblib.
- **Quand l'utiliser** : après chaque nouvelle collecte ou ajustement du dataset. Vérifiez que `gesture.csv` existe et n'est pas vide, sinon le script s'arrête avec un message explicite.

## run_realtime.py
- **But** : exécuter l'inférence temps réel et publier les commandes correspondantes sur ROS 2 (`gesture_cmd`).
- **Fonctionnement** : charge le modèle SVM (`gesture_svm.joblib`), ouvre la caméra (via index ou chemin explicite), calcule les probabilités avec MediaPipe + scikit-learn et applique un vote glissant sur 7 frames. En cas de changement de geste, le nœud ROS `gesture_client` publie le message `std_msgs/String` correspondant sur le topic `gesture_cmd` (queue size 10) et loggue l'événement. L'affichage OpenCV superpose les landmarks et la confiance (`q` pour quitter).
- **Quand l'utiliser** : pour piloter le robot/logiciel via ROS avec le modèle courant. Ajustez caméra (`--device` ou `--device-path`), résolution et fps selon votre matériel.

## Package ROS 2 `gesture_bridge`
- **But** : convertir les commandes publiées sur `gesture_cmd` en JSON conforme au firmware moteur et les diffuser sur `driver_topic`.
- **Fichier principal** : `gesture_receiver.py` (installé via `ros2 run gesture_bridge gesture_receiver`).Il écoute `gesture_cmd`, exécute un auto-test au démarrage (`droite` puis `stop`) et traduit chaque libellé (`stop`, `avance`, `recul`, `droite`, `gauche`) vers les vitesses moteurs attendues.
- **Implémentation** : le paquet vit dans le workspace ROS (`r2_ws/src/gesture_bridge`). Après modification, lancer `colcon build --packages-select gesture_bridge` puis `source install/setup.bash` pour exposer l'exécutable.
- **Quand l'utiliser** : Démarrez le nœud avec `ros2 run gesture_bridge gesture_receiver` afin d'alimenter `driver_topic` pendant que `run_realtime.py` publie `gesture_cmd`.

### Intégrer le projet dans un workspace ROS 2
1. **Copier les ressources** : placez-vous dans votre workspace (`r2_ws` par exemple) et copiez les dossiers `data/`, `models/`, `scripts/` (facultatif si vous n’exécutez pas les scripts Python depuis le Pi) ainsi que le dossier du package `gesture_bridge/` dans `src/`. On obtient `r2_ws/src/data`, `r2_ws/src/models`, `r2_ws/src/scripts` et `r2_ws/src/gesture_bridge`.
2. **Vérifier le package** : assurez-vous que `gesture_bridge/setup.py` expose bien `gesture_receiver` dans `entry_points['console_scripts']`.
3. **Construire** : depuis la racine du workspace (`cd r2_ws`), lancez `colcon build --packages-select gesture_bridge`.
4. **Sourcer** : `source install/setup.bash` (ou `.zsh`) pour que le nouvel exécutable soit disponible.
5. **Lancer** : `ros2 run gesture_bridge gesture_receiver` (le nœud attend des messages `gesture_cmd`).

### Flux ROS lié aux gestes
- **`gesture_client` (`run_realtime.py`)** : publie `gesture_cmd` avec les chaînes `stop`, `avance`, `recul`, `droite`, `gauche` dès qu'un geste différent est reconnu.
- **`gesture_receiver` (package `gesture_bridge`)** : souscrit à `gesture_cmd`, applique la correspondance vers le JSON moteur (`motor1Speed`, `motor2Speed`, message écran) et publie le résultat sur `driver_topic` (`std_msgs/String`, queue 1). Un auto-test optionnel valide la liaison au démarrage.
- **`serial_json_sender` (`pico_serial_node.py`)** : souscrit à `driver_topic`, nettoie le JSON et l'envoie via USB au microcontrôleur (détection auto de `/dev/ttyACM*`). Les réponses série sont journalisées pour le debug.
- **Résumé** : `run_realtime.py` → `gesture_cmd` → `gesture_receiver` (`gesture_bridge`) → `driver_topic` → `pico_serial_node.py` → carte Pico.

## Réinitialiser le dataset / modèle
- **Objectif** : repartir de zéro avant une nouvelle session de collecte/entraînement.
- **Étapes** :
  1. Supprimer ou renommer `../data/gesture.csv` pour remettre les compteurs de collecte à zéro.
  2. Supprimer `../models/gesture_svm.joblib` afin d'éviter de recharger un ancien modèle lors de l'inférence.
  3. Relancer `collect_gestures.py` pour générer un nouveau dataset vierge, puis `train_classifier.py` une fois la collecte terminée.
- **Astuce** : utilisez `ls ../data` et `ls ../models` pour vérifier qu'aucun fichier legacy ne reste avant de recommencer.

### Dépendances communes
- OpenCV (`cv2`), MediaPipe, NumPy
- scikit-learn & joblib pour l'entraînement / l'inférence
- Une webcam accessible par OpenCV

Astuce : placez-vous dans ce dossier et activez l'environnement Python adapté (`poetry`, `venv`, etc.) avant d'exécuter les scripts.