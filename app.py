"""SignBridge: real-time Indian Sign Language hand-sign to text and speech.

Run: python app.py
Controls: SPACE=append predicted sign, BACKSPACE=undo, C=clear, S=speak, Q=quit.
"""
from pathlib import Path
from collections import Counter, deque
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH = Path(__file__).with_name("signbridge_model.pkl")
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.60


def normalize_landmarks(hand_landmarks):
    """Convert MediaPipe's 21 xyz points to scale/position-normalized features."""
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    points -= points[0]  # wrist becomes origin
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale
    return points.flatten()


def speak(text):
    if not text.strip():
        print("Nothing to speak yet.")
        return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as exc:
        print(f"Speech engine error: {exc}")
        print("Install/repair pyttsx3 and your system's speech engine to enable speech.")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. First run: python train_model.py")

    model = joblib.load(MODEL_PATH)
    print("Loaded signs:", ", ".join(map(str, model.classes_)))

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        hands.close()
        raise SystemExit("Could not open webcam. Check camera permissions or change CAMERA_INDEX.")

    recent_predictions = deque(maxlen=7)
    sentence = []
    current_label = ""
    current_confidence = 0.0
    last_added = None
    no_hand_since = time.time()

    print("\nSignBridge is running. Press SPACE to add the current sign; S to speak; Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            current_label = ""
            current_confidence = 0.0

            if result.multi_hand_landmarks:
                no_hand_since = time.time()
                hand = result.multi_hand_landmarks[0]
                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                features = normalize_landmarks(hand).reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[best_index])
                if current_confidence >= CONFIDENCE_THRESHOLD:
                    recent_predictions.append(str(model.classes_[best_index]))
                    if recent_predictions:
                        current_label, count = Counter(recent_predictions).most_common(1)[0]
                        if count < 4:
                            current_label = ""
                else:
                    recent_predictions.clear()
            else:
                recent_predictions.clear()
                if time.time() - no_hand_since > 0.5:
                    last_added = None

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 145), (25, 25, 25), -1)
            status = f"Prediction: {current_label or 'Show a trained sign'}"
            cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 240, 120), 2)
            cv2.putText(frame, f"Confidence: {current_confidence:.0%}", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(frame, "SPACE Add | S Speak | BACKSPACE Undo | C Clear | Q Quit", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            sentence_text = " ".join(sentence)
            cv2.putText(frame, f"Text: {sentence_text[-75:]}", (15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 90), 1)
            cv2.imshow("SignBridge - ISL to Text & Speech", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and current_label:
                # Prevent accidental repeated additions while the same sign is held.
                if current_label != last_added:
                    sentence.append(current_label)
                    last_added = current_label
                    print("Text:", " ".join(sentence))
            elif key in (8, 127) and sentence:
                sentence.pop()
            elif key == ord('c'):
                sentence.clear()
                last_added = None
            elif key == ord('s'):
                speak(" ".join(sentence))
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
