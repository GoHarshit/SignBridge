"""Collect labelled hand-landmark examples for SignBridge.
Run once per sign, e.g. python collect_data.py and enter HELLO.
S saves one frame; Q quits. Collect varied examples for each sign.
"""
import csv
from pathlib import Path
import cv2
import mediapipe as mp

DATASET_PATH = Path(__file__).with_name("dataset.csv")
HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def main():
    label = input("Enter the sign label (e.g. HELLO): ").strip().upper()
    if not label:
        raise SystemExit("Label cannot be empty.")

    new_file = not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Webcam not detected. Check camera permissions.")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    saved = 0
    try:
        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.65, min_tracking_confidence=0.60,
        ) as detector, open(DATASET_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(HEADER)
            print("Show the sign. Press S to save a sample; Q to quit.")
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Unable to read webcam frame.")
                    break
                frame = cv2.flip(frame, 1)
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    # Save raw xyz coordinates; training and live prediction normalize identically.
                    landmarks = [v for p in hand.landmark for v in (p.x, p.y, p.z)]
                    message = "Hand detected - press S to save"
                    color = (0, 220, 0)
                else:
                    message, color = "Show your hand", (0, 0, 255)
                cv2.putText(frame, f"Label: {label} | Saved: {saved}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 230, 0), 2)
                cv2.putText(frame, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, "S = save sample | Q = quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("SignBridge - Collect Training Data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if landmarks is None:
                        print("No hand detected; sample not saved.")
                    else:
                        writer.writerow([label] + landmarks)
                        file.flush()
                        saved += 1
                        print(f"Saved {saved} sample(s) for {label}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Collection finished. {saved} sample(s) added for {label}.")


if __name__ == "__main__":
    main()
