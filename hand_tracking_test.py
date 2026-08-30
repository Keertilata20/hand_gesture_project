import os

import cv2
import mediapipe as mp


def main():
    camera = (cv2.VideoCapture(0, cv2.CAP_DSHOW)
              if os.name == "nt" else cv2.VideoCapture(0))
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try changing 0 to 1 in VideoCapture.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils

    print("Webcam started. Show your hand; press Q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    drawer.draw_landmarks(
                        frame,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            cv2.imshow("MediaPipe hand-tracking test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
