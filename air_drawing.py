import os

import cv2
import mediapipe as mp
import numpy as np


def main():
    camera = (cv2.VideoCapture(0, cv2.CAP_DSHOW)
              if os.name == "nt" else cv2.VideoCapture(0))
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try changing 0 to 1 in VideoCapture.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils
    canvas = None
    previous_point = None
    smooth_point = None

    print("Air drawing started. Move your index finger to draw.")
    print("Press C to clear, Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            current_point = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                drawer.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                )

                index_tip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
                height, width = frame.shape[:2]
                raw_point = (int(index_tip.x * width), int(index_tip.y * height))

                if smooth_point is None:
                    smooth_point = raw_point
                else:
                    smooth_point = (
                        int(0.75 * smooth_point[0] + 0.25 * raw_point[0]),
                        int(0.75 * smooth_point[1] + 0.25 * raw_point[1]),
                    )

                pointing_only = (
                    index_tip.y < index_pip.y and middle_tip.y > middle_pip.y
                )
                if pointing_only:
                    current_point = smooth_point
                    cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, current_point, (255, 0, 255), 7)

            previous_point = current_point
            display = frame.copy()
            canvas_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            blended = cv2.addWeighted(frame, 0.35, canvas, 0.65, 0)
            display[canvas_mask] = blended[canvas_mask]
            cv2.imshow("Air drawing test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                canvas[:] = 0
                previous_point = None
                smooth_point = None
            elif key == ord("q"):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
