import os

import cv2
import mediapipe as mp
import numpy as np
import math


COLORS = {
    0: ((0, 0, 255), "RED"),
    2: ((255, 0, 0), "BLUE"),
    3: ((0, 255, 0), "GREEN"),
    4: ((0, 255, 255), "YELLOW"),
}


def raised_fingers(hand):
    points = mp.solutions.hands.HandLandmark
    wrist = hand.landmark[points.WRIST]

    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    pairs = [
        (points.INDEX_FINGER_TIP, points.INDEX_FINGER_PIP),
        (points.MIDDLE_FINGER_TIP, points.MIDDLE_FINGER_PIP),
        (points.RING_FINGER_TIP, points.RING_FINGER_PIP),
        (points.PINKY_TIP, points.PINKY_PIP),
    ]
    # Compare each fingertip with its wrist-relative bend. This is less
    # sensitive to camera angle than checking only the y-coordinate.
    return sum(
        distance(hand.landmark[tip], wrist)
        > distance(hand.landmark[pip], wrist) * 1.10
        for tip, pip in pairs
    )


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
    color = (255, 0, 255)
    color_name = "PURPLE"
    window_name = "Air drawing V2 - gesture colors"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    candidate_count = None
    candidate_frames = 0

    print("Air drawing V2 started.")
    print("Index only: draw | 2 fingers: blue | 3: green | 4: yellow | fist: red")
    print("Press B/G/R/Y to choose a color, C to clear, Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)
                frame_height, frame_width = frame.shape[:2]
                cv2.resizeWindow(window_name, frame_width, frame_height)
                window_sized = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            current_point = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                count = raised_fingers(hand)

                # Require the same gesture for several frames before changing
                # color, preventing one noisy frame from switching colors.
                if count == candidate_count:
                    candidate_frames += 1
                else:
                    candidate_count = count
                    candidate_frames = 1
                if candidate_frames >= 8 and count in COLORS:
                    color, color_name = COLORS[count]

                index_tip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
                height, width = frame.shape[:2]
                raw_point = (int(index_tip.x * width), int(index_tip.y * height))
                smooth_point = raw_point if smooth_point is None else (
                    int(0.85 * smooth_point[0] + 0.15 * raw_point[0]),
                    int(0.85 * smooth_point[1] + 0.15 * raw_point[1]),
                )

                # More forgiving draw gesture: index up, middle down.
                # Other fingers can be in any position.
                drawing_gesture = (
                    count == 1
                    and index_tip.y < index_pip.y
                    and middle_tip.y > middle_pip.y
                )
                if drawing_gesture:
                    current_point = smooth_point
                    cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, current_point, color, 7)

            previous_point = current_point
            display = frame.copy()
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            blended = cv2.addWeighted(frame, 0.35, canvas, 0.65, 0)
            display[mask] = blended[mask]
            cv2.rectangle(display, (0, 0), (300, 48), (35, 35, 35), -1)
            cv2.putText(display, f"COLOR: {color_name}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("b"):
                color, color_name = COLORS[2]
            elif key == ord("g"):
                color, color_name = COLORS[3]
            elif key == ord("r"):
                color, color_name = COLORS[0]
            elif key == ord("y"):
                color, color_name = COLORS[4]
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
