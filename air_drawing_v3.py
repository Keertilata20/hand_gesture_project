import math

import cv2
import mediapipe as mp
import numpy as np


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
    return sum(
        distance(hand.landmark[tip], wrist)
        > distance(hand.landmark[pip], wrist) * 1.10
        for tip, pip in pairs
    )


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    candidate_count = None
    candidate_frames = 0
    manual_eraser = False
    window_name = "Air drawing V3 - gesture eraser"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Air drawing V3 started.")
    print("Index only: draw | closed fist: erase")
    print("2 fingers: blue | 3: green | 4: yellow | B/G/R/Y: colors")
    print("Press E to toggle eraser, C to clear, Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)
                height, width = frame.shape[:2]
                cv2.resizeWindow(window_name, width, height)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            current_point = None
            mode = "READY"

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                count = raised_fingers(hand)

                if count == candidate_count:
                    candidate_frames += 1
                else:
                    candidate_count = count
                    candidate_frames = 1

                if candidate_frames >= 8 and count in COLORS:
                    color, color_name = COLORS[count]

                points = mp.solutions.hands.HandLandmark
                index_tip = hand.landmark[points.INDEX_FINGER_TIP]
                index_pip = hand.landmark[points.INDEX_FINGER_PIP]
                middle_tip = hand.landmark[points.MIDDLE_FINGER_TIP]
                middle_pip = hand.landmark[points.MIDDLE_FINGER_PIP]
                height, width = frame.shape[:2]
                raw_point = (int(index_tip.x * width), int(index_tip.y * height))
                smooth_point = raw_point if smooth_point is None else (
                    int(0.85 * smooth_point[0] + 0.15 * raw_point[0]),
                    int(0.85 * smooth_point[1] + 0.15 * raw_point[1]),
                )

                drawing_gesture = (
                    index_tip.y < index_pip.y
                    and middle_tip.y > middle_pip.y
                )
                # Require a folded index finger for the fist eraser. This
                # prevents a noisy finger count from blocking drawing.
                fist_gesture = count == 0 and index_tip.y > index_pip.y
                eraser_active = manual_eraser or fist_gesture
                if eraser_active:
                    mode = "ERASER"
                    palm = hand.landmark[points.MIDDLE_FINGER_MCP]
                    eraser_point = (int(palm.x * width), int(palm.y * height))
                    cv2.circle(frame, eraser_point, 35, (255, 255, 255), 2)
                    cv2.circle(canvas, eraser_point, 35, (0, 0, 0), -1)
                    previous_point = None
                elif drawing_gesture:
                    mode = "DRAWING"
                    current_point = smooth_point
                    cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, current_point, color, 7)
                else:
                    previous_point = None
            else:
                previous_point = None

            display = frame.copy()
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            blended = cv2.addWeighted(frame, 0.35, canvas, 0.65, 0)
            display[mask] = blended[mask]
            cv2.rectangle(display, (0, 0), (430, 52), (35, 35, 35), -1)
            cv2.putText(display, f"{mode} | COLOR: {color_name}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
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
            elif key == ord("e"):
                manual_eraser = not manual_eraser
                previous_point = None
            elif key == ord("c"):
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
