import math
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from camera_helper import open_camera


COLORS = {
    "P": ((255, 0, 255), "PURPLE"),
    0: ((0, 0, 255), "RED"),
    2: ((255, 0, 0), "BLUE"),
    3: ((0, 255, 0), "GREEN"),
    4: ((0, 255, 255), "YELLOW"),
}


def raised_fingers(hand):
    points = mp.solutions.hands.HandLandmark

    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    pairs = [
        (points.INDEX_FINGER_TIP, points.INDEX_FINGER_PIP, points.INDEX_FINGER_MCP),
        (points.MIDDLE_FINGER_TIP, points.MIDDLE_FINGER_PIP, points.MIDDLE_FINGER_MCP),
        (points.RING_FINGER_TIP, points.RING_FINGER_PIP, points.RING_FINGER_MCP),
        (points.PINKY_TIP, points.PINKY_PIP, points.PINKY_MCP),
    ]
    return sum(
        distance(hand.landmark[tip], hand.landmark[mcp])
        > distance(hand.landmark[pip], hand.landmark[mcp]) * 1.15
        for tip, pip, mcp in pairs
    )


def main():
    camera, backend = open_camera(0)
    if camera is None:
        raise RuntimeError("Could not open webcam. Try changing 0 to 1 or check camera permissions.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    drawer = mp.solutions.drawing_utils
    canvas = None
    previous_point = None
    smooth_point = None
    color = COLORS["P"][0]
    color_name = COLORS["P"][1]
    candidate_count = None
    candidate_frames = 0
    fist_frames = 0
    manual_eraser = False
    history = []
    max_history = 20
    last_action = None
    show_help = False
    window_name = "Air drawing V3 - gesture eraser"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Air drawing V3 started.")
    print("Index only: draw | 2/3/4 fingers: choose blue/green/yellow")
    print("Closed fist held briefly: erase | E: toggle eraser")
    print("B/G/R/Y/P: colors | U: undo | S: save | C: clear | H: help | Q: quit")
    print(f"Camera backend: {backend}")

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

                if count == 0:
                    fist_frames += 1
                else:
                    fist_frames = 0

                if candidate_frames >= 8 and count in COLORS:
                    color, color_name = COLORS[count]

                points = mp.solutions.hands.HandLandmark
                index_tip = hand.landmark[points.INDEX_FINGER_TIP]
                height, width = frame.shape[:2]
                raw_point = (int(index_tip.x * width), int(index_tip.y * height))
                smooth_point = raw_point if smooth_point is None else (
                    int(0.65 * smooth_point[0] + 0.35 * raw_point[0]),
                    int(0.65 * smooth_point[1] + 0.35 * raw_point[1]),
                )

                # A closed fist held briefly activates the gesture eraser.
                # E remains available as a manual fallback.
                gesture_eraser = fist_frames >= 8
                selecting_color = count in (2, 3, 4)
                holding_fist = count == 0 and not gesture_eraser

                if manual_eraser or gesture_eraser:
                    mode = "ERASER"
                    if last_action != "ERASER":
                        history.append(canvas.copy())
                        history = history[-max_history:]
                    last_action = "ERASER"
                    palm = hand.landmark[points.MIDDLE_FINGER_MCP]
                    eraser_point = (int(palm.x * width), int(palm.y * height))
                    cv2.circle(frame, eraser_point, 35, (255, 255, 255), 2)
                    cv2.circle(canvas, eraser_point, 35, (0, 0, 0), -1)
                    previous_point = None
                elif selecting_color:
                    mode = f"SELECT {color_name if candidate_frames < 8 else COLORS[count][1]}"
                    previous_point = None
                    last_action = None
                elif holding_fist:
                    mode = f"HOLD FIST {fist_frames}/8"
                    previous_point = None
                    last_action = None
                elif count == 1:
                    mode = "DRAWING"
                    if last_action != "DRAWING":
                        history.append(canvas.copy())
                        history = history[-max_history:]
                    last_action = "DRAWING"
                    current_point = smooth_point
                    cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, current_point, color, 7)
                else:
                    mode = "READY"
                    previous_point = None
                    last_action = None
            else:
                previous_point = None
                last_action = None
                mode = "NO HAND"

            # Remember the fingertip so the next frame can connect a line.
            if current_point is not None:
                previous_point = current_point

            display = frame.copy()
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            blended = cv2.addWeighted(frame, 0.35, canvas, 0.65, 0)
            display[mask] = blended[mask]
            height, width = display.shape[:2]
            if show_help:
                panel_height = 150
                overlay = display.copy()
                cv2.rectangle(overlay, (0, height - panel_height),
                              (min(width, 650), height), (35, 35, 35), -1)
                display = cv2.addWeighted(overlay, 0.82, display, 0.18, 0)
                controls = [
                    "INDEX: DRAW    2/3/4: COLOR",
                    "FIST: ERASE    E: ERASE TOGGLE",
                    "B/G/R/Y/P: COLOR   U: UNDO",
                    "S: SAVE   C: CLEAR   H: HIDE HELP   Q: QUIT",
                ]
                for row, help_text in enumerate(controls):
                    cv2.putText(display, help_text, (15, height - 120 + row * 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            else:
                status = f"{mode} | {color_name} | H: HELP | Q: QUIT"
                cv2.rectangle(display, (0, height - 34), (min(width, 430), height),
                              (25, 25, 25), -1)
                cv2.putText(display, status, (10, height - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
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
            elif key == ord("p"):
                color, color_name = COLORS["P"]
            elif key == ord("e"):
                manual_eraser = not manual_eraser
                previous_point = None
                last_action = None
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("u"):
                if history:
                    canvas[:] = history.pop()
                previous_point = None
                smooth_point = None
                last_action = None
            elif key == ord("s"):
                filename = f"drawing_{datetime.now():%Y%m%d_%H%M%S}.png"
                if cv2.imwrite(filename, canvas):
                    print(f"Saved {filename}")
            elif key == ord("c"):
                history.append(canvas.copy())
                history = history[-max_history:]
                canvas[:] = 0
                previous_point = None
                smooth_point = None
                last_action = None
            elif key == ord("q"):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
