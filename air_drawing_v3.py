import math
from datetime import datetime
import time

import cv2
import mediapipe as mp
import numpy as np

from camera_helper import open_camera


COLORS = {
    "P": ((255, 0, 255), "PURPLE"),
    "R": ((0, 0, 255), "RED"),
    "B": ((255, 0, 0), "BLUE"),
    "G": ((0, 255, 0), "GREEN"),
    "Y": ((0, 255, 255), "YELLOW"),
}

TOOLBAR = ["R", "B", "G", "Y", "P", "E"]


def toolbar_hit(point):
    """Return the toolbar tool under a point, or None."""
    if point is None or point[0] > 105:
        return None
    button_height = 64
    gap = 7
    index = (point[1] - 10) // (button_height + gap)
    within_button = (point[1] - 10) % (button_height + gap) < button_height
    if within_button and 0 <= index < len(TOOLBAR):
        return TOOLBAR[index]
    return None


def draw_toolbar(image, selected_tool, hover_tool):
    """Draw a larger, narrow finger-selectable palette on the left."""
    button_width = 86
    button_height = 64
    gap = 7
    for index, tool in enumerate(TOOLBAR):
        x = 10
        y = 10 + index * (button_height + gap)
        if tool == "E":
            button_color = (70, 70, 70)
            label = "ERASE"
        else:
            button_color, label = COLORS[tool]
        thickness = 3 if tool == selected_tool else 1
        border = (255, 255, 255) if tool == hover_tool else button_color
        cv2.rectangle(image, (x, y), (x + button_width, y + button_height), border, thickness)
        cv2.rectangle(image, (x + 6, y + 6),
                      (x + button_width - 6, y + button_height - 6), button_color, -1)
        cv2.putText(image, label[0], (x + 32, y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_music_visualizer(image, energy, phase):
    """Draw a lightweight music-energy visualization driven by hand movement."""
    height, width = image.shape[:2]
    center = (width - 105, height - 82)
    base_radius = 28 + int(energy * 18)
    pulse = 0.5 + 0.5 * math.sin(phase)
    cv2.circle(image, center, base_radius + int(pulse * 8), (255, 0, 180), 2)
    cv2.circle(image, center, 5 + int(energy * 8), (255, 255, 255), -1)

    for index in range(16):
        angle = (2 * math.pi * index / 16) - math.pi / 2
        wave = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(phase * 1.7 + index * 0.8))
        bar = 8 + int(energy * 42 * wave)
        inner = (int(center[0] + math.cos(angle) * 42),
                 int(center[1] + math.sin(angle) * 42))
        outer = (int(center[0] + math.cos(angle) * (42 + bar)),
                 int(center[1] + math.sin(angle) * (42 + bar)))
        cv2.line(image, inner, outer, (255, 0, 180), 3)


def drawing_pose(hand):
    """Detect the simple, reliable drawing pose: index up, middle down."""
    points = mp.solutions.hands.HandLandmark
    index_up = hand.landmark[points.INDEX_FINGER_TIP].y < hand.landmark[points.INDEX_FINGER_PIP].y
    middle_down = hand.landmark[points.MIDDLE_FINGER_TIP].y > hand.landmark[points.MIDDLE_FINGER_PIP].y
    return index_up and middle_down


def pinching(hand):
    """Return True when the index fingertip and thumb are close together."""
    points = mp.solutions.hands.HandLandmark

    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    thumb_tip = hand.landmark[points.THUMB_TIP]
    index_tip = hand.landmark[points.INDEX_FINGER_TIP]
    wrist = hand.landmark[points.WRIST]
    middle_mcp = hand.landmark[points.MIDDLE_FINGER_MCP]
    palm_size = distance(wrist, middle_mcp)
    return palm_size > 0 and distance(thumb_tip, index_tip) < palm_size * 0.42


def main():
    camera, backend = open_camera(0)
    if camera is None:
        raise RuntimeError("Could not open webcam. Try changing 0 to 1 or check camera permissions.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp.solutions.hands.Hands(
        # Re-detect every frame instead of relying on tracking between frames.
        # This is more tolerant of a soft or unstable webcam feed.
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    drawer = mp.solutions.drawing_utils
    canvas = None
    previous_point = None
    smooth_point = None
    color = COLORS["P"][0]
    color_name = COLORS["P"][1]
    manual_eraser = False
    selected_tool = "P"
    hover_tool = None
    hover_frames = 0
    free_draw = False
    history = []
    max_history = 20
    last_action = None
    show_help = False
    movement_energy = 0.0
    visualizer_phase = 0.0
    last_tick = time.monotonic()
    window_name = "Air drawing V3 - gesture eraser"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Air drawing V3 started.")
    print("Index up + middle folded, then pinch: draw | E: toggle eraser")
    print("D: toggle free draw | B/G/R/Y/P: choose color | U: undo")
    print("S: save | C: clear | H: help | Q: quit")
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
                points = mp.solutions.hands.HandLandmark
                index_tip = hand.landmark[points.INDEX_FINGER_TIP]
                height, width = frame.shape[:2]
                raw_point = (int(index_tip.x * width), int(index_tip.y * height))
                smooth_point = raw_point if smooth_point is None else (
                    int(0.65 * smooth_point[0] + 0.35 * raw_point[0]),
                    int(0.65 * smooth_point[1] + 0.35 * raw_point[1]),
                )

                tool_under_finger = toolbar_hit(smooth_point)
                if tool_under_finger is not None:
                    mode = f"SELECT {tool_under_finger} {min(hover_frames, 5)}/5"
                    previous_point = None
                    last_action = None
                    if tool_under_finger == hover_tool:
                        hover_frames += 1
                    else:
                        hover_tool = tool_under_finger
                        hover_frames = 1
                    if hover_frames >= 5:
                        selected_tool = tool_under_finger
                        if selected_tool == "E":
                            manual_eraser = True
                        else:
                            manual_eraser = False
                            color, color_name = COLORS[selected_tool]
                elif drawing_pose(hand):
                    hover_tool = None
                    hover_frames = 0
                    pinch = pinching(hand)
                    cursor_color = (0, 255, 0) if pinch else (255, 255, 255)
                    cv2.circle(frame, smooth_point, 12, cursor_color, 2)
                    can_draw = pinch or free_draw
                    if manual_eraser and can_draw:
                        mode = "ERASER"
                        if last_action != "ERASER":
                            history.append(canvas.copy())
                            history = history[-max_history:]
                        last_action = "ERASER"
                        eraser_point = smooth_point
                        cv2.circle(frame, eraser_point, 35, (255, 255, 255), 2)
                        cv2.circle(canvas, eraser_point, 35, (0, 0, 0), -1)
                        previous_point = None
                    elif can_draw:
                        mode = "DRAWING"
                        if last_action != "DRAWING":
                            history.append(canvas.copy())
                            history = history[-max_history:]
                        last_action = "DRAWING"
                        current_point = smooth_point
                        cv2.circle(frame, current_point, 10, (0, 255, 0), -1)
                        if previous_point is not None:
                            distance_moved = math.hypot(
                                current_point[0] - previous_point[0],
                                current_point[1] - previous_point[1],
                            )
                            movement_energy = min(
                                1.0, 0.8 * movement_energy + 0.2 * distance_moved / 35.0
                            )
                        if previous_point is not None:
                            cv2.line(canvas, previous_point, current_point, color, 7)
                    else:
                        mode = "PINCH TO DRAW"
                        previous_point = None
                        last_action = None
                elif manual_eraser:
                    mode = "READY - POINT TO ERASE"
                    previous_point = None
                    last_action = None
                    hover_tool = None
                    hover_frames = 0
                else:
                    hover_tool = None
                    hover_frames = 0
                    mode = "READY"
                    previous_point = None
                    last_action = None
            else:
                previous_point = None
                last_action = None
                mode = "NO HAND"
                hover_tool = None
                hover_frames = 0
                movement_energy *= 0.9

            # Remember the fingertip so the next frame can connect a line.
            if current_point is not None:
                previous_point = current_point

            now = time.monotonic()
            visualizer_phase += (now - last_tick) * (2.0 + movement_energy * 8.0)
            last_tick = now

            display = frame.copy()
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            glow = cv2.GaussianBlur(canvas, (0, 0), 18)
            glow_mask = cv2.cvtColor(glow, cv2.COLOR_BGR2GRAY) > 2
            glow_blended = cv2.addWeighted(frame, 0.35, glow, 0.9, 0)
            display[glow_mask] = glow_blended[glow_mask]
            core_blended = cv2.addWeighted(display, 0.25, canvas, 0.95, 0)
            display[mask] = core_blended[mask]
            draw_music_visualizer(display, movement_energy, visualizer_phase)
            draw_toolbar(display, selected_tool, hover_tool)
            height, width = display.shape[:2]
            if show_help:
                panel_height = 150
                overlay = display.copy()
                cv2.rectangle(overlay, (0, height - panel_height),
                              (min(width, 650), height), (35, 35, 35), -1)
                display = cv2.addWeighted(overlay, 0.82, display, 0.18, 0)
                controls = [
                    "INDEX UP + MIDDLE FOLDED, PINCH: DRAW",
                    "E: ERASE TOGGLE   D: FREE DRAW TOGGLE",
                    "B/G/R/Y/P: COLOR   U: UNDO",
                    "S: SAVE   C: CLEAR   H: HIDE HELP   Q: QUIT",
                ]
                for row, help_text in enumerate(controls):
                    cv2.putText(display, help_text, (15, height - 120 + row * 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            else:
                status = f"{mode} | {color_name} | ENERGY {int(movement_energy * 100):02d}% | H: HELP | Q: QUIT"
                cv2.rectangle(display, (0, height - 34), (min(width, 430), height),
                              (25, 25, 25), -1)
                cv2.putText(display, status, (10, height - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("b"):
                color, color_name = COLORS["B"]
                selected_tool, manual_eraser = "B", False
            elif key == ord("g"):
                color, color_name = COLORS["G"]
                selected_tool, manual_eraser = "G", False
            elif key == ord("r"):
                color, color_name = COLORS["R"]
                selected_tool, manual_eraser = "R", False
            elif key == ord("y"):
                color, color_name = COLORS["Y"]
                selected_tool, manual_eraser = "Y", False
            elif key == ord("p"):
                color, color_name = COLORS["P"]
                selected_tool, manual_eraser = "P", False
            elif key == ord("e"):
                manual_eraser = not manual_eraser
                selected_tool = "E" if manual_eraser else selected_tool
                previous_point = None
                last_action = None
            elif key == ord("d"):
                free_draw = not free_draw
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
                movement_energy = 0.0
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
                movement_energy = 0.0
            elif key == ord("q"):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
