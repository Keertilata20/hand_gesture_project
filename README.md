# Hand Gesture Drawing (OpenCV + MediaPipe)

A small real-time computer-vision project that uses a webcam to detect hand landmarks (via MediaPipe) and convert fingertip movement into digital drawings.

This repository contains a compact set of scripts demonstrating hand tracking and a gesture-controlled "air drawing" application.

Maintainer: @Keertilata20

---

## Contents

- `hand_tracking_test.py` — simple webcam demo that shows MediaPipe hand landmarks.
- `air_drawing.py` — basic air-drawing using the index fingertip; press `C` to clear and `Q` to quit.
- `air_drawing_v2.py` — improved drawing with smoothing and gesture-based color selection; keyboard fallbacks available.
- `requirements.txt` — Python dependencies (OpenCV, MediaPipe).

---

## Features

- Real-time hand landmark detection with MediaPipe
- Index-fingertip tracking and smoothing for stable lines
- Persistent drawing canvas blended with camera feed
- Gesture-based color selection (in `air_drawing_v2.py`) plus keyboard shortcuts
- Clear canvas and clean shutdown controls

---

## Scripts & Controls

### hand_tracking_test.py

Opens the webcam and draws MediaPipe landmarks (up to 2 hands). Use this to verify that the camera and MediaPipe are working.

Run:

```bash
python hand_tracking_test.py
```

Press `Q` in the video window to quit.


### air_drawing.py (Basic air drawing)

Tracks the index fingertip and draws while you hold the pointing gesture (index finger up, middle finger down). Useful as a minimal demo.

Run:

```bash
python air_drawing.py
```

Controls:

- Point with index finger (index up, middle down) — draw
- `C` — clear canvas
- `Q` — quit


### air_drawing_v2.py (Gesture colors)

Adds gesture-based color selection and improved smoothing. Gesture selection requires a consistent gesture for several frames to avoid accidental switches.

Run:

```bash
python air_drawing_v2.py
```

Gesture color mapping (requires holding the gesture for several frames):

- 1 finger (index) — draw (no color change)
- 2 fingers — blue
- 3 fingers — green
- 4 fingers — yellow
- Closed fist — red

Keyboard fallbacks:

- `B` — blue
- `G` — green
- `R` — red
- `Y` — yellow
- `C` — clear canvas
- `Q` — quit

---

## Requirements

- Python 3.8+ (3.10 recommended)
- A working webcam
- Platforms: Windows, macOS, Linux

Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

requirements.txt currently contains:

```
opencv-python
mediapipe
```

Notes:

- On Windows the scripts open the camera with the DirectShow flag (`cv2.VideoCapture(0, cv2.CAP_DSHOW)`). If your webcam does not open, try changing the camera index (0 -> 1) or remove the `cv2.CAP_DSHOW` flag.

---

## Troubleshooting

- Webcam does not open:
  - Close other apps that may use the camera (Camera, Zoom, Teams, etc.).
  - Try a different camera index in the scripts (replace the 0 with 1).
  - On some systems the `cv2.CAP_DSHOW` flag can help (Windows). On others it may be unnecessary.

- Hand detection is unstable:
  - Face a light source (avoid strong backlighting).
  - Keep your hand roughly 40–70 cm from the camera.
  - Use a clear background and keep the hand inside the camera frame.

- MediaPipe installation issues:
  - Use a virtual environment and install with `python -m pip install -r requirements.txt` to avoid package conflicts.

---

## Roadmap / Ideas

- Improve gesture reliability and drawing usability
- Add an eraser gesture and undo
- Add a simple on-screen gesture-controlled UI
- Save/export drawings to image files
- Create a polished demo and packaging for easier sharing

---

## License

This project is provided for learning and experimentation. No formal license is specified in the repository — add a LICENSE file (MIT, Apache-2.0, etc.) if you want to publish or share the code for reuse.
