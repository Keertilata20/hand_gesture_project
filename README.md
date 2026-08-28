# Hand Gesture Drawing with OpenCV and MediaPipe

A real-time computer-vision project that uses a webcam to detect hand landmarks and turn fingertip movement into digital drawings.

This project began as a simple webcam hand-tracking experiment and is being developed step by step into a gesture-controlled drawing application.

## Current Features

- Real-time webcam input using OpenCV
- Hand landmark detection using MediaPipe
- Index-fingertip tracking
- Air drawing on a persistent canvas
- Smoothed fingertip movement for more stable lines
- Gesture-based color selection
- Keyboard color-selection fallbacks
- Canvas clearing and clean shutdown controls

## Project Versions

### Version 1 — Air Drawing

`air_drawing.py`

Tracks the index fingertip and draws its movement on the screen.

### Version 2 — Gesture Colors

`air_drawing_v2.py`

Adds gesture-based color selection and improved tracking stability.

Current controls:

- Index finger raised, middle finger lowered — draw
- Two fingers — select blue
- Three fingers — select green
- Four fingers — select yellow
- Closed fist — select red
- `B` — blue
- `G` — green
- `R` — red
- `Y` — yellow
- `C` — clear the canvas
- `Q` — quit

### Webcam Hand-Tracking Test

`hand_tracking_test.py`

Opens the webcam, detects hands, and displays MediaPipe landmarks without drawing.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 recommended
- A working webcam
- OpenCV
- MediaPipe

## Setup

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Project

Run the basic hand-tracking test:

```powershell
python hand_tracking_test.py
```

Run the original air-drawing version:

```powershell
python air_drawing.py
```

Run the gesture-color version:

```powershell
python air_drawing_v2.py
```

Press `Q` while the camera window is focused to exit.

## Troubleshooting

### Webcam does not open

Close other applications that may be using the webcam, such as Camera, Zoom, Teams, or WhatsApp. If necessary, change the camera index in the script:

```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

Try `1` instead of `0` if the computer has multiple cameras.

### Hand detection is unstable

Try the following:

- Face a light source instead of sitting with the light behind you
- Keep your hand 40–70 cm from the camera
- Use a clear, uncluttered background
- Keep your hand inside the visible camera frame

### MediaPipe installation problems

Use the project virtual environment and run commands with `python -m pip`. Keeping project dependencies inside `.venv` prevents conflicts with unrelated global packages.

### Cross-platform camera helper

On Windows the scripts currently attempt to open the camera with the DirectShow flag (`cv2.CAP_DSHOW`). If the camera does not open, try changing the camera index (0 -> 1) or removing the backend flag.

I added `camera_helper.py` — a small cross-platform helper that tries common OpenCV backends (DirectShow/MSMF on Windows, AVFoundation on macOS, V4L2 on Linux) and falls back to the default `cv2.VideoCapture`.

Example usage:

```python
from camera_helper import open_camera

cap, backend = open_camera(0)
if cap is None:
    raise RuntimeError("Could not open camera — try a different index or check permissions.")
print("Using backend:", backend)

# Then use cap as you would a normal cv2.VideoCapture:
ok, frame = cap.read()
...
cap.release()
```

Notes:

- The helper tries sensible backends depending on the OS and reads a few frames to confirm the camera is functional.
- If you want me to update the drawing scripts to use `camera_helper.open_camera(...)`, I've already applied that change in the repository.

## Roadmap

- [x] Webcam hand-tracking test
- [x] Basic air drawing
- [x] Gesture-based color selection
- [ ] Improve gesture reliability and drawing usability
- [ ] Add an eraser gesture
- [ ] Add a gesture-controlled user interface
- [ ] Add saving and exporting drawings
- [ ] Create a polished demo for sharing

## Why This Project?

This project is a practical introduction to computer vision and human-computer interaction. It explores how camera input, hand landmarks, gesture recognition, and real-time graphics can work together to create a simple, fun input modality.

## License

This project is intended for learning and experimentation. A formal license can be added when the project is ready for public reuse.
