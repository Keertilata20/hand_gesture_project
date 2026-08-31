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
- Neon air-graffiti trails
- Gesture-based erasing
- Movement-driven music-energy visualization
- Keyboard color-selection fallbacks
- Canvas clearing, saving, undo, and clean shutdown controls

## Project Versions

### Version 1 — Air Drawing

`air_drawing.py`

Tracks the index fingertip and draws its movement on the screen.

### Version 2 — Gesture Colors (Legacy)

`air_drawing_v2.py`

Adds gesture-based color selection and improved tracking stability.

Legacy V2 controls:

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

V3 controls:

- Index finger up with middle finger folded, then pinch index finger and thumb — draw
- Point at a color or eraser button in the left-side palette and hold for about half a second to select it
- Keyboard color selection remains available when tracking is weak
- `E` — toggle manual eraser
- `D` — toggle free-draw mode when pinching is difficult
- `B` / `G` / `R` / `Y` / `P` — select blue / green / red / yellow / purple
- `U` — undo the last drawing, erasing, or clear action
- `S` — save the drawing as a PNG in the project folder
- `C` — clear the canvas
- `H` — show or hide the help panel
- `Q` — quit

The V3 application is the recommended version for demonstrations and daily use.

V3 currently provides a visual music layer: faster hand movement increases the energy of the animated visualizer. Audio generation can be added as a later phase once the interaction is stable. The camera feed is used without software enhancement.

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

Run the legacy gesture-color version:

```powershell
python air_drawing_v2.py
```

Run the latest version with gesture erasing, undo, and saving:

```powershell
python air_drawing_v3.py
```

V3 opens fullscreen to maximize the drawing area. Press `H` to show or hide the help panel, and press `Q` while the camera view is focused to exit.

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
- Move the hand closer until it fills a reasonable part of the camera view
- Use the V3 status bar to confirm that the camera is running and the hand is detected

### MediaPipe installation problems

Use the project virtual environment and run commands with `python -m pip`. Keeping project dependencies inside `.venv` prevents conflicts with unrelated global packages.

## Roadmap

- [x] Webcam hand-tracking test
- [x] Basic air drawing
- [x] Gesture-based color selection
- [x] Gesture-based erasing
- [x] Improve gesture reliability and drawing usability
- [x] Add an on-screen user interface
- [x] Add saving and exporting drawings
- [x] Add undo support
- [ ] Create a polished demo for sharing

## Why This Project?

This project is a practical introduction to computer vision and human-computer interaction. It explores how camera input, hand landmarks, gesture recognition, and real-time graphics can work together to create a natural interface.

## License

This project is intended for learning and experimentation. A formal license can be added when the project is ready for public reuse.
