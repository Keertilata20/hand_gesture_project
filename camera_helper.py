# camera_helper.py
# Small helper to open the system webcam using common OpenCV backends.
# Returns (cap, used_backend) where cap is an opened cv2.VideoCapture or None.

import cv2
import platform
from typing import Tuple, Optional


def open_camera(index: int = 0, timeout_frames: int = 30) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    """
    Try to open the camera using several common backend options for the current OS.
    Returns (VideoCapture, backend_name) or (None, None) if none succeed.
    """
    system = platform.system()
    backends = []

    # Order: platform-specific backend first, then a generic attempt
    if system == "Windows":
        # CAP_DSHOW and CAP_MSMF tend to work reliably on many Windows setups
        backends = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("DEFAULT", None)]
    elif system == "Darwin":  # macOS
        backends = [("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION), ("DEFAULT", None)]
    else:  # Linux / other
        backends = [("CAP_V4L2", cv2.CAP_V4L2), ("DEFAULT", None)]

    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            # Read a few frames to give the camera time to initialize
            ok = False
            for _ in range(timeout_frames):
                ok, _ = cap.read()
                if ok:
                    break
            if cap.isOpened() and ok:
                return cap, name
            cap.release()
        except Exception:
            # If a backend constant is not available or something else fails, try the next
            try:
                cap.release()
            except Exception:
                pass

    return None, None


if __name__ == "__main__":
    cap, backend = open_camera(0)
    if cap:
        print(f"Opened camera with backend: {backend}")
        # Show a small preview window
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("camera_helper preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Could not open camera with any tested backend. Try a different index (1, 2) or check camera permissions.")
