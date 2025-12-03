import os
import cv2
import time
import logging
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.infrastructure.hardware.camera import Camera
from src.mediapipe.body_posture import MediapipeUpperBodyModel
from src.utils.constants import UPPER_BODY_LANDMARKS, UpperBodyIdx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("UpperBodyDemo")

def draw_landmarks_bgr(frame_bgr, landmarks_dict, color=(0, 255, 0)):
    h, w = frame_bgr.shape[:2]
    for idx, (x, y, z, vis) in landmarks_dict.items():
        # Convert normalized coords to pixel coords
        px = int(x * w)
        py = int(y * h)
        # Visibility gating: only draw if visible enough
        if vis is None or vis < 0.3:
            continue
        cv2.circle(frame_bgr, (px, py), 4, color, -1)
        cv2.putText(frame_bgr, str(idx), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    # Suggest a reasonable resolution for RaspiCam v3
    os.environ.setdefault("DS_CAMERA_SOURCE", "auto")      # auto: picamera2 → opencv
    os.environ.setdefault("DS_CAMERA_RES", "1280x720")     # adjust to 640x480 if needed for performance

    cam = Camera(source=os.getenv("DS_CAMERA_SOURCE", "auto"),
                 resolution=tuple(map(int, os.getenv("DS_CAMERA_RES", "1280x720").split("x"))))

    if not cam.ready:
        log.error("Camera not ready.")
        return

    model = MediapipeUpperBodyModel()
    fps_last = time.time()
    fps_val = 0

    log.info("Press 'q' or ESC to quit.")
    while True:
        frame_rgb = cam.read()
        if frame_rgb is None:
            # brief backoff if no frame
            time.sleep(0.01)
            continue

        # Mediapipe expects RGB; our Camera.read() already returns RGB
        landmarks = model.inference(frame_rgb, preprocessed=True)

        # Convert to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if landmarks:
            # Draw upper-body landmarks
            draw_landmarks_bgr(frame_bgr, landmarks, color=(0, 255, 0))
        else:
            cv2.putText(frame_bgr, "No pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # FPS update
        now = time.time()
        if now - fps_last >= 0.5:
            fps_val = int(1.0 / max(1e-6, now - fps_last))
            fps_last = now
        cv2.putText(frame_bgr, f"FPS: {fps_val}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Show backend info
        cv2.putText(frame_bgr, f"Backend: {cam.backend} @ {cam.resolution[0]}x{cam.resolution[1]}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("Upper-Body Pose Demo", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()