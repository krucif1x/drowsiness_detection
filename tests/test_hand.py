# Test script for MediaPipeHandsWrapper
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import cv2
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.mediapipe.hand import MediaPipeHandsWrapper
from src.infrastructure.hardware.camera import Camera

def main():
    # Use unified Camera interface
    cam = Camera()  # auto-selects Picamera2 or OpenCV, uses env vars for config

    if not cam.ready:
        print("Camera not ready.")
        return

    hands_model = MediaPipeHandsWrapper()

    while True:
        frame = cam.read()  # always returns RGB
        if frame is None:
            continue

        hands_data = hands_model.infer(frame)
        colors = [(0, 255, 0), (255, 0, 0)]  # green, blue

        if hands_data:
            for hand_idx, hand in enumerate(hands_data):
                color = colors[hand_idx % len(colors)]
                for idx in range(21):  # MediaPipe Hands has 21 landmarks
                    lm = hands_model.get_landmark(hand, idx)
                    if lm:
                        h, w = frame.shape[:2]
                        x, y = int(lm[0] * w), int(lm[1] * h)
                        cv2.circle(frame, (x, y), 5, color, -1)
                        cv2.putText(frame, str(idx), (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                # Optionally, label the hand
                wrist = hands_model.get_landmark(hand, 0)
                if wrist:
                    x, y = int(wrist[0] * w), int(wrist[1] * h)
                    cv2.putText(frame, f"Hand {hand_idx+1}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert RGB to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Detection", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()