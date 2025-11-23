# Test script for MediaPipeHandsWrapper
import cv2
from src.mediapipe.hand import MediaPipeHandsWrapper

def main():
    cap = cv2.VideoCapture(0)
    hands_model = MediaPipeHandsWrapper()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands_data = hands_model.infer(frame)
        if hands_data:
            # Example: Draw a circle on the index finger tip of the first hand
            index_tip = hands_model.get_landmark(hands_data[0], 8)  # 8 = INDEX_FINGER_TIP
            if index_tip:
                h, w = frame.shape[:2]
                x, y = int(index_tip[0] * w), int(index_tip[1] * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(frame, "Index Tip", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()