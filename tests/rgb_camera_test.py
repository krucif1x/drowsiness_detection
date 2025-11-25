"""
Test if your camera is outputting RGB or grayscale
"""
import cv2
import numpy as np

def test_camera():
    print("Testing camera output...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Try to force color mode
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        
        print(f"\nFrame {i}:")
        print(f"  Shape: {frame.shape}")
        print(f"  Dtype: {frame.dtype}")
        
        if len(frame.shape) == 2:
            print("  ❌ GRAYSCALE! Camera is outputting 1-channel images")
            print("  This is the problem! Your camera is in grayscale mode.")
        elif frame.shape[2] == 1:
            print("  ❌ GRAYSCALE! Camera is outputting (H, W, 1) images")
        elif frame.shape[2] == 3:
            print("  ✓ RGB/BGR: Camera is outputting 3-channel images")
            
            # Check if it's actually grayscale data in 3 channels
            if np.allclose(frame[:,:,0], frame[:,:,1]) and np.allclose(frame[:,:,1], frame[:,:,2]):
                print("  ⚠️ WARNING: 3 channels but all identical (fake RGB)")
            else:
                print("  ✓ True color image")
        
        if i == 0:
            # Save first frame for inspection
            cv2.imwrite("camera_test.jpg", frame)
            print(f"  Saved to camera_test.jpg")
        
        if i >= 2:
            break
    
    cap.release()
    
    print("\n" + "="*60)
    print("SOLUTION:")
    print("="*60)
    print("If camera is grayscale:")
    print("1. Check camera settings (might have IR mode enabled)")
    print("2. Try different camera index: cv2.VideoCapture(1)")
    print("3. Force RGB in your code:")
    print("   cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)")
    print("4. Or convert in your image validator")

if __name__ == "__main__":
    test_camera()