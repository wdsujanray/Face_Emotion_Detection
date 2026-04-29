import cv2
import numpy as np

print("Testing camera access...")

# Try different camera indices
for i in range(3):
    print(f"\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ Camera {i} works!")
            print(f"  Frame shape: {frame.shape}")
            cap.release()
            break
        else:
            print(f"✗ Camera {i} opened but couldn't read frame")
            cap.release()
    else:
        print(f"✗ Camera {i} could not be opened")

# If no camera works, try with different backend
print("\nTrying with DSHOW backend...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✓ Camera works with DSHOW backend!")
    cap.release()
else:
    print("✗ DSHOW backend failed")

print("\nDone.")