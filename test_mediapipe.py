"""
Test if MediaPipe is working correctly
"""
import sys

print("Testing MediaPipe installation...")
print(f"Python version: {sys.version}")

try:
    import mediapipe as mp
    print(f"✓ MediaPipe imported successfully")
    print(f"  Version: {mp.__version__}")
except ImportError as e:
    print(f"✗ Failed to import mediapipe: {e}")
    print("\nFix: pip install mediapipe==0.10.9")
    sys.exit(1)

try:
    face_mesh = mp.solutions.face_mesh
    print(f"✓ Face Mesh solution available")
except AttributeError as e:
    print(f"✗ Face Mesh not available: {e}")
    print("\nFix: pip uninstall mediapipe -y && pip install mediapipe==0.10.9")
    sys.exit(1)

try:
    drawing_utils = mp.solutions.drawing_utils
    print(f"✓ Drawing utilities available")
except AttributeError as e:
    print(f"✗ Drawing utilities not available: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV imported successfully")
    print(f"  Version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ Failed to import OpenCV: {e}")
    print("\nFix: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy imported successfully")
    print(f"  Version: {np.__version__}")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")
    print("\nFix: pip install numpy")
    sys.exit(1)

try:
    import pyautogui
    print(f"✓ PyAutoGUI imported successfully")
    print(f"  Version: {pyautogui.__version__}")
except ImportError as e:
    print(f"✗ Failed to import PyAutoGUI: {e}")
    print("\nFix: pip install pyautogui")
    sys.exit(1)

print("\n" + "="*50)
print("All dependencies installed correctly!")
print("="*50)

# Test MediaPipe Face Mesh initialization
print("\nTesting Face Mesh initialization...")
try:
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        print("✓ Face Mesh initialized successfully")
        print("\nYou're ready to run the calibration tool!")
except Exception as e:
    print(f"✗ Face Mesh initialization failed: {e}")
    sys.exit(1)