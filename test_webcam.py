"""
Simple webcam test
Press 'q' to quit
"""
import cv2

print("Testing webcam...")
print("Press 'q' in the video window to quit")

# Try to open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam!")
    print("Trying alternative camera index...")
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: No webcam found!")
        print("Please check:")
        print("  1. Webcam is connected")
        print("  2. No other app is using it")
        print("  3. Camera permissions are granted")
        exit()

print("✓ Webcam opened successfully!")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Cannot read from webcam")
        break
    
    # Flip horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Show instructions
    cv2.putText(frame, "Webcam Working! Press 'q' to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    cv2.imshow('Webcam Test', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Test completed successfully!")