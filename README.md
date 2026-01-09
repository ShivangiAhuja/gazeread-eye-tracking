# Eye-Gaze Controlled PDF Scrolling System

## Overview
A hands-free PDF scrolling system using webcam-based eye tracking. Designed for students to maintain focus while studying without manual scrolling interruptions.

## Features
- ✅ Real-time eye gaze detection using MediaPipe
- ✅ Automatic scrolling based on sustained gaze direction
- ✅ Double-blink pause/resume functionality
- ✅ No special hardware required (just a webcam)
- ✅ Personal calibration tool included
- ✅ Smooth scrolling with accidental trigger prevention

## System Requirements
- Python 3.7+
- Webcam (built-in or USB)
- 4GB RAM minimum
- Windows, Mac, or Linux

## Installation

### 1. Clone or Download
```bash
git clone <your-repo-url>
cd EyeGazeScroller
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python mediapipe pyautogui numpy
```

### 3. Test Webcam
```bash
python test_webcam.py
```

## Usage

### Quick Start
1. Open your PDF in any viewer (Chrome, Adobe, Preview)
2. Run the application:
   ```bash
   python eye_gaze_scroller.py
   ```
3. Follow on-screen instructions
4. Position yourself 50-70cm from webcam
5. Focus on your PDF and read naturally

### Controls
- **Look Up (hold 1s):** Scroll up
- **Look Down (hold 1s):** Scroll down
- **Double-blink:** Pause/Resume system
- **Press 'Q':** Quit application

### Calibration (Recommended)
For best results, calibrate the system to your eye movement:

```bash
python calibration_tool.py
```

Follow the 3-step process:
1. Look straight at camera → Press SPACE → Hold 3 seconds
2. Look up naturally → Press SPACE → Hold 3 seconds
3. Look down naturally → Press SPACE → Hold 3 seconds

Apply the recommended threshold values to `eye_gaze_scroller.py`.

## Configuration

Edit `eye_gaze_scroller.py` to adjust:

```python
class Config:
    # Sensitivity
    LOOK_UP_THRESHOLD = 0.699    # Lower = more sensitive up detection
    LOOK_DOWN_THRESHOLD = 0.642    # Higher = more sensitive down detection
    
    # Timing
    SUSTAINED_GAZE_FRAMES = 30    # Frames to hold gaze (30 ≈ 1 second)
    
    # Scrolling
    SCROLL_AMOUNT = 3             # Pixels per scroll event
    SCROLL_COOLDOWN = 0.1         # Seconds between scrolls
    
    # Blink detection
    EAR_THRESHOLD = 0.21          # Eye aspect ratio for blink
```

## Troubleshooting

### Face Not Detected
- Improve lighting (face should be well-lit)
- Move closer to webcam (50-70cm optimal)
- Ensure face is fully in frame
- Check camera permissions

### Scrolling Too Sensitive
```python
SUSTAINED_GAZE_FRAMES = 45  # Increase from 30
```

### Scrolling Not Working
- Ensure PDF window is focused (click on it)
- Check if system is paused (look for red "PAUSED" text)
- Run calibration tool for your personal thresholds

### High CPU Usage
- Close unnecessary applications
- Reduce frame processing (edit code to skip every other frame)

## Project Structure
```
EyeGazeScroller/
├── eye_gaze_scroller.py    # Main application
├── calibration_tool.py      # Personal calibration
├── test_webcam.py           # Webcam test utility
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Technical Details

### Architecture
- **Input:** Webcam video stream (OpenCV)
- **Processing:** MediaPipe Face Mesh with iris tracking
- **Algorithm:** Vertical gaze ratio calculation
- **Output:** PyAutoGUI system-level scroll events

### Gaze Detection Algorithm
```
vertical_ratio = (iris_y - eye_top_y) / (eye_bottom_y - eye_top_y)

- ratio < 0.35: Looking up
- ratio > 0.65: Looking down
- 0.35-0.65: Center (no scroll)
```

### Performance
- CPU Usage: ~10-20%
- Memory: ~500MB
- Latency: <100ms (detection to scroll)
- Frame Rate: 30 FPS

## Limitations
- Requires consistent lighting
- Works best with minimal head movement
- PDF must be the active window
- Calibration needed per user
- Not suitable for rapid skim-reading

## Future Enhancements
- [ ] Automatic calibration on first run
- [ ] Speed control based on gaze intensity
- [ ] Multi-direction scrolling (left/right)
- [ ] Direct PDF reader integration
- [ ] Machine learning-based gaze estimation
- [ ] Reading pattern analysis

## Author
[Your Name]  
[Your University/Institution]  
[Date]

## License
MIT License - Feel free to use and modify for educational purposes.

## Acknowledgments
- MediaPipe by Google for face mesh detection
- OpenCV community for computer vision tools
- Inspiration from accessibility technology research

## Contact
For questions or issues: [Your Email]

---


**Star this project if it helped your studies! 📚👁️**
