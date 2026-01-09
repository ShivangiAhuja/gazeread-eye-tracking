"""
Enhanced Eye-Gaze Controlled PDF Scrolling System with Analytics
Tracks usage metrics, performance, and generates reports
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Gaze thresholds
    LOOK_UP_THRESHOLD = 0.699
    LOOK_DOWN_THRESHOLD = 0.642
    
    # Timing
    SUSTAINED_GAZE_FRAMES = 30
    SMOOTHING_WINDOW = 5
    
    # Blink detection
    EAR_THRESHOLD = 0.21
    BLINK_FRAMES = 2
    DOUBLE_BLINK_WINDOW = 0.7
    
    # Scroll settings
    SCROLL_AMOUNT = 3
    SCROLL_COOLDOWN = 0.1
    
    # Analytics
    SAVE_ANALYTICS = True
    ANALYTICS_DIR = "analytics"

# ============================================================================
# ANALYTICS TRACKER
# ============================================================================
class AnalyticsTracker:
    """Track and save usage metrics for analysis"""
    
    def __init__(self):
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics
        self.total_scrolls_up = 0
        self.total_scrolls_down = 0
        self.total_blinks = 0
        self.total_pauses = 0
        self.pause_duration = 0
        self.last_pause_time = None
        
        # Performance metrics
        self.gaze_ratios = []
        self.ear_values = []
        self.fps_values = []
        
        # Time tracking
        self.active_time = 0
        self.pause_start = None
        
        # Create analytics directory
        if Config.SAVE_ANALYTICS:
            os.makedirs(Config.ANALYTICS_DIR, exist_ok=True)
    
    def record_scroll(self, direction):
        """Record a scroll event"""
        if direction == 'up':
            self.total_scrolls_up += 1
        elif direction == 'down':
            self.total_scrolls_down += 1
    
    def record_blink(self):
        """Record a blink"""
        self.total_blinks += 1
    
    def record_pause(self, is_paused):
        """Record pause/resume"""
        if is_paused and self.pause_start is None:
            self.pause_start = time.time()
            self.total_pauses += 1
        elif not is_paused and self.pause_start is not None:
            self.pause_duration += time.time() - self.pause_start
            self.pause_start = None
    
    def record_metrics(self, gaze_ratio, ear, fps):
        """Record performance metrics"""
        self.gaze_ratios.append(gaze_ratio)
        self.ear_values.append(ear)
        self.fps_values.append(fps)
    
    def get_session_duration(self):
        """Get total session duration"""
        return time.time() - self.session_start
    
    def get_summary(self):
        """Generate session summary"""
        duration = self.get_session_duration()
        active_time = duration - self.pause_duration
        
        summary = {
            "session_id": self.session_id,
            "duration_seconds": round(duration, 2),
            "active_time_seconds": round(active_time, 2),
            "pause_time_seconds": round(self.pause_duration, 2),
            "total_scrolls": self.total_scrolls_up + self.total_scrolls_down,
            "scrolls_up": self.total_scrolls_up,
            "scrolls_down": self.total_scrolls_down,
            "total_blinks": self.total_blinks,
            "total_pauses": self.total_pauses,
            "avg_gaze_ratio": round(np.mean(self.gaze_ratios), 3) if self.gaze_ratios else 0,
            "avg_fps": round(np.mean(self.fps_values), 1) if self.fps_values else 0,
            "scrolls_per_minute": round((self.total_scrolls_up + self.total_scrolls_down) / (active_time / 60), 2) if active_time > 0 else 0
        }
        
        return summary
    
    def save_session(self):
        """Save session data to JSON file"""
        if not Config.SAVE_ANALYTICS:
            return
        
        summary = self.get_summary()
        filename = os.path.join(Config.ANALYTICS_DIR, f"session_{self.session_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n📊 Analytics saved to: {filename}")
        return filename

# ============================================================================
# ENHANCED EYE TRACKER
# ============================================================================
class EyeTracker:
    def __init__(self, analytics_tracker=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks
        self.RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173]
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]
        self.LEFT_EYE = [362, 263, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        
        # Smoothing
        self.gaze_history = deque(maxlen=Config.SMOOTHING_WINDOW)
        self.up_gaze_counter = 0
        self.down_gaze_counter = 0
        
        # Blink detection
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_times = []
        
        # System state
        self.is_paused = False
        self.last_scroll_time = 0
        
        # Analytics
        self.analytics = analytics_tracker
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h == 0:
            return 0
        
        return (v1 + v2) / (2.0 * h)
    
    def get_eye_landmarks(self, landmarks, eye_indices, frame_width, frame_height):
        """Extract eye landmarks"""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            eye_points.append([x, y])
        return np.array(eye_points)
    
    def get_iris_position(self, landmarks, iris_indices, frame_width, frame_height):
        """Get iris center"""
        iris_points = []
        for idx in iris_indices:
            landmark = landmarks[idx]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            iris_points.append([x, y])
        return np.mean(iris_points, axis=0)
    
    def calculate_vertical_ratio(self, eye_landmarks, iris_center):
        """Calculate vertical gaze ratio"""
        eye_top = np.min(eye_landmarks[:, 1])
        eye_bottom = np.max(eye_landmarks[:, 1])
        eye_height = eye_bottom - eye_top
        
        if eye_height == 0:
            return 0.5
        
        iris_y = iris_center[1]
        ratio = (iris_y - eye_top) / eye_height
        return max(0, min(1, ratio))
    
    def detect_blink(self, ear):
        """Detect blinks"""
        if ear < Config.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= Config.BLINK_FRAMES:
                current_time = time.time()
                self.blink_times.append(current_time)
                self.total_blinks += 1
                
                if self.analytics:
                    self.analytics.record_blink()
                
                self.blink_times = [t for t in self.blink_times 
                                   if current_time - t < Config.DOUBLE_BLINK_WINDOW]
                
                if len(self.blink_times) >= 2:
                    self.is_paused = not self.is_paused
                    if self.analytics:
                        self.analytics.record_pause(self.is_paused)
                    self.blink_times.clear()
                    self.blink_counter = 0
                    return True
            
            self.blink_counter = 0
        
        return False
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        return 0
    
    def process_frame(self, frame):
        """Main processing function"""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return 'neutral', frame
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Extract features
        right_eye = self.get_eye_landmarks(landmarks, self.RIGHT_EYE, 
                                          frame_width, frame_height)
        left_eye = self.get_eye_landmarks(landmarks, self.LEFT_EYE, 
                                         frame_width, frame_height)
        right_iris = self.get_iris_position(landmarks, self.RIGHT_IRIS,
                                           frame_width, frame_height)
        left_iris = self.get_iris_position(landmarks, self.LEFT_IRIS,
                                          frame_width, frame_height)
        
        # Calculate metrics
        right_ratio = self.calculate_vertical_ratio(right_eye, right_iris)
        left_ratio = self.calculate_vertical_ratio(left_eye, left_iris)
        avg_ratio = (right_ratio + left_ratio) / 2
        
        self.gaze_history.append(avg_ratio)
        smoothed_ratio = np.mean(self.gaze_history)
        
        right_ear = self.calculate_ear(right_eye)
        left_ear = self.calculate_ear(left_eye)
        avg_ear = (right_ear + left_ear) / 2
        
        # Record analytics
        if self.analytics and not self.is_paused:
            self.analytics.record_metrics(smoothed_ratio, avg_ear, fps)
        
        # Detect blinks
        self.detect_blink(avg_ear)
        
        # Draw visualization
        self.draw_enhanced_visualization(frame, right_eye, left_eye, 
                                        right_iris, left_iris, 
                                        smoothed_ratio, avg_ear, fps)
        
        # Check if paused
        if self.is_paused:
            cv2.putText(frame, "PAUSED", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return 'paused', frame
        
        # Determine gaze direction
        if smoothed_ratio < Config.LOOK_UP_THRESHOLD:
            self.up_gaze_counter += 1
            self.down_gaze_counter = 0
            
            if self.up_gaze_counter >= Config.SUSTAINED_GAZE_FRAMES:
                return 'up', frame
        
        elif smoothed_ratio > Config.LOOK_DOWN_THRESHOLD:
            self.down_gaze_counter += 1
            self.up_gaze_counter = 0
            
            if self.down_gaze_counter >= Config.SUSTAINED_GAZE_FRAMES:
                return 'down', frame
        
        else:
            self.up_gaze_counter = 0
            self.down_gaze_counter = 0
        
        return 'neutral', frame
    
    def draw_enhanced_visualization(self, frame, right_eye, left_eye, 
                                   right_iris, left_iris, ratio, ear, fps):
        """Enhanced visualization with more info"""
        h, w = frame.shape[:2]
        
        # Draw eyes
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        
        # Draw iris
        cv2.circle(frame, tuple(right_iris.astype(int)), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_iris.astype(int)), 3, (255, 0, 0), -1)
        
        # Info panel background
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        # Display metrics
        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Gaze: {ratio:.3f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gaze direction
        y_offset += 25
        if ratio < Config.LOOK_UP_THRESHOLD:
            direction = "UP"
            color = (0, 255, 255)
            progress = self.up_gaze_counter / Config.SUSTAINED_GAZE_FRAMES
        elif ratio > Config.LOOK_DOWN_THRESHOLD:
            direction = "DOWN"
            color = (0, 255, 255)
            progress = self.down_gaze_counter / Config.SUSTAINED_GAZE_FRAMES
        else:
            direction = "CENTER"
            color = (255, 255, 255)
            progress = 0
        
        cv2.putText(frame, f"Dir: {direction}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Progress bar
        if progress > 0:
            y_offset += 25
            bar_width = int(progress * 260)
            cv2.rectangle(frame, (20, y_offset - 15), (20 + bar_width, y_offset - 5),
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (20, y_offset - 15), (280, y_offset - 5),
                         (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Double-blink: Pause | Q: Quit", 
                   (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ============================================================================
# SCROLL CONTROLLER
# ============================================================================
class ScrollController:
    def __init__(self, analytics_tracker=None):
        self.last_scroll_time = 0
        self.analytics = analytics_tracker
    
    def scroll(self, direction):
        """Execute scroll with analytics"""
        current_time = time.time()
        
        if current_time - self.last_scroll_time < Config.SCROLL_COOLDOWN:
            return
        
        if direction == 'up':
            pyautogui.scroll(Config.SCROLL_AMOUNT)
            print("↑ Scrolling UP")
        elif direction == 'down':
            pyautogui.scroll(-Config.SCROLL_AMOUNT)
            print("↓ Scrolling DOWN")
        
        if self.analytics:
            self.analytics.record_scroll(direction)
        
        self.last_scroll_time = current_time

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def print_session_summary(analytics):
    """Print beautiful session summary"""
    summary = analytics.get_summary()
    
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Session ID:        {summary['session_id']}")
    print(f"Total Duration:    {summary['duration_seconds']:.1f}s ({summary['duration_seconds']/60:.1f} min)")
    print(f"Active Time:       {summary['active_time_seconds']:.1f}s")
    print(f"Pause Time:        {summary['pause_time_seconds']:.1f}s")
    print(f"\nScrolling Stats:")
    print(f"  Total Scrolls:   {summary['total_scrolls']}")
    print(f"  Scrolls Up:      {summary['scrolls_up']}")
    print(f"  Scrolls Down:    {summary['scrolls_down']}")
    print(f"  Scrolls/Min:     {summary['scrolls_per_minute']}")
    print(f"\nInteraction:")
    print(f"  Total Blinks:    {summary['total_blinks']}")
    print(f"  Pause Count:     {summary['total_pauses']}")
    print(f"\nPerformance:")
    print(f"  Avg FPS:         {summary['avg_fps']}")
    print(f"  Avg Gaze Ratio:  {summary['avg_gaze_ratio']}")
    print("="*60 + "\n")

def main():
    print("=" * 60)
    print("Enhanced Eye-Gaze PDF Scrolling System")
    print("With Analytics & Performance Tracking")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Open your PDF in any viewer")
    print("2. Position yourself 50-70cm from webcam")
    print("3. Look UP/DOWN continuously to scroll")
    print("4. Double-blink to pause/resume")
    print("5. Press 'Q' to quit and see analytics")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize
    analytics = AnalyticsTracker()
    eye_tracker = EyeTracker(analytics_tracker=analytics)
    scroll_controller = ScrollController(analytics_tracker=analytics)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam!")
        return
    
    print("\n✓ System ready! Analytics tracking enabled.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gaze_direction, processed_frame = eye_tracker.process_frame(frame)
            
            if gaze_direction in ['up', 'down']:
                scroll_controller.scroll(gaze_direction)
            
            cv2.imshow('Eye Tracking - Press Q to Quit', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Show and save analytics
        print_session_summary(analytics)
        analytics.save_session()
        
        print("\n✓ System stopped. Thank you for using GazeRead!")

if __name__ == "__main__":

    main()
