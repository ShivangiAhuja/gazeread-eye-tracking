"""
Personal Calibration Tool for Eye Gaze System
This helps you find optimal threshold values for your setup
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class Calibrator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173]
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]
        self.LEFT_EYE = [362, 263, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        
        self.calibration_data = {
            'center': [],
            'up': [],
            'down': []
        }
        
        self.current_phase = 'center'
        self.collection_start = None
        self.collection_duration = 3  # seconds
    
    def get_vertical_ratio(self, landmarks, frame_width, frame_height):
        """Calculate vertical gaze ratio"""
        right_eye = []
        for idx in self.RIGHT_EYE:
            lm = landmarks[idx]
            right_eye.append([lm.x * frame_width, lm.y * frame_height])
        right_eye = np.array(right_eye)
        
        left_eye = []
        for idx in self.LEFT_EYE:
            lm = landmarks[idx]
            left_eye.append([lm.x * frame_width, lm.y * frame_height])
        left_eye = np.array(left_eye)
        
        right_iris = []
        for idx in self.RIGHT_IRIS:
            lm = landmarks[idx]
            right_iris.append([lm.x * frame_width, lm.y * frame_height])
        right_iris_center = np.mean(right_iris, axis=0)
        
        left_iris = []
        for idx in self.LEFT_IRIS:
            lm = landmarks[idx]
            left_iris.append([lm.x * frame_width, lm.y * frame_height])
        left_iris_center = np.mean(left_iris, axis=0)
        
        # Calculate ratios
        right_top = np.min(right_eye[:, 1])
        right_bottom = np.max(right_eye[:, 1])
        right_ratio = (right_iris_center[1] - right_top) / (right_bottom - right_top)
        
        left_top = np.min(left_eye[:, 1])
        left_bottom = np.max(left_eye[:, 1])
        left_ratio = (left_iris_center[1] - left_top) / (left_bottom - left_top)
        
        return (right_ratio + left_ratio) / 2
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam!")
            return None
        
        print("\n" + "="*60)
        print("PERSONAL CALIBRATION TOOL")
        print("="*60)
        print("\nThis will measure YOUR eye movement ranges.")
        print("Follow the on-screen instructions carefully.\n")
        
        phase_instructions = {
            'center': "Look STRAIGHT at the camera",
            'up': "Look UP naturally (as if reading top of screen)",
            'down': "Look DOWN naturally (as if reading bottom of screen)"
        }
        
        phases = ['center', 'up', 'down']
        phase_idx = 0
        
        while phase_idx < len(phases):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            self.current_phase = phases[phase_idx]
            
            # Draw instructions
            h, w = frame.shape[:2]
            instruction = phase_instructions[self.current_phase]
            
            # Background for text
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
            
            cv2.putText(frame, f"Step {phase_idx + 1}/3", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
            
            cv2.putText(frame, instruction,
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ratio = self.get_vertical_ratio(landmarks, w, h)
                
                # Show current ratio
                cv2.putText(frame, f"Current Ratio: {ratio:.3f}",
                           (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)
                
                # Collection countdown
                if self.collection_start is None:
                    cv2.putText(frame, "Press SPACE when ready",
                               (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                else:
                    elapsed = time.time() - self.collection_start
                    remaining = self.collection_duration - elapsed
                    
                    if remaining > 0:
                        # Collecting data
                        self.calibration_data[self.current_phase].append(ratio)
                        
                        cv2.putText(frame, f"HOLD POSITION: {remaining:.1f}s",
                                   (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.8, (0, 255, 0), 2)
                        
                        # Progress bar
                        bar_width = int((elapsed / self.collection_duration) * (w - 40))
                        cv2.rectangle(frame, (20, h - 20), (20 + bar_width, h - 10),
                                    (0, 255, 0), -1)
                    else:
                        # Done with this phase
                        phase_idx += 1
                        self.collection_start = None
                        
                        if phase_idx < len(phases):
                            print(f"✓ {self.current_phase.upper()} position recorded")
            else:
                cv2.putText(frame, "FACE NOT DETECTED!",
                           (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 0, 255), 2)
            
            cv2.imshow('Calibration Tool - Press Q to Quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == ord(' ') and self.collection_start is None:
                self.collection_start = time.time()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate thresholds
        return self.calculate_thresholds()
    
    def calculate_thresholds(self):
        """Calculate optimal thresholds from collected data"""
        center_avg = np.mean(self.calibration_data['center'])
        center_std = np.std(self.calibration_data['center'])
        
        up_avg = np.mean(self.calibration_data['up'])
        down_avg = np.mean(self.calibration_data['down'])
        
        # Calculate thresholds with buffer zones
        up_threshold = up_avg + 0.03  # Slightly above "up" average
        down_threshold = down_avg - 0.03  # Slightly below "down" average
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"\nYour measured values:")
        print(f"  Looking STRAIGHT: {center_avg:.3f} (±{center_std:.3f})")
        print(f"  Looking UP:       {up_avg:.3f}")
        print(f"  Looking DOWN:     {down_avg:.3f}")
        print(f"\nRecommended threshold values:")
        print(f"  LOOK_UP_THRESHOLD   = {up_threshold:.3f}")
        print(f"  LOOK_DOWN_THRESHOLD = {down_threshold:.3f}")
        print("\n" + "="*60)
        print("\nTo use these values:")
        print("1. Open eye_gaze_scroller.py")
        print("2. Find the Config class at the top")
        print("3. Update these lines:")
        print(f"   LOOK_UP_THRESHOLD = {up_threshold:.3f}")
        print(f"   LOOK_DOWN_THRESHOLD = {down_threshold:.3f}")
        print("4. Save and run the main program")
        print("="*60 + "\n")
        
        return {
            'up_threshold': up_threshold,
            'down_threshold': down_threshold,
            'center_avg': center_avg,
            'up_avg': up_avg,
            'down_avg': down_avg
        }

if __name__ == "__main__":
    calibrator = Calibrator()
    results = calibrator.run()
    
    if results:
        print("✓ Calibration complete!")
    else:
        print("✗ Calibration cancelled")