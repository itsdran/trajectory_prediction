import cv2
import numpy as np
import math
from collections import deque

class ProjectileTracker:
    def __init__(self, box_size=50):
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide path to video file
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=False)
        self.box_size = box_size  # Size of the landing boxes
        self.reset()
        
    def reset(self):
        self.trajectory_points = deque(maxlen=10)  # Store recent trajectory points
        self.predicted_landing = None
        self.actual_landing = None
        self.object_stopped = False
        self.tracking_active = False
        self.object_bbox = None
        print("Tracking reset")

    def detect_moving_object(self, frame):
        # Apply background subtraction
        fgmask = self.fgbg.apply(frame)
        
        # Apply some morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        
        # Create mask for moving object visualization
        moving_object_mask = np.zeros_like(fgmask)
        
        if filtered_contours:
            # Use the largest contour (assuming it's the projectile)
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw the contour on the mask
            cv2.drawContours(moving_object_mask, [largest_contour], 0, 255, -1)
            
            # Get center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            
            return (x, y, w, h), (center_x, center_y), moving_object_mask
        
        return None, None, moving_object_mask

    def predict_trajectory(self, points):
        if len(points) < 5:  # Need enough points for prediction
            return None, None
        
        # Extract x, y coordinates
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Use polynomial regression to fit trajectory
        # For x-coordinate: assume constant velocity
        x_velocity = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
        
        # For y-coordinate: use quadratic fit (y = at² + bt + c)
        # Convert list indices to time values
        t_vals = list(range(len(y_vals)))
        
        # Fit quadratic polynomial for y trajectory
        if len(t_vals) >= 3:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(t_vals, y_vals, 2)
            a, b, c = coeffs
            
            # Predict future trajectory points
            future_points = []
            frame_height, frame_width = self.frame_size
            
            max_predict_frames = 5
            
            for i in range(1, max_predict_frames + 1):
                future_t = len(t_vals) - 1 + i
                future_x = x_vals[-1] + x_velocity * i
                future_y = a * (future_t ** 2) + b * future_t + c
                
                if 0 <= future_x < frame_width and 0 <= future_y < frame_height:
                    future_points.append((int(future_x), int(future_y)))
            
            # Last point is our predicted landing position
            if future_points:
                return future_points, future_points[-1]
            
        return None, None

    def check_object_stopped(self, current_center):
        if len(self.trajectory_points) < 10:
            return False
            
        # Check if the object has not moved significantly in recent frames
        recent_points = list(self.trajectory_points)[-5:]
        total_displacement = 0
        
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i-1][0]
            dy = recent_points[i][1] - recent_points[i-1][1]
            total_displacement += math.sqrt(dx**2 + dy**2)
        
        # If average displacement per frame is very small, consider the object stopped
        return total_displacement / 4 < 1.0  # Threshold for considering as stopped
        
    def check_boxes_intersect(self, point1, point2, box_size):
        """Check if two boxes centered at point1 and point2 with width/height of box_size intersect"""
        # Calculate box boundaries
        x1_min, y1_min = point1[0] - box_size, point1[1] - box_size
        x1_max, y1_max = point1[0] + box_size, point1[1] + box_size
        
        x2_min, y2_min = point2[0] - box_size, point2[1] - box_size
        x2_max, y2_max = point2[0] + box_size, point2[1] + box_size
        
        # Check for intersection
        x_overlap = (x1_min <= x2_max) and (x2_min <= x1_max)
        y_overlap = (y1_min <= y2_max) and (y2_min <= y1_max)
        
        return x_overlap and y_overlap
        
    def run(self):
        ret, frame = self.cap.read()            
        self.frame_size = frame.shape[:2]
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect moving object
            bbox, center, _ = self.detect_moving_object(frame)
            
            if bbox and center:
                x, y, w, h = bbox
                
                # Draw bounding box around the moving object
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Start tracking if not already tracking
                if not self.tracking_active:
                    self.tracking_active = True
                    self.object_stopped = False
                    self.trajectory_points.clear()
                
                # Add current center to trajectory
                self.trajectory_points.append(center)
                
                # Draw past trajectory (GREEN)
                points = list(self.trajectory_points)
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
                
                # Check if object has stopped
                if self.check_object_stopped(center) and not self.object_stopped:
                    self.object_stopped = True
                    self.actual_landing = center
                    print(f"Object landed at: {self.actual_landing}")
                
                # If object is still moving, predict trajectory
                if not self.object_stopped:
                    future_points, predicted_landing = self.predict_trajectory(points)
                    
                    if future_points and predicted_landing:
                        # Update the predicted landing point
                        self.predicted_landing = predicted_landing
                        
                        # Draw predicted trajectory (BLUE)
                        for i in range(1, len(future_points)):
                            cv2.line(frame, future_points[i-1], future_points[i], (255, 0, 0), 2)
            
            # Always draw predicted landing point (WHITE) if it exists, even after tracking stops
            if self.predicted_landing:
                px, py = self.predicted_landing
                s = self.box_size
                cv2.rectangle(frame, (px-s, py-s), (px+s, py+s), (255, 255, 255), 2)
                
                # Add a label for the predicted landing point
                cv2.putText(frame, "Predicted Landing Point", (px-s, py-s-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            
            # Always draw actual landing point (YELLOW) if it exists, even after tracking stops
            if self.actual_landing:
                ax, ay = self.actual_landing
                s = self.box_size  
                cv2.rectangle(frame, (ax-s, ay-s), (ax+s, ay+s), (255, 255, 0), 2)
                
                # Add a label for the actual landing point
                cv2.putText(frame, "Actual Landing Point", (ax-s, ay-s-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
                
                # Compare predicted and actual landing points if both exist
                if self.predicted_landing:
                    # Check if boxes intersect instead of using pixel distance
                    boxes_intersect = self.check_boxes_intersect(self.actual_landing, 
                                                                self.predicted_landing, 
                                                                self.box_size)
                    
                    match_text = f"Match: {boxes_intersect}"
                    cv2.putText(frame, match_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the combined frame
            cv2.imshow('Projectile Landing Point Prediction', frame)
            
            # Handle key presses
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ProjectileTracker(box_size=50)
    tracker.run()