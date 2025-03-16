# Application of Kalman Filter to predict the land point of a trajectory
import cv2
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class TrajectoryPredictor:
    def __init__(self, 
                 history_size=20, 
                 min_object_size=500, 
                 landing_box_size=50, 
                 near_miss_threshold=25,
                 bg_history=100,
                 bg_threshold=50):
        self.trajectory = deque(maxlen=history_size)
        self.min_object_size = min_object_size
        self.landing_box_size = landing_box_size
        self.near_miss_threshold = near_miss_threshold
        
        # Background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=bg_history, 
            varThreshold=bg_threshold
        )
        
        # Tracking state
        self.tracking = False
        self.object_bbox = None
        self.landing_zone = None
        self.prediction_matched = None
        self.kf = None
        self.future_points = []

    def initialize_kalman_filter(self, initial_pos):
        """
        Initialize a Kalman filter for trajectory prediction.
        
        Args:
            initial_pos (tuple): Initial (x, y) position
        """
        kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1  # Time step per frame
        
        # State transition matrix (constant acceleration model)
        # State vector: [x, y, vx, vy, ax, ay]
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe x and y positions)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Initial state
        kf.x = np.array([*initial_pos, 0, 0, 0, 0])
        
        # Covariance matrix (high initial uncertainty)
        kf.P *= 1000
        
        # Measurement noise
        kf.R = np.diag([10, 10])
        
        # Process noise
        kf.Q = np.eye(6) * 0.1
        
        return kf

    def get_landing_point(self, frame_height, frame_width):
        """
        Calculate where the object will land based on the current Kalman filter state.
        
        Args:
            frame_height (int): Height of the frame
            frame_width (int): Width of the frame
            
        Returns:
            tuple or None: (x, y) landing coordinates or None if no landing is predicted
        """
        if self.kf is None:
            return None
            
        # Vertical motion parameters (y = y0 + v0*t + 0.5*a*t^2)
        a = 0.5 * self.kf.x[5]  # Vertical acceleration
        b = self.kf.x[3]        # Vertical velocity
        c = self.kf.x[1] - frame_height  # Current y position - frame height
        
        # Solve for time when object reaches bottom of frame
        discriminant = b**2 - 4*a*c if a != 0 else 0
        
        if a != 0 and discriminant >= 0:
            # Quadratic equation solution (take the positive time)
            t_land1 = (-b + np.sqrt(discriminant))/(2*a)
            t_land2 = (-b - np.sqrt(discriminant))/(2*a)
            t_land = max([t for t in [t_land1, t_land2] if t > 0], default=-1)
        elif b != 0:
            # Linear equation solution (if no acceleration)
            t_land = -c / b
        else:
            return None
        
        # Only predict if time is positive
        if t_land > 0:
            # Horizontal motion prediction
            x_land = self.kf.x[0] + self.kf.x[2]*t_land + 0.5*self.kf.x[4]*t_land**2
            x_land = int(np.clip(x_land, 0, frame_width-1))
            return (x_land, frame_height-1)
            
        return None

    def check_landing_match(self, obj_bbox, lz):
        """
        Check if the object landing matches the prediction.
            
        Returns:
            tuple: (match status, exact match flag)
        """
        if not lz:
            return None, False
            
        x, y, w, h = obj_bbox
        lx, ly, lw, lh = lz
        
        # Check for rectangle intersection
        intersect = not (x+w < lx or x > lx+lw or y+h < ly or y > ly+lh)
        if intersect:
            return True, True
        
        # Check for near miss
        obj_center = (x + w//2, y + h//2)
        lz_center = (lx + lw//2, ly + lh//2)
        dist = np.hypot(obj_center[0]-lz_center[0], obj_center[1]-lz_center[1])
        
        return dist <= self.near_miss_threshold, False

    def process_frame(self, frame):
        # Detect moving objects using background subtraction
        fg_mask = self.bg_sub.apply(frame)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_object_size]
        
        # Update tracking based on detected objects
        if valid_contours:
            if not self.tracking:
                # Start tracking new object (largest contour)
                largest = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                self.object_bbox = (x, y, w, h)
                
                # Initialize Kalman filter with object center
                self.kf = self.initialize_kalman_filter((x+w//2, y+h//2))
                
                # Reset trajectory
                self.trajectory.clear()
                self.trajectory.append((x+w//2, y+h//2))
                
                # Start tracking
                self.tracking = True
                self.landing_zone = None
                self.prediction_matched = None
            else:
                # Continue tracking existing object (closest contour to last position)
                if self.trajectory:
                    closest = min(valid_contours, key=lambda c: np.hypot(
                        (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]//2 - self.trajectory[-1][0]),
                        (cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2 - self.trajectory[-1][1])
                    ))
                    
                    x, y, w, h = cv2.boundingRect(closest)
                    self.object_bbox = (x, y, w, h)
                    
                    # Update Kalman filter
                    self.kf.predict()
                    self.kf.update(np.array([x+w//2, y+h//2]))
                    
                    # Update trajectory with filtered position
                    filtered_pos = tuple(map(int, self.kf.x[:2]))
                    self.trajectory.append(filtered_pos)
                    
                    # Check if object has landed
                    if filtered_pos[1] > frame.shape[0] - 50:
                        self.prediction_matched, _ = self.check_landing_match(
                            self.object_bbox, self.landing_zone)
                    
                    # Reset if object at bottom
                    if filtered_pos[1] >= frame.shape[0] - 10:
                        self.tracking = False
                        self.kf = None
        else:
            # No valid contours detected, reset tracking if active
            if self.tracking:
                self.tracking = False
                self.kf = None
                self.trajectory.clear()
        
        # Predict future trajectory and landing zone
        self.future_points = []
        self.landing_zone = None
        
        if self.tracking and self.kf is not None:
            # Predict positions for next 20 frames
            self.future_points = [(
                int(self.kf.x[0] + self.kf.x[2]*t + 0.5*self.kf.x[4]*t**2),
                int(self.kf.x[1] + self.kf.x[3]*t + 0.5*self.kf.x[5]*t**2)
            ) for t in range(1, 21)]
            
            # Predict landing point
            lp = self.get_landing_point(frame.shape[0], frame.shape[1])
            if lp:
                self.landing_zone = (
                    lp[0] - self.landing_box_size//2, 
                    lp[1] - self.landing_box_size//2,
                    self.landing_box_size, 
                    self.landing_box_size
                )
        
        # Return current state
        return {
            'tracking': self.tracking,
            'object_bbox': self.object_bbox,
            'trajectory': list(self.trajectory),
            'future_points': self.future_points,
            'landing_zone': self.landing_zone,
            'prediction_matched': self.prediction_matched
        }

    def draw_visualization(self, frame):
        vis = frame.copy()
        
        # Draw tracked object
        if self.tracking and self.object_bbox:
            x, y, w, h = self.object_bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Draw trajectory
        for i in range(1, len(self.trajectory)):
            cv2.line(vis, self.trajectory[i-1], self.trajectory[i], (0, 255, 0), 2)
            cv2.circle(vis, self.trajectory[i], 3, (0, 255, 0), -1)
        
        # Draw predicted future trajectory
        if self.future_points:
            prev_point = self.trajectory[-1] if self.trajectory else self.future_points[0]
            for point in self.future_points:
                cv2.line(vis, prev_point, point, (255, 0, 0), 2)
                cv2.circle(vis, point, 3, (255, 0, 0), -1)
                prev_point = point
        
        # Draw landing zone
        if self.landing_zone:
            x, y, w, h = self.landing_zone
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # Show prediction result
        if self.prediction_matched is not None:
            text = "MATCH!" if self.prediction_matched else "MISMATCH!"
            color = (0, 255, 0) if self.prediction_matched else (0, 0, 255)
            cv2.putText(vis, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return vis

    def reset(self):
        """Reset the tracker state."""
        self.tracking = False
        self.object_bbox = None
        self.landing_zone = None
        self.prediction_matched = None
        self.kf = None
        self.trajectory.clear()
        self.future_points = []