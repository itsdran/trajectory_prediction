import numpy as np
import cv2
from scipy.optimize import curve_fit
import time

class TrajectoryPredictor:
    def __init__(self, camera_matrix, dist_coeffs):
        """
        Initialize the 3D trajectory predictor
        
        Parameters:
        - camera_matrix: 3x3 camera intrinsic matrix
        - dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.history_points = []
        self.history_timestamps = []
        self.g = 9.81  # gravity acceleration (m/s^2)
        self.last_prediction = None
        self.cart_plane = None  # Will be defined by four corners in 3D space
        
    def set_cart_plane(self, cart_corners_3d):
        """
        Define the plane of the cart using four 3D points
        
        Parameters:
        - cart_corners_3d: Four 3D points representing the corners of the cart
        """
        self.cart_plane = cart_corners_3d
    
    def add_point(self, point_2d, timestamp=None):
        """
        Add a 2D point from the object detection to the trajectory history
        
        Parameters:
        - point_2d: (x, y) coordinates in image space
        - timestamp: Time when the point was detected
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.history_points.append(point_2d)
        self.history_timestamps.append(timestamp)
        
        # Keep only the recent history (last 10 points)
        if len(self.history_points) > 10:
            self.history_points.pop(0)
            self.history_timestamps.pop(0)
    
    def _estimate_depth(self, point_2d):
        """
        Estimate the depth (Z coordinate) of a 2D point using triangulation or depth cues
        This is a simplified version - in a real system, use stereo vision or depth sensors
        
        Parameters:
        - point_2d: (x, y) coordinates in image space
        
        Returns:
        - Estimated depth in meters
        """
        # In a real implementation, you would use:
        # 1. Stereo vision
        # 2. Time-of-flight sensor
        # 3. Object size as a depth cue
        # 4. Structured light
        
        # For this example, we'll use a simple heuristic based on the y-coordinate
        # Assuming objects farther away appear higher in the image
        height = self.camera_matrix[1, 2]  # principal point y
        
        # Simple depth estimation based on vertical position
        # This is just a placeholder - real implementation would use proper depth estimation
        rel_pos = (height - point_2d[1]) / height
        min_depth = 0.3  # 30cm minimum depth
        max_depth = 1.5  # 1.5m maximum depth
        
        return min_depth + rel_pos * (max_depth - min_depth)
    
    def _convert_2d_to_3d(self, point_2d, depth):
        """
        Convert a 2D point to 3D using camera parameters and estimated depth
        
        Parameters:
        - point_2d: (x, y) coordinates in image space
        - depth: Estimated depth (Z coordinate)
        
        Returns:
        - 3D point in camera coordinate system
        """
        # Get camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Back-project from image to 3D
        x = (point_2d[0] - cx) * depth / fx
        y = (point_2d[1] - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def _fit_3d_trajectory(self):
        """
        Fit a 3D ballistic trajectory to the observed points
        
        Returns:
        - Initial velocity vector (vx, vy, vz)
        - Initial position (x0, y0, z0)
        """
        if len(self.history_points) < 5:
            return None, None
        
        # Convert 2D points to 3D
        points_3d = []
        for i, point in enumerate(self.history_points):
            depth = self._estimate_depth(point)
            point_3d = self._convert_2d_to_3d(point, depth)
            points_3d.append(point_3d)
        
        # Get time intervals
        t0 = self.history_timestamps[0]
        times = [t - t0 for t in self.history_timestamps]
        
        # Separate x, y, z coordinates
        xs = [p[0] for p in points_3d]
        ys = [p[1] for p in points_3d]
        zs = [p[2] for p in points_3d]
        
        # Define the trajectory model in 3D
        def x_model(t, x0, vx):
            return x0 + vx * t
        
        def y_model(t, y0, vy):
            return y0 + vy * t
        
        def z_model(t, z0, vz):
            return z0 + vz * t - 0.5 * self.g * t**2
        
        # Fit the trajectories
        try:
            x_params, _ = curve_fit(x_model, times, xs)
            y_params, _ = curve_fit(y_model, times, ys)
            z_params, _ = curve_fit(z_model, times, zs)
            
            x0, vx = x_params
            y0, vy = y_params
            z0, vz = z_params
            
            return np.array([vx, vy, vz]), np.array([x0, y0, z0])
        except:
            return None, None
    
    def _intersect_ray_plane(self, ray_origin, ray_direction, plane_point, plane_normal):
        """
        Calculate the intersection of a ray with a plane
        
        Parameters:
        - ray_origin: Origin point of the ray
        - ray_direction: Direction vector of the ray
        - plane_point: A point on the plane
        - plane_normal: Normal vector of the plane
        
        Returns:
        - Intersection point
        """
        denom = np.dot(ray_direction, plane_normal)
        
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to the plane
        
        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        
        if t < 0:
            return None  # Intersection is behind the ray's origin
        
        return ray_origin + t * ray_direction
    
    def _compute_plane_normal(self):
        """
        Compute the normal vector of the cart plane
        
        Returns:
        - Normal vector of the plane
        """
        if self.cart_plane is None or len(self.cart_plane) < 3:
            # Default to up-facing plane if not defined
            return np.array([0, 0, 1])
        
        # Get vectors in the plane
        v1 = self.cart_plane[1] - self.cart_plane[0]
        v2 = self.cart_plane[2] - self.cart_plane[0]
        
        # Cross product gives the normal
        normal = np.cross(v1, v2)
        
        # Normalize
        return normal / np.linalg.norm(normal)
    
    def predict_landing_point(self, future_time=1.0):
        """
        Predict where the object will land on the cart plane
        
        Parameters:
        - future_time: How far in the future to predict (seconds)
        
        Returns:
        - Predicted 3D landing position
        - Time until landing
        - Confidence score (0-1)
        """
        if len(self.history_points) < 5:
            return None, None, 0.0
        
        # Fit trajectory
        velocity, position = self._fit_3d_trajectory()
        if velocity is None or position is None:
            return None, None, 0.0
        
        # Get the current time
        current_time = time.time() - self.history_timestamps[0]
        
        # Get cart plane parameters
        plane_point = self.cart_plane[0] if self.cart_plane is not None else np.array([0, 0, 0])
        plane_normal = self._compute_plane_normal()
        
        # Time points for trajectory
        times = np.linspace(current_time, current_time + future_time, 100)
        
        # Generate trajectory points
        trajectory = []
        for t in times:
            x = position[0] + velocity[0] * t
            y = position[1] + velocity[1] * t
            z = position[2] + velocity[2] * t - 0.5 * self.g * t**2
            trajectory.append((x, y, z))
            
            # Check if this point is below the cart plane (has landed)
            point = np.array([x, y, z])
            if self._is_point_below_plane(point, plane_point, plane_normal):
                # Find precise intersection with plane
                t_prev = times[times < t][-1] if any(times < t) else current_time
                point_prev = trajectory[-2] if len(trajectory) > 1 else position
                
                # Linear interpolation to find intersection time
                ray_origin = np.array(point_prev)
                ray_direction = point - ray_origin
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                intersection = self._intersect_ray_plane(ray_origin, ray_direction, plane_point, plane_normal)
                if intersection is not None:
                    landing_time = t_prev + np.linalg.norm(intersection - ray_origin) / np.linalg.norm(point - np.array(point_prev)) * (t - t_prev)
                    time_to_landing = landing_time - current_time
                    
                    # Calculate confidence based on trajectory quality
                    confidence = min(1.0, len(self.history_points) / 10.0)
                    
                    self.last_prediction = (intersection, time_to_landing, confidence)
                    return intersection, time_to_landing, confidence
        
        # If no intersection found in the time window, extrapolate
        if self.last_prediction is not None:
            return self.last_prediction
        
        return None, None, 0.0
    
    def _is_point_below_plane(self, point, plane_point, plane_normal):
        """
        Check if a point is below the plane
        
        Parameters:
        - point: The 3D point to check
        - plane_point: A point on the plane
        - plane_normal: Normal vector of the plane
        
        Returns:
        - True if the point is below the plane, False otherwise
        """
        return np.dot(point - plane_point, plane_normal) < 0
    
    def visualize_trajectory(self, image, future_time=1.0):
        """
        Visualize the predicted trajectory and landing point on the image
        
        Parameters:
        - image: Input image
        - future_time: How far into the future to predict
        
        Returns:
        - Image with visualized trajectory
        """
        vis_img = image.copy()
        
        # Draw historical points
        for point in self.history_points:
            cv2.circle(vis_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        
        # Predict and draw future trajectory
        if len(self.history_points) >= 5:
            velocity, position = self._fit_3d_trajectory()
            if velocity is not None and position is not None:
                current_time = time.time() - self.history_timestamps[0]
                times = np.linspace(current_time, current_time + future_time, 50)
                
                prev_point = None
                for t in times:
                    x = position[0] + velocity[0] * t
                    y = position[1] + velocity[1] * t
                    z = position[2] + velocity[2] * t - 0.5 * self.g * t**2
                    
                    # Project 3D point back to image
                    point_3d = np.array([[x, y, z]], dtype=np.float32)
                    point_2d, _ = cv2.projectPoints(point_3d, np.zeros(3), np.zeros(3), 
                                                   self.camera_matrix, self.dist_coeffs)
                    point_2d = tuple(map(int, point_2d[0][0]))
                    
                    if prev_point is not None:
                        cv2.line(vis_img, prev_point, point_2d, (255, 0, 0), 2)
                    
                    prev_point = point_2d
                
                # Draw predicted landing point
                landing_point, time_to_landing, confidence = self.predict_landing_point(future_time)
                if landing_point is not None:
                    # Project landing point to image
                    landing_3d = np.array([[landing_point[0], landing_point[1], landing_point[2]]], dtype=np.float32)
                    landing_2d, _ = cv2.projectPoints(landing_3d, np.zeros(3), np.zeros(3),
                                                     self.camera_matrix, self.dist_coeffs)
                    landing_2d = tuple(map(int, landing_2d[0][0]))
                    
                    # Draw landing point with confidence-based color
                    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                    cv2.circle(vis_img, landing_2d, 10, color, -1)
                    cv2.putText(vis_img, f"Landing in {time_to_landing:.2f}s", 
                               (landing_2d[0] + 10, landing_2d[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_img
    
    def verify_landing(self, actual_landing_point, threshold=0.1):
        """
        Verify if the actual landing point matches the prediction
        
        Parameters:
        - actual_landing_point: The actual 3D landing point
        - threshold: Distance threshold for matching (meters)
        
        Returns:
        - Match result (True/False)
        - Distance between predicted and actual points
        """
        predicted_point, _, _ = self.predict_landing_point()
        if predicted_point is None:
            return False, float('inf')
        
        distance = np.linalg.norm(predicted_point - actual_landing_point)
        return distance <= threshold, distance

# Example usage:
def example_usage():
    # Create a dummy camera matrix (replace with your actual camera calibration)
    focal_length = 1000.0
    principal_point = (640, 360)
    camera_matrix = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(5)
    
    # Initialize the predictor
    predictor = TrajectoryPredictor(camera_matrix, dist_coeffs)
    
    # Define the cart plane (replace with your actual cart dimensions)
    # Format: four corners in 3D space (in meters)
    cart_corners = np.array([
        [0.0, 0.0, 0.0],  # bottom-left
        [0.5, 0.0, 0.0],  # bottom-right
        [0.5, 0.5, 0.0],  # top-right
        [0.0, 0.5, 0.0]   # top-left
    ])
    predictor.set_cart_plane(cart_corners)

if __name__ == "__main__":
    example_usage()