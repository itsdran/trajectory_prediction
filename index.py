# example_usage.py
import cv2
from trajectory_predictor import TrajectoryPredictor

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create trajectory predictor instance
predictor = TrajectoryPredictor(
    history_size=20,
    min_object_size=500,
    landing_box_size=50,
    near_miss_threshold=25
)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process the frame
    tracking_info = predictor.process_frame(frame)
    
    # Draw visualization
    visualization = predictor.draw_visualization(frame)
    
    # Display status information
    status_text = "Tracking: " + ("Yes" if tracking_info['tracking'] else "No")
    cv2.putText(visualization, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Trajectory Prediction', visualization)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
