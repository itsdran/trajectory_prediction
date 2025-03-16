# Projectile Tracking with Kalman Filter

## Overview
This project implements a **Projectile Tracking System** using **OpenCV** and **Kalman Filter** in Python. The system detects a moving object, tracks its trajectory, predicts the landing point, and verifies if the predicted landing point matches the actual landing point. If the prediction is accurate (within a set threshold), it displays **"MATCH!"** on the screen.

## Features
- **Object Detection**: Uses background subtraction (MOG2) to detect moving objects.
- **Trajectory Tracking**: Captures and visualizes the object's movement.
- **Prediction Algorithm**: Implements a **Kalman Filter** to predict the landing point.
- **Landing Verification**: Compares predicted and actual landing points.
- **Visual Feedback**: Displays labels and bounding boxes for trajectory analysis.

## Installation
### Prerequisites
Ensure you have Python installed and install the required dependencies:
```bash
pip install opencv-python numpy
```

## Usage
Run the script to start real-time tracking:
```bash
python projectile_tracking.py
```
Press **'q'** to exit the program.

## How It Works
1. **Detect Movement**: The system applies background subtraction to identify a moving object.
2. **Track Trajectory**: Captures the object's past positions and displays a trajectory.
3. **Predict Landing**: Uses a **Kalman Filter** to estimate the object's landing position.
4. **Verify Match**: Compares the predicted and actual landing points.
5. **Display Results**: If the predicted landing is close to the actual landing, "MATCH!" appears.

## Key Components
- **`ProjectileTracker` Class**: Manages object tracking and prediction.
- **`detect_moving_object()`**: Detects and isolates moving objects.
- **`predict_trajectory()`**: Uses Kalman Filter to predict future positions.
- **`check_match()`**: Compares predicted and actual landing points.
- **`run()`**: Captures frames, processes movement, and visualizes tracking.

## Example Output
- Green line: **Detected trajectory**
- White box: **Predicted landing point**
- Green box: **Actual landing point**
- "MATCH!": **Displayed when prediction is accurate**

## Future Improvements
- Enhance 3D plane consideration for better accuracy.
- Optimize background subtraction for dynamic environments.
- Integrate deep learning models for improved object recognition.

## License
This project is open-source under the MIT License.

## Author
Dranoel Flores
