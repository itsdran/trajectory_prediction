import cv2

from trajectory_prediction_3D import TrajectoryPredictor

camera = cv2.VideoCapture(0)
TrajectoryPredictor.example_usage()
TrajectoryPredictor.set_cart_plane(cart_corners)

while True:
        frame = camera.read()
        objects = TrajectoryPredictor.predictor.detect_objects(frame)
        for obj in objects:
            if obj.is_moving:
                TrajectoryPredictor.add_point(obj.center)
        
        landing_point, time_to_landing, confidence = TrajectoryPredictor.predict_landing_point()
        if landing_point is not None:
            # Check if the landing point is within the cart boundaries
            is_inside_cart = TrajectoryPredictor.check_point_in_cart(landing_point, cart_corners)
            if not is_inside_cart:
                TrajectoryPredictor.trigger_alarm("Potential theft detected!")
        
        vis_frame = TrajectoryPredictor.visualize_trajectory(frame)
        TrajectoryPredictor.display(vis_frame)