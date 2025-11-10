
import cv2
from ultralytics import YOLO
import numpy as np
import datetime

# Load the YOLOv8n model for general object detection
model = YOLO('runs/detect/models/yolov8n.pt')

# Load the custom-trained pothole detection model
pothole_model = YOLO('runs/detect/models/best.pt')

def process_frame(frame):
    """
    Processes a single frame for object detection and risk assessment.
    """
    height, width, _ = frame.shape
    
    # --- Zone Definitions for Risk Assessment ---
    left_zone_end = width // 3
    center_zone_end = 2 * width // 3

    # --- Top-half detection zone ---
    detection_zone_y = height / 2

    # --- Detection ---
    results = model(frame, verbose=False)
    pothole_results = pothole_model(frame, verbose=False)

    risk_score = 0
    detections = []
    pothole_detections = []

    # Process general detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # --- Filter detections to top half of the screen ---
            if (y1 + y2) / 2 > detection_zone_y:
                continue

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            class_name = model.names[cls]
            
            # Filter for relevant classes
            if class_name in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']:
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class": class_name,
                    "confidence": conf
                })

    # Process pothole detections
    for r in pothole_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # --- Filter detections to top half of the screen ---
            if (y1 + y2) / 2 > detection_zone_y:
                continue

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            class_name = pothole_model.names[cls]
            
            if class_name == 'pothole':
                pothole_detections.append({
                    "box": (x1, y1, x2, y2),
                    "class": class_name,
                    "confidence": conf
                })

    # --- Risk Logic ---
    for det in detections:
        x1, y1, x2, y2 = det['box']
        box_center_x = (x1 + x2) / 2
        box_area = (x2 - x1) * (y2 - y1)

        # --- 3-Zone Risk Assessment ---
        zone_multiplier = 1.0
        if box_center_x < left_zone_end:
            zone_multiplier = 1.2 # Lesser risk for left zone
        elif box_center_x < center_zone_end:
            zone_multiplier = 1.5 # Highest risk for center zone
        else:
            zone_multiplier = 1.2 # Lesser risk for right zone

        # Base risk on class
        risk_weight = 1.0
        if det['class'] in ['person', 'bicycle', 'motorcycle']:
            risk_weight = 1.0
        elif det['class'] in ['bus', 'truck']:
            risk_weight = 1.2
        elif det['class'] in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow']:
            risk_weight = 1.5 # Higher risk for unpredictable animals

        # Increase risk for larger (closer) objects
        risk_weight *= (box_area / (width * height)) * 10

        risk_score += risk_weight * zone_multiplier
        
    for det in pothole_detections:
        # Make pothole risk proportional to confidence, capped at a reasonable value
        pothole_risk_contribution = det['confidence'] * 10 # Reduced multiplier for potholes
        risk_score += min(pothole_risk_contribution, 10) # Cap individual pothole contribution

    risk_score = min(int(risk_score / 4 * 100), 100) # Global scaling factor and cap

    # --- Object Counting ---
    object_counts = {}
    for det in detections:
        class_name = det['class']
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
    for det in pothole_detections:
        class_name = det['class']
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    # --- Visualization ---
    # Draw zone lines
    cv2.line(frame, (left_zone_end, 0), (left_zone_end, height), (255, 255, 255), 1)
    cv2.line(frame, (center_zone_end, 0), (center_zone_end, height), (255, 255, 255), 1)
    
    # Draw boxes for regular detections
    for det in detections:
        x1, y1, x2, y2 = det['box']
        
        # Color based on risk
        if risk_score > 70:
            color = (0, 0, 255) # Red
        elif risk_score > 40:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 255, 0) # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw boxes for potholes
    for det in pothole_detections:
        x1, y1, x2, y2 = det['box']
        color = (0, 165, 255) # Orange for potholes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    # Display risk score
    risk_text = f"Risk Score: {risk_score}"
    (text_width, text_height), baseline = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    risk_text_x = (width - text_width) // 2
    cv2.putText(frame, risk_text, (risk_text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # Display current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display object counts horizontally at the bottom
    x_offset = 10
    y_offset = height - 20 # Position near the bottom
    for obj_class, count in object_counts.items():
        text = f"{obj_class}: {count}"
        cv2.putText(frame, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_offset += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 20 # Add padding

    return frame, risk_score

def run_detection():
    """
    Generator function to yield processed frames from the video source.
    """
    video_path = 'videos/test_vid3.mp4'
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow('Ride AI', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Ride AI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get video properties for VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video writer setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output/ride_log_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame, risk_score = process_frame(frame)
        
        # Write frame to output file
        out.write(processed_frame)

        # Display the resulting frame
        cv2.imshow('Ride AI', processed_frame)
 
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()
