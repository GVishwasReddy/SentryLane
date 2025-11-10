# SentryLane

SentryLane is a real-time, multi-factor risk assessment engine for drivers. It uses a dual-model computer vision system to proactively analyze the road ahead for potential hazards, with a special focus on road quality and the driver's immediate path.

## Key Features

- **Dual-Model System**: Uniquely combines a general object detector (for cars, people, animals) with a custom-trained model specifically for **pothole detection**.
- **3-Zone Risk Logic**: The road ahead is divided into three vertical zones. The system intelligently weights risk based on a hazard's location, assigning higher priority to threats in the vehicle's direct path.
- **Weighted Scoring Algorithm**: Generates a real-time "Risk Score" based on a weighted calculation of various factors, including:
    - The type of object (e.g., unpredictable animals are weighted higher).
    - The object's size and proximity.
    - The object's location within the 3-zone grid.
- **Web Interface**: Features a Flask-based web server that streams the annotated video feed, allowing for remote monitoring or integration with a web dashboard.

## Technology Stack

- **Backend**: Flask
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Models**: 
    - `yolov8n.pt` for general object detection.
    - `best.pt` for custom pothole detection.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SentryLane.git
    cd SentryLane
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place necessary model and video files:**
    - Place your YOLO models (`best.pt`, `yolov8n.pt`) in a `runs/detect/models/` directory.
    - Place your test video (e.g., `test_vid3.mp4`) in a `videos/` directory.

## Usage

The application is managed via the `ride_ai` module.

### To run the web application and view the stream in a browser:

1.  Start the Flask server:
    ```bash
    python ride_ai/app.py
    ```
2.  Open your web browser and navigate to `http://127.0.0.1:5001/`.

### To run the standalone detection script and view the output directly:

```bash
python ride_ai/detect.py
```
