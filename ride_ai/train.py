from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, img_size=640, batch_size=16, model_config='yolov8n.pt', project_name='SentryLane_Training', name='exp'):
    """
    Trains a YOLOv8 model with the specified configuration.

    Args:
        data_yaml_path (str): Path to the dataset YAML file (e.g., 'data.yaml').
                              This file should define paths to train/val images and class names.
        epochs (int): Number of training epochs.
        img_size (int): Input image size for training.
        batch_size (int): Batch size for training.
        model_config (str): Pre-trained model to start from (e.g., 'yolov8n.pt', 'yolov8s.pt').
        project_name (str): Name of the project for saving results.
        name (str): Name of the experiment for saving results.
    """
    print(f"Starting YOLOv8 training for project: {project_name}/{name}")
    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Epochs: {epochs}, Image Size: {img_size}, Batch Size: {batch_size}")
    print(f"Starting from model: {model_config}")

    # Load a pre-trained YOLOv8 model (e.g., 'yolov8n.pt' for nano, 'yolov8s.pt' for small)
    # Or load a custom YAML configuration for a new model: model = YOLO('yolov8n.yaml')
    model = YOLO(model_config)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=name
    )

    print("\nTraining complete. Results saved to 'runs/detect/train/' directory.")
    print("You can find the best trained model at 'runs/detect/train/exp/weights/best.pt'")
    print("Remember to update the path in ride_ai/detect.py if you want to use this new model.")

if __name__ == '__main__':
    # --- USER CONFIGURATION ---
    # IMPORTANT: Replace 'path/to/your/data.yaml' with the actual path to your dataset configuration file.
    # This YAML file should define the paths to your training and validation images, and class names.
    # Example data.yaml content:
    # train: ../datasets/coco128/images/train2017/
    # val: ../datasets/coco128/images/val2017/
    # nc: 80  # number of classes
    # names: ['person', 'bicycle', ..., 'toothbrush'] # class names
    
    # For custom pothole detection, your data.yaml would list your pothole dataset.
    
    # Example usage for training a pothole model:
    # train_model(
    #     data_yaml_path='path/to/your/pothole_data.yaml',
    #     epochs=100,
    #     img_size=640,
    #     batch_size=16,
    #     model_config='yolov8n.pt', # Start from a pre-trained nano model
    #     project_name='SentryLane_Pothole_Training',
    #     name='pothole_exp_v1'
    # )

    # Example usage for fine-tuning a general object detection model:
    # train_model(
    #     data_yaml_path='path/to/your/general_object_data.yaml',
    #     epochs=50,
    #     img_size=640,
    #     batch_size=8,
    #     model_config='yolov8s.pt', # Start from a pre-trained small model
    #     project_name='SentryLane_General_Training',
    #     name='general_obj_exp_v1'
    # )

    # Default example - user needs to fill this in
    print("Please configure the 'data_yaml_path' in ride_ai/train.py before running.")
    # train_model(
    #     data_yaml_path='path/to/your/data.yaml', # <<<--- IMPORTANT: Update this path!
    #     epochs=50,
    #     img_size=640,
    #     batch_size=16,
    #     model_config='yolov8n.pt',
    #     project_name='SentryLane_Training',
    #     name='default_exp'
    # )
