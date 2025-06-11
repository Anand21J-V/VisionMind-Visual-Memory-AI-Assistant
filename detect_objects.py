from ultralytics import YOLO
import cv2
import os

def detect_objects_in_frames(frame_dir='frames', output_dir='detections', model_name='yolov8n.pt'):
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(model_name)

    # Loop through each frame
    for filename in sorted(os.listdir(frame_dir)):
        if filename.endswith('.jpg'):
            frame_path = os.path.join(frame_dir, filename)
            image = cv2.imread(frame_path)

            # Inference
            results = model(image)[0]

            # Draw results on the image
            annotated_frame = results.plot()

            # Save annotated frame
            out_path = os.path.join(output_dir, f"detection_{filename}")
            cv2.imwrite(out_path, annotated_frame)
            print(f"Detected and saved: {out_path}")

            # Print detected objects
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"{filename}: {name} ({conf:.2f})")

