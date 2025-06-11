import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_objects_in_video(video_path, output_video='tracked_output.mp4', model_name='yolov8n.pt'):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Load YOLOv8 model
    model = YOLO(model_name)

    # Initialize Deep SORT
    tracker = DeepSort(max_age=30)

    memory = []  # to store object appearance info: (frame_no, object, id, box)

    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        results = model(frame)[0]

        detections = []
        object_labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Pass label in metadata
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, name))
            object_labels.append(name)

        # Track objects
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(r), int(b)

            # Get label back from track data
            label = track.det_class  # only available if DeepSort stored it

            if label is None:
                label = "unknown"

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save to memory
            memory.append({
                "frame": frame_no,
                "object": label,
                "id": track_id,
                "bbox": (x1, y1, x2, y2)
            })

        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()

    print(f"Tracking complete. Video saved as: {output_video}")
    return memory
