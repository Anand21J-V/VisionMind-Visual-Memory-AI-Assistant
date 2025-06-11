from detect_objects import detect_objects_in_frames

detect_objects_in_frames(
    frame_dir='test_frames',
    output_dir='test_detections',
    model_name='yolov8n.pt'  # use yolov8s.pt if you want better accuracy
)
