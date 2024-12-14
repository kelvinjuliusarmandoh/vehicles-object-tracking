from ultralytics import YOLO
import cv2


def load_model(model_path):
    """Load Model using pretrained model.
    
    Args:
    --------
    model_path: Path to pretrained model

    Return:
    YOLO11 Pretrained model from ultralytics
    """
    return YOLO(model_path)

def drawing_bounding_boxes(frame, results):
    colors_pallette = {
        "0": (255, 0, 0), # Red
        "1": (0, 255, 0), # Green
        "2": (0, 0, 255), # Blue
        "3": (0, 255, 255), # Cyan
        "4": (255, 0, 255), # Violet
        "5": (128, 0, 0) # Brown
    }
    for result in results:
        for box in result.boxes:
            x_min, y_min = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            x_max, y_max = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

            conf = box.conf[0]
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors_pallette[str(class_id)])
            cv2.putText(frame, f"{class_name}, {conf:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 0),
                        1)
    return frame 