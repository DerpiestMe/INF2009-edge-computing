from pathlib import Path
from typing import Dict, List, Optional

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class VisionInference:
    """YOLOv8 person detector for Raspberry Pi edge inference."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.6,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.person_class_id = 0  # COCO class id for 'person'
        self.model = None
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        if YOLO is None:
            self.load_error = "ultralytics not installed. Run: pip install ultralytics"
            return
        try:
            if self.model_path.endswith(".pt") and not Path(self.model_path).exists() and self.model_path != "yolov8n.pt":
                self.load_error = f"Model file not found: {self.model_path}"
                return
            self.model = YOLO(self.model_path)
        except Exception as exc:
            self.load_error = f"Failed to load YOLO model: {exc}"
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def detect_persons(self, frame) -> List[Dict]:
        if self.model is None:
            return []

        kwargs = {"verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        detections: List[Dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id != self.person_class_id or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    {
                        "label": "person",
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detections

    @staticmethod
    def draw_detections(frame, detections: List[Dict]):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            color = (0, 0, 255) if det.get("inside_zone") else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return frame