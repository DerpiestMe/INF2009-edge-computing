import cv2
from typing import Dict, List, Tuple


class MotionDetector:
    """Lightweight frame-difference motion detector for edge pre-filtering."""

    def __init__(
        self,
        blur_size: int = 21,
        diff_threshold: int = 25,
        min_area: int = 1500,
    ) -> None:
        # Gaussian kernel must be odd and >= 3
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.blur_size = max(3, self.blur_size)
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.prev_frame = None

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

    def detect_motion(self, frame) -> Dict:
        processed = self.preprocess_frame(frame)

        if self.prev_frame is None:
            self.prev_frame = processed
            return {"motion": False, "motion_score": 0.0, "motion_boxes": []}

        frame_delta = cv2.absdiff(self.prev_frame, processed)
        thresh = cv2.threshold(frame_delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes: List[Tuple[int, int, int, int]] = []
        total_motion_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
            total_motion_area += area

        self.prev_frame = processed
        frame_area = float(frame.shape[0] * frame.shape[1]) if frame is not None else 1.0
        motion_score = total_motion_area / frame_area

        return {
            "motion": len(motion_boxes) > 0,
            "motion_score": motion_score,
            "motion_boxes": motion_boxes,
        }

    @staticmethod
    def draw_motion_boxes(frame, motion_boxes):
        for x, y, w, h in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame