import json
import sys
import time
from pathlib import Path

import cv2

# Support running either from repo root or directly from edge/vision
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.sensor.sensor_webcam import WebcamSensor
from edge.vision.intrusion_events import IntrusionEventManager
from edge.vision.motion_detector import MotionDetector
from edge.vision.vision_inference import VisionInference
from edge.vision.zone_manager import ZoneManager


def main() -> None:
    webcam = WebcamSensor(width=640, height=480, target_fps=10.0)
    motion_detector = MotionDetector(blur_size=21, diff_threshold=25, min_area=1500)
    vision = VisionInference(model_path="yolov8n.pt", conf_threshold=0.6)
    zone_manager = ZoneManager()
    event_manager = IntrusionEventManager(
        confirm_frames=4,
        cooldown_seconds=5,
        snapshot_dir=str(REPO_ROOT / "snapshots"),
    )

    if not vision.is_ready():
        print(f"[WARN] Vision model not ready: {vision.load_error}")
        print("[WARN] Person detection will return no detections until the model issue is fixed.")

    try:
        while True:
            record = webcam.read()
            latest = webcam.get_latest_frame()
            if latest is None:
                time.sleep(0.05)
                continue

            _, _, frame = latest
            display_frame = cv2.resize(frame, (640, 480))
            frame_for_inference = display_frame.copy()

            motion_result = motion_detector.detect_motion(frame_for_inference)
            zone_manager.draw_zones(display_frame)
            motion_detector.draw_motion_boxes(display_frame, motion_result["motion_boxes"])

            detections = []
            if motion_result["motion"]:
                detections = vision.detect_persons(frame_for_inference)
                detections = zone_manager.annotate_detections_with_zone(detections)
                vision.draw_detections(display_frame, detections)

                for det in detections:
                    if det["inside_zone"]:
                        x1, y1, _, y2 = det["bbox"]
                        cv2.putText(
                            display_frame,
                            f"IN {det['zone_id']}",
                            (x1, min(470, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                event = event_manager.process(frame_for_inference, detections)
                if event:
                    print("INTRUSION EVENT")
                    print(json.dumps(event, indent=2))
                    cv2.putText(
                        display_frame,
                        "INTRUSION DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

            status = record["status"] if record else "unknown"
            overlay = f"Camera: {status} | Motion: {motion_result['motion']} | Score: {motion_result['motion_score']:.4f}"
            cv2.putText(display_frame, overlay, (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("PuppyPi Edge Vision", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()