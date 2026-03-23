import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.control.robot_controller import CameraServoController
from edge.sensor.sensor_webcam import WebcamSensor
from edge.vision.vision_inference import VisionInference


class VisionServoTrackingOnlyApp:
    """Webcam + person detection + camera-servo tracking only (no locomotion)."""

    def __init__(
        self,
        camera_index: int = 2,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        infer_width: int = 384,
        infer_height: int = 288,
        infer_interval_s: float = 0.12,
        servo_id: int = 9,
        servo_mode: str = "pwm",
        servo_direction: int = -1,
        servo_min_pulse: int = 500,
        servo_max_pulse: int = 1500,
        servo_center_pulse: int = 1000,
        track_deadband_px: int = 24,
        track_max_step: int = 18,
        track_interval_s: float = 0.06,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False

        self.webcam = WebcamSensor(device_index=camera_index, width=width, height=height, target_fps=fps)
        self.vision = VisionInference(model_path=model_path, conf_threshold=confidence, imgsz=max(infer_width, infer_height))
        self.servo = CameraServoController(
            servo_id=servo_id,
            servo_mode=servo_mode,
            min_pulse=servo_min_pulse,
            max_pulse=servo_max_pulse,
            center_pulse=servo_center_pulse,
            default_duration_ms=150,
        )

        self.infer_width = max(64, int(infer_width))
        self.infer_height = max(64, int(infer_height))
        self.infer_interval_s = max(0.05, float(infer_interval_s))
        self.servo_direction = -1 if int(servo_direction) < 0 else 1
        self.track_deadband_px = max(0, int(track_deadband_px))
        self.track_max_step = max(1, int(track_max_step))
        self.track_interval_s = max(0.02, float(track_interval_s))

        self._last_infer_ts = 0.0
        self._last_track_ts = 0.0
        self._last_detections: List[Dict[str, Any]] = []
        self._fps_ema = 0.0
        self._infer_fps_ema = 0.0
        self._last_loop_ts = time.time()

    @staticmethod
    def _rescale_detections(
        detections: List[Dict[str, Any]],
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
    ) -> List[Dict[str, Any]]:
        if src_width <= 0 or src_height <= 0:
            return detections
        sx = float(dst_width) / float(src_width)
        sy = float(dst_height) / float(src_height)
        scaled: List[Dict[str, Any]] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            out = det.copy()
            out["bbox"] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
            scaled.append(out)
        return scaled

    def _track_first_person(self, detections: List[Dict[str, Any]], frame_width: int) -> None:
        if not detections:
            return
        now = time.time()
        if now - self._last_track_ts < self.track_interval_s:
            return
        self._last_track_ts = now

        x1, _, x2, _ = detections[0]["bbox"]
        person_cx = int((x1 + x2) / 2)
        frame_cx = int(frame_width / 2)
        error_px = person_cx - frame_cx
        if abs(error_px) <= self.track_deadband_px:
            return

        half_w = max(1, frame_width / 2.0)
        error_norm = max(-1.0, min(1.0, error_px / half_w))
        step = int(round(error_norm * self.track_max_step))
        if step == 0:
            step = 1 if error_norm > 0 else -1
        self.servo.step(step * self.servo_direction, duration_ms=80)

    def start(self) -> None:
        self._running = True
        self.webcam.start()
        self.servo.center(duration_ms=300)
        self._logger.info("Servo driver: %s", self.servo.describe())
        self._logger.info("Controls: q quit | [ left | ] right | c center")
        if not self.vision.is_ready():
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)

    def stop(self) -> None:
        self._running = False
        self.webcam.stop()
        cv2.destroyAllWindows()

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                record = self.webcam.read()
                latest = self.webcam.get_latest_frame()
                if latest is None:
                    time.sleep(0.02)
                    continue

                _, _, frame = latest
                display = frame.copy()
                frame_h, frame_w = display.shape[:2]

                now = time.time()
                infer_due = (now - self._last_infer_ts) >= self.infer_interval_s
                if self.vision.is_ready() and infer_due:
                    infer_frame = cv2.resize(display, (self.infer_width, self.infer_height))
                    det_small = self.vision.detect_persons(infer_frame)
                    detections = self._rescale_detections(
                        det_small,
                        src_width=self.infer_width,
                        src_height=self.infer_height,
                        dst_width=frame_w,
                        dst_height=frame_h,
                    )
                    self._last_detections = detections
                    dt_infer = max(1e-6, now - self._last_infer_ts) if self._last_infer_ts > 0 else self.infer_interval_s
                    self._infer_fps_ema = (1.0 / dt_infer) if self._infer_fps_ema == 0.0 else (0.85 * self._infer_fps_ema + 0.15 * (1.0 / dt_infer))
                    self._last_infer_ts = now

                detections = self._last_detections
                self._track_first_person(detections, frame_width=frame_w)
                self.vision.draw_detections(display, detections)

                now_loop = time.time()
                dt_loop = max(1e-6, now_loop - self._last_loop_ts)
                fps = 1.0 / dt_loop
                self._fps_ema = fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * fps)
                self._last_loop_ts = now_loop

                status = record["status"] if record else "unknown"
                cv2.putText(
                    display,
                    f"Camera: {status} | FPS: {self._fps_ema:.1f} | InferFPS: {self._infer_fps_ema:.1f} | Persons: {len(detections)}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Vision + Servo Tracking Only", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord("["):
                    self.servo.step(-20 * self.servo_direction, duration_ms=120)
                elif key == ord("]"):
                    self.servo.step(20 * self.servo_direction, duration_ms=120)
                elif key in (ord("c"), ord("C")):
                    self.servo.center(duration_ms=200)
        finally:
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PuppyPi vision+servo tracking only")
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--infer-width", type=int, default=384)
    parser.add_argument("--infer-height", type=int, default=288)
    parser.add_argument("--infer-interval", type=float, default=0.12)
    parser.add_argument("--servo-id", type=int, default=9)
    parser.add_argument("--servo-mode", choices=["auto", "pwm", "bus"], default="pwm")
    parser.add_argument("--servo-direction", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--servo-min-pulse", type=int, default=500)
    parser.add_argument("--servo-max-pulse", type=int, default=1500)
    parser.add_argument("--servo-center-pulse", type=int, default=1000)
    parser.add_argument("--track-deadband-px", type=int, default=24)
    parser.add_argument("--track-max-step", type=int, default=18)
    parser.add_argument("--track-interval", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    app = VisionServoTrackingOnlyApp(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model_path,
        confidence=args.confidence,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
        infer_interval_s=args.infer_interval,
        servo_id=args.servo_id,
        servo_mode=args.servo_mode,
        servo_direction=args.servo_direction,
        servo_min_pulse=args.servo_min_pulse,
        servo_max_pulse=args.servo_max_pulse,
        servo_center_pulse=args.servo_center_pulse,
        track_deadband_px=args.track_deadband_px,
        track_max_step=args.track_max_step,
        track_interval_s=args.track_interval,
    )

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("run_vision_servo_tracking_only").info("Received signal %s", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run_forever()


if __name__ == "__main__":
    main()
