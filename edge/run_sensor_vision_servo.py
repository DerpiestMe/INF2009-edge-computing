import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.control.robot_controller import CameraServoController
from edge.sensor.sensor_gas import GasSensor
from edge.sensor.sensor_temp import TemperatureHumiditySensor
from edge.sensor.sensor_webcam import WebcamSensor
from edge.vision.motion_detector import MotionDetector
from edge.vision.vision_inference import VisionInference


class IntegratedEdgeApp:
    """Runs gas + temperature/humidity + webcam inference, with optional servo sweep."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 10.0,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.6,
        servo_id: int = 9,
        servo_mode: str = "pwm",
        servo_direction: int = -1,
        sweep_left: int = 500,
        sweep_right: int = 1500,
        sweep_step: int = 15,
        sweep_interval_s: float = 0.12,
        servo_min_pulse: int = 500,
        servo_max_pulse: int = 1500,
        servo_center_pulse: int = 1000,
        infer_width: int = 320,
        infer_height: int = 240,
        infer_interval_s: float = 0.25,
        force_infer_interval_s: float = 1.2,
        motion_gate: bool = True,
        max_loop_fps: float = 18.0,
        auto_sweep: bool = False,
        show_window: bool = True,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False

        self.webcam = WebcamSensor(
            sensor_id="webcam-01",
            device_index=camera_index,
            width=width,
            height=height,
            target_fps=fps,
            stale_seconds=3.0,
            buffer_size=30,
        )
        self.gas_sensor = GasSensor(
            sensor_id="gas-01",
            candidate_ports=["/dev/ttyACM0", "/dev/ttyUSB0", "COM3", "COM4"],
            baudrate=115200,
            stale_seconds=5.0,
            buffer_size=100,
        )
        self.temp_sensor = TemperatureHumiditySensor(
            sensor_id="temp-01",
            i2c_bus_index=1,
            i2c_address=0x38,
            stale_seconds=5.0,
            buffer_size=100,
        )
        self.vision = VisionInference(model_path=model_path, conf_threshold=confidence)
        self.motion_detector = MotionDetector(blur_size=15, diff_threshold=24, min_area=1200)
        self.servo = CameraServoController(
            servo_id=servo_id,
            servo_mode=servo_mode,
            min_pulse=servo_min_pulse,
            max_pulse=servo_max_pulse,
            center_pulse=servo_center_pulse,
            default_duration_ms=180,
        )

        self.sweep_left = sweep_left
        self.sweep_right = sweep_right
        self.sweep_step = sweep_step
        self.sweep_interval_s = sweep_interval_s
        self.servo_direction = -1 if int(servo_direction) < 0 else 1
        self.infer_width = max(64, int(infer_width))
        self.infer_height = max(64, int(infer_height))
        self.infer_interval_s = max(0.05, float(infer_interval_s))
        self.force_infer_interval_s = max(self.infer_interval_s, float(force_infer_interval_s))
        self.motion_gate = bool(motion_gate)
        self.max_loop_fps = max(1.0, float(max_loop_fps))
        self.loop_min_period_s = 1.0 / self.max_loop_fps
        self.auto_sweep = auto_sweep
        self.show_window = show_window

        self._last_temp_record: Optional[Dict[str, Any]] = None
        self._last_gas_record: Optional[Dict[str, Any]] = None
        self._last_print_temp_ts = 0.0
        self._last_print_gas_ts = 0.0
        self._last_detections: List[Dict[str, Any]] = []
        self._last_infer_ts = 0.0
        self._last_loop_ts = time.time()
        self._fps_ema = 0.0
        self._infer_fps_ema = 0.0
        self._inference_reused = False
        self._motion_active = False

    def start(self) -> None:
        self._running = True
        self.webcam.start()
        self.gas_sensor.start()
        self.temp_sensor.start()
        self.servo.center(duration_ms=300)

        if not self.vision.is_ready():
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)
        self._logger.info("Servo driver: %s", self.servo.describe())
        if not self.servo.is_available:
            self._logger.warning("Servo diagnostics: %s", self.servo.diagnostics())
        self._logger.info(
            "Controls: q/ESC quit | s toggle auto-sweep | [ left | ] right | c center"
        )
        self._logger.info("Servo direction multiplier: %s", self.servo_direction)
        self._logger.info(
            "Perf: infer %sx%s every %.2fs, motion_gate=%s, max_loop_fps=%.1f",
            self.infer_width,
            self.infer_height,
            self.infer_interval_s,
            self.motion_gate,
            self.max_loop_fps,
        )

    def stop(self) -> None:
        self._running = False
        self.webcam.stop()
        self.gas_sensor.stop()
        self.temp_sensor.stop()
        if self.show_window:
            cv2.destroyAllWindows()

    def _print_sensor_readings(self) -> None:
        now = time.time()
        if self._last_temp_record is not None and now - self._last_print_temp_ts >= 1.0:
            payload = self._last_temp_record.get("payload", {})
            if "temperature_c" in payload and "humidity_rh" in payload:
                print(
                    f"[TEMP] {payload['temperature_c']:.2f} C | "
                    f"[HUM] {payload['humidity_rh']:.2f} %RH"
                )
                self._last_print_temp_ts = now

        if self._last_gas_record is not None and now - self._last_print_gas_ts >= 1.0:
            payload = self._last_gas_record.get("payload", {})
            if "ppm" in payload:
                print(f"[GAS] {payload['ppm']:.2f} ppm")
                self._last_print_gas_ts = now

    def _annotate_overlay(
        self,
        frame,
        camera_status: str,
        person_count: int,
    ) -> None:
        infer_mode = "reuse" if self._inference_reused else "fresh"
        lines = [
            f"Camera: {camera_status} | FPS: {self._fps_ema:.1f} | InferFPS: {self._infer_fps_ema:.1f}",
            f"Persons: {person_count} | Motion: {self._motion_active} | Infer: {infer_mode}",
            f"Servo: {self.servo.describe()}",
            f"Servo pulse: {self.servo.current_pulse} | Auto sweep: {self.auto_sweep}",
            "Keys: q/ESC quit | s toggle sweep | [ left | ] right | c center",
        ]
        y = 20
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            y += 22

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            self.auto_sweep = not self.auto_sweep
            self._logger.info("Auto sweep set to %s", self.auto_sweep)
        elif key == ord("["):
            self.servo.step(-self.sweep_step * self.servo_direction, duration_ms=120)
        elif key == ord("]"):
            self.servo.step(self.sweep_step * self.servo_direction, duration_ms=120)
        elif key == ord("c"):
            self.servo.center(duration_ms=200)
        return True

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
            scaled_det = det.copy()
            scaled_det["bbox"] = [
                int(x1 * sx),
                int(y1 * sy),
                int(x2 * sx),
                int(y2 * sy),
            ]
            scaled.append(scaled_det)
        return scaled

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                loop_start = time.time()
                camera_record = self.webcam.read()
                camera_status = camera_record["status"] if camera_record else "unknown"

                temp_record = self.temp_sensor.read()
                if temp_record is not None:
                    self._last_temp_record = temp_record

                gas_record = self.gas_sensor.read()
                if gas_record is not None:
                    self._last_gas_record = gas_record

                self._print_sensor_readings()

                latest = self.webcam.get_latest_frame()
                if latest is None:
                    time.sleep(0.05)
                    continue

                _, _, frame = latest
                display_frame = frame.copy()
                frame_h, frame_w = display_frame.shape[:2]

                motion_result = self.motion_detector.detect_motion(display_frame)
                self._motion_active = bool(motion_result["motion"])

                now = time.time()
                infer_due = (now - self._last_infer_ts) >= self.infer_interval_s
                force_due = (now - self._last_infer_ts) >= self.force_infer_interval_s
                allow_infer = (not self.motion_gate) or self._motion_active or force_due

                if self.vision.is_ready() and infer_due and allow_infer:
                    infer_frame = cv2.resize(display_frame, (self.infer_width, self.infer_height))
                    detections_small = self.vision.detect_persons(infer_frame)
                    detections = self._rescale_detections(
                        detections_small,
                        src_width=self.infer_width,
                        src_height=self.infer_height,
                        dst_width=frame_w,
                        dst_height=frame_h,
                    )
                    self._last_detections = detections
                    dt_infer = max(1e-6, now - self._last_infer_ts) if self._last_infer_ts > 0 else self.infer_interval_s
                    inst_infer_fps = 1.0 / dt_infer
                    self._infer_fps_ema = inst_infer_fps if self._infer_fps_ema == 0.0 else (0.85 * self._infer_fps_ema + 0.15 * inst_infer_fps)
                    self._last_infer_ts = now
                    self._inference_reused = False
                else:
                    detections = self._last_detections
                    self._inference_reused = True

                self.vision.draw_detections(display_frame, detections)
                self._annotate_overlay(display_frame, camera_status=camera_status, person_count=len(detections))

                if self.auto_sweep:
                    self.servo.sweep_tick(
                        left_pulse=self.sweep_left,
                        right_pulse=self.sweep_right,
                        step_pulse=self.sweep_step * self.servo_direction,
                        tick_interval_s=self.sweep_interval_s,
                    )

                if self.show_window:
                    cv2.imshow("Edge Sensors + Vision + Servo", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key):
                        break

                loop_end = time.time()
                loop_dt = max(1e-6, loop_end - self._last_loop_ts)
                inst_fps = 1.0 / loop_dt
                self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst_fps)
                self._last_loop_ts = loop_end

                elapsed = loop_end - loop_start
                sleep_for = self.loop_min_period_s - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated gas/temp/webcam + person detection + servo sweep")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--servo-id", type=int, default=9, help="Camera servo ID/channel (default: 9)")
    parser.add_argument(
        "--servo-mode",
        choices=["auto", "pwm", "bus"],
        default="pwm",
        help="Servo control mode preference for supported SDKs",
    )
    parser.add_argument("--servo-direction", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--sweep-left", type=int, default=500)
    parser.add_argument("--sweep-right", type=int, default=1500)
    parser.add_argument("--sweep-step", type=int, default=15)
    parser.add_argument("--sweep-interval", type=float, default=0.12)
    parser.add_argument("--servo-min-pulse", type=int, default=500)
    parser.add_argument("--servo-max-pulse", type=int, default=1500)
    parser.add_argument("--servo-center-pulse", type=int, default=1000)
    parser.add_argument("--infer-width", type=int, default=320)
    parser.add_argument("--infer-height", type=int, default=240)
    parser.add_argument("--infer-interval", type=float, default=0.25)
    parser.add_argument("--force-infer-interval", type=float, default=1.2)
    parser.add_argument("--disable-motion-gate", action="store_true")
    parser.add_argument("--max-loop-fps", type=float, default=18.0)
    parser.add_argument("--auto-sweep", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()
    app = IntegratedEdgeApp(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model_path,
        confidence=args.confidence,
        servo_id=args.servo_id,
        servo_mode=args.servo_mode,
        servo_direction=args.servo_direction,
        sweep_left=args.sweep_left,
        sweep_right=args.sweep_right,
        sweep_step=args.sweep_step,
        sweep_interval_s=args.sweep_interval,
        servo_min_pulse=args.servo_min_pulse,
        servo_max_pulse=args.servo_max_pulse,
        servo_center_pulse=args.servo_center_pulse,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
        infer_interval_s=args.infer_interval,
        force_infer_interval_s=args.force_infer_interval,
        motion_gate=not args.disable_motion_gate,
        max_loop_fps=args.max_loop_fps,
        auto_sweep=args.auto_sweep,
        show_window=not args.headless,
    )

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("run_sensor_vision_servo").info("Received signal %s, shutting down", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run_forever()


if __name__ == "__main__":
    main()
