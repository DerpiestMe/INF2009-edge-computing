import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.control.robot_controller import CameraServoController
from edge.sensor.sensor_gas import GasSensor
from edge.sensor.sensor_temp import TemperatureHumiditySensor
from edge.sensor.sensor_webcam import WebcamSensor
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
        sweep_left: int = 1100,
        sweep_right: int = 1900,
        sweep_step: int = 35,
        sweep_interval_s: float = 0.1,
        auto_sweep: bool = False,
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
        self.servo = CameraServoController(servo_id=servo_id)

        self.sweep_left = sweep_left
        self.sweep_right = sweep_right
        self.sweep_step = sweep_step
        self.sweep_interval_s = sweep_interval_s
        self.auto_sweep = auto_sweep

        self._last_temp_record: Optional[Dict[str, Any]] = None
        self._last_gas_record: Optional[Dict[str, Any]] = None
        self._last_print_temp_ts = 0.0
        self._last_print_gas_ts = 0.0

    def start(self) -> None:
        self._running = True
        self.webcam.start()
        self.gas_sensor.start()
        self.temp_sensor.start()
        self.servo.center(duration_ms=300)

        if not self.vision.is_ready():
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)
        self._logger.info("Servo driver: %s", self.servo.describe())
        self._logger.info(
            "Controls: q/ESC quit | s toggle auto-sweep | [ left | ] right | c center"
        )

    def stop(self) -> None:
        self._running = False
        self.webcam.stop()
        self.gas_sensor.stop()
        self.temp_sensor.stop()
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
        lines = [
            f"Camera: {camera_status}",
            f"Persons: {person_count}",
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
            self.servo.step(-self.sweep_step, duration_ms=120)
        elif key == ord("]"):
            self.servo.step(self.sweep_step, duration_ms=120)
        elif key == ord("c"):
            self.servo.center(duration_ms=200)
        return True

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
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
                detections = self.vision.detect_persons(display_frame) if self.vision.is_ready() else []
                self.vision.draw_detections(display_frame, detections)
                self._annotate_overlay(display_frame, camera_status=camera_status, person_count=len(detections))

                if self.auto_sweep:
                    self.servo.sweep_tick(
                        left_pulse=self.sweep_left,
                        right_pulse=self.sweep_right,
                        step_pulse=self.sweep_step,
                        tick_interval_s=self.sweep_interval_s,
                    )

                cv2.imshow("Edge Sensors + Vision + Servo", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
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
    parser.add_argument("--sweep-left", type=int, default=1100)
    parser.add_argument("--sweep-right", type=int, default=1900)
    parser.add_argument("--sweep-step", type=int, default=35)
    parser.add_argument("--sweep-interval", type=float, default=0.1)
    parser.add_argument("--auto-sweep", action="store_true")
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
        sweep_left=args.sweep_left,
        sweep_right=args.sweep_right,
        sweep_step=args.sweep_step,
        sweep_interval_s=args.sweep_interval,
        auto_sweep=args.auto_sweep,
    )

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("run_sensor_vision_servo").info("Received signal %s, shutting down", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run_forever()


if __name__ == "__main__":
    main()
