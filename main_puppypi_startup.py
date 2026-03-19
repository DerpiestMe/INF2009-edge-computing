import argparse
import importlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.sensor.sensor_gas import GasSensor
from edge.sensor.sensor_temp import TemperatureHumiditySensor
from edge.sensor.sensor_webcam import WebcamSensor
from edge.vision.vision_inference import VisionInference


DEFAULT_CAMERA_SERVO_CHANNEL = 9
DEFAULT_CAMERA_SERVO_PULSE = 1500


class CameraServoController:
    """Best-effort adapter for common HiWonder servo APIs on PuppyPi."""

    def __init__(self, channel: int = DEFAULT_CAMERA_SERVO_CHANNEL) -> None:
        self.channel = channel
        self._driver_name: Optional[str] = None
        self._driver: Any = None
        self._load_driver()

    def _load_driver(self) -> None:
        candidates: Sequence[Tuple[str, str]] = (
            ("Board", "setPWMServoPulse"),
            ("HiwonderSDK.Board", "setPWMServoPulse"),
            ("hiwonder.Board", "setPWMServoPulse"),
        )
        for module_name, attr_name in candidates:
            module = self._safe_import(module_name)
            if module is None:
                continue
            if hasattr(module, attr_name):
                self._driver = module
                self._driver_name = module_name
                return

    @staticmethod
    def _safe_import(module_name: str) -> Optional[Any]:
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    @property
    def is_available(self) -> bool:
        return self._driver is not None

    def set_angle_pulse(self, pulse: int, duration_ms: int = 500) -> bool:
        if not self.is_available:
            return False
        pulse = max(500, min(2500, int(pulse)))
        self._driver.setPWMServoPulse(self.channel, pulse, duration_ms)
        return True

    def describe(self) -> str:
        if self._driver_name:
            return f"{self._driver_name}.setPWMServoPulse(channel={self.channel})"
        return "No supported PuppyPi servo driver found"


class PuppyPiStartupApp:
    """Integrated PuppyPi startup loop for webcam, sensors, and person detection."""

    def __init__(
        self,
        camera_index: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: float = 10.0,
        model_path: str = "yolov8n.pt",
        person_confidence: float = 0.6,
        gas_ports: Optional[List[str]] = None,
        servo_channel: int = DEFAULT_CAMERA_SERVO_CHANNEL,
        camera_start_servo_pulse: int = DEFAULT_CAMERA_SERVO_PULSE,
        show_window: bool = True,
    ) -> None:
        self.show_window = show_window
        self.camera_start_servo_pulse = camera_start_servo_pulse
        self._running = False
        self._last_temp_record: Optional[Dict[str, Any]] = None
        self._last_gas_record: Optional[Dict[str, Any]] = None
        self._last_detection_summary: str = "No inference yet"
        self._last_camera_status: str = "offline"
        self._last_loop_error: Optional[str] = None

        self.webcam = WebcamSensor(
            sensor_id="webcam-01",
            device_index=camera_index,
            width=camera_width,
            height=camera_height,
            target_fps=camera_fps,
            stale_seconds=3.0,
            buffer_size=30,
        )
        self.temp_sensor = TemperatureHumiditySensor(
            sensor_id="temp-01",
            i2c_bus_index=1,
            i2c_address=0x38,
            stale_seconds=5.0,
            buffer_size=100,
        )
        self.gas_sensor = GasSensor(
            sensor_id="gas-01",
            candidate_ports=gas_ports or ["/dev/ttyACM0", "/dev/ttyUSB0"],
            baudrate=115200,
            stale_seconds=5.0,
            buffer_size=100,
        )
        self.vision = VisionInference(
            model_path=model_path,
            conf_threshold=person_confidence,
            device=None,
        )
        self.camera_servo = CameraServoController(channel=servo_channel)
        self._logger = logging.getLogger(self.__class__.__name__)

    def start(self) -> None:
        self._running = True
        self._logger.info("Starting PuppyPi startup app")
        self.webcam.start()
        self.temp_sensor.start()
        self.gas_sensor.start()
        self._set_startup_servo_position()

        if not self.vision.is_ready():
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)

    def stop(self) -> None:
        self._running = False
        self._logger.info("Stopping PuppyPi startup app")
        self.webcam.stop()
        self.temp_sensor.stop()
        self.gas_sensor.stop()
        if self.show_window:
            cv2.destroyAllWindows()

    def _set_startup_servo_position(self) -> None:
        moved = self.camera_servo.set_angle_pulse(self.camera_start_servo_pulse)
        if moved:
            self._logger.info(
                "Camera servo startup pulse set to %s using %s",
                self.camera_start_servo_pulse,
                self.camera_servo.describe(),
            )
        else:
            self._logger.warning(
                "Servo driver not found. Set camera_start_servo_pulse=%s once the PuppyPi servo SDK is installed.",
                self.camera_start_servo_pulse,
            )

    @staticmethod
    def _format_temp(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Temp: no data"
        payload = record.get("payload", {})
        if "temperature_c" not in payload:
            return f"Temp: {record.get('status', 'unknown')}"
        return f"Temp: {payload['temperature_c']:.1f}C  Hum: {payload['humidity_rh']:.1f}%"

    @staticmethod
    def _format_gas(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Gas: no data"
        payload = record.get("payload", {})
        if "ppm" not in payload:
            return f"Gas: {record.get('status', 'unknown')}"
        port = payload.get("port") or record.get("health", {}).get("port") or "unknown"
        return f"Gas: {payload['ppm']:.1f} ppm ({port})"

    @staticmethod
    def _draw_text_block(frame, lines: Sequence[str], origin: Tuple[int, int] = (10, 20)) -> None:
        x, y = origin
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 22

    def _read_sensors(self) -> None:
        temp_record = self.temp_sensor.read()
        if temp_record is not None:
            self._last_temp_record = temp_record

        gas_record = self.gas_sensor.read()
        if gas_record is not None:
            self._last_gas_record = gas_record

    def _detect_people(self, frame) -> List[Dict[str, Any]]:
        if not self.vision.is_ready():
            self._last_detection_summary = f"Model unavailable: {self.vision.load_error}"
            return []
        detections = self.vision.detect_persons(frame)
        count = len(detections)
        self._last_detection_summary = f"Persons detected: {count}"
        return detections

    def _annotate_frame(self, frame, detections: Sequence[Dict[str, Any]]) -> None:
        if detections:
            self.vision.draw_detections(frame, list(detections))

        overlay_lines = [
            f"Camera: {self._last_camera_status}",
            self._format_temp(self._last_temp_record),
            self._format_gas(self._last_gas_record),
            self._last_detection_summary,
            f"Camera servo startup pulse: {self.camera_start_servo_pulse}",
        ]
        if self._last_loop_error:
            overlay_lines.append(f"Last warning: {self._last_loop_error}")
        self._draw_text_block(frame, overlay_lines)

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                camera_record = self.webcam.read()
                if camera_record is not None:
                    self._last_camera_status = camera_record.get("status", "unknown")

                self._read_sensors()
                latest = self.webcam.get_latest_frame()
                if latest is None:
                    time.sleep(0.05)
                    continue

                _, _, frame = latest
                display_frame = frame.copy()
                detections = self._detect_people(display_frame)
                self._annotate_frame(display_frame, detections)

                if self.show_window:
                    cv2.imshow("PuppyPi Startup", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                time.sleep(0.02)
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received")
        except Exception as exc:
            self._last_loop_error = str(exc)
            self._logger.exception("Unhandled runtime error: %s", exc)
            raise
        finally:
            self.stop()

    def compatibility_summary(self) -> Dict[str, Any]:
        return {
            "webcam_cv2": cv2 is not None,
            "vision_ready": self.vision.is_ready(),
            "vision_error": self.vision.load_error,
            "temp_sensor_class": self.temp_sensor.__class__.__name__,
            "gas_sensor_class": self.gas_sensor.__class__.__name__,
            "servo_driver": self.camera_servo.describe(),
            "gas_candidate_ports": self.gas_sensor.candidate_ports,
            "camera_index": self.webcam.device_index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PuppyPi startup app with webcam, sensors, and person detection")
    parser.add_argument("--camera-index", type=int, default=0, help="USB webcam device index for OpenCV")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=float, default=10.0, help="Capture FPS")
    parser.add_argument("--model-path", default="yolov8n.pt", help="YOLO model path or model name")
    parser.add_argument("--confidence", type=float, default=0.6, help="Minimum confidence for person detection")
    parser.add_argument(
        "--gas-port",
        action="append",
        dest="gas_ports",
        help="Serial port(s) to try for Pico gas readings. Repeat to provide multiple ports.",
    )
    parser.add_argument("--servo-channel", type=int, default=DEFAULT_CAMERA_SERVO_CHANNEL, help="PWM servo channel for camera/claw servo")
    parser.add_argument(
        "--camera-start-servo-pulse",
        type=int,
        default=DEFAULT_CAMERA_SERVO_PULSE,
        help="Startup servo pulse (500-2500us) for the camera/claw servo",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening an OpenCV display window")
    parser.add_argument(
        "--print-compatibility",
        action="store_true",
        help="Print the compatibility summary and exit without starting the main loop",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()
    app = PuppyPiStartupApp(
        camera_index=args.camera_index,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        model_path=args.model_path,
        person_confidence=args.confidence,
        gas_ports=args.gas_ports,
        servo_channel=args.servo_channel,
        camera_start_servo_pulse=args.camera_start_servo_pulse,
        show_window=not args.headless,
    )

    if args.print_compatibility:
        print(json.dumps(app.compatibility_summary(), indent=2))
        return

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("main_puppypi_startup").info("Received signal %s, shutting down", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run_forever()


if __name__ == "__main__":
    main()
