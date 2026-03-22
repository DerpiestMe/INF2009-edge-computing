import argparse
import json
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
from edge.vision.intrusion_events import IntrusionEventManager
from edge.vision.motion_detector import MotionDetector
from edge.vision.vision_inference import VisionInference
from edge.vision.zone_manager import ZoneManager


class FullEdgePipelineApp:
    """Uses all non-empty edge sensor/vision/control modules in one runtime."""

    def __init__(
        self,
        camera_index: int = 2,
        width: int = 640,
        height: int = 480,
        fps: float = 10.0,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.6,
        servo_id: int = 9,
        servo_mode: str = "pwm",
        infer_width: int = 320,
        infer_height: int = 240,
        infer_interval_s: float = 0.25,
        force_infer_interval_s: float = 1.2,
        max_loop_fps: float = 18.0,
        disable_motion_gate: bool = False,
        show_window: bool = True,
        auto_sweep: bool = False,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self.show_window = show_window
        self.auto_sweep = auto_sweep
        self.infer_width = max(64, int(infer_width))
        self.infer_height = max(64, int(infer_height))
        self.infer_interval_s = max(0.05, float(infer_interval_s))
        self.force_infer_interval_s = max(self.infer_interval_s, float(force_infer_interval_s))
        self.motion_gate = not bool(disable_motion_gate)
        self.max_loop_fps = max(1.0, float(max_loop_fps))
        self.loop_min_period_s = 1.0 / self.max_loop_fps

        self.webcam = WebcamSensor(device_index=camera_index, width=width, height=height, target_fps=fps)
        self.gas_sensor = GasSensor(candidate_ports=["/dev/ttyACM0", "/dev/ttyUSB0", "COM3", "COM4"])
        self.temp_sensor = TemperatureHumiditySensor(i2c_bus_index=1, i2c_address=0x38)
        self.motion = MotionDetector(blur_size=15, diff_threshold=24, min_area=1200)
        self.vision = VisionInference(model_path=model_path, conf_threshold=confidence)
        self.zones = ZoneManager()
        self.events = IntrusionEventManager(
            confirm_frames=4,
            cooldown_seconds=5,
            snapshot_dir=str(REPO_ROOT / "snapshots"),
        )
        self.servo = CameraServoController(
            servo_id=servo_id,
            servo_mode=servo_mode,
            min_pulse=500,
            max_pulse=1500,
            center_pulse=1000,
            default_duration_ms=180,
        )

        self._last_temp: Optional[Dict[str, Any]] = None
        self._last_gas: Optional[Dict[str, Any]] = None
        self._last_print_ts = 0.0
        self._fps_ema = 0.0
        self._infer_fps_ema = 0.0
        self._last_loop_ts = time.time()
        self._last_infer_ts = 0.0
        self._last_detections: List[Dict[str, Any]] = []
        self._inference_reused = False

    def start(self) -> None:
        self._running = True
        self.webcam.start()
        self.gas_sensor.start()
        self.temp_sensor.start()
        self.servo.center()
        self._logger.info("Servo: %s", self.servo.describe())
        self._logger.info("Controls: q quit | s toggle sweep | [ left | ] right | c center")
        self._logger.info(
            "Perf: infer %sx%s every %.2fs, force %.2fs, motion_gate=%s, max_loop_fps=%.1f",
            self.infer_width,
            self.infer_height,
            self.infer_interval_s,
            self.force_infer_interval_s,
            self.motion_gate,
            self.max_loop_fps,
        )
        if not self.vision.is_ready():
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)

    def stop(self) -> None:
        self._running = False
        self.webcam.stop()
        self.gas_sensor.stop()
        self.temp_sensor.stop()
        if self.show_window:
            cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            self.auto_sweep = not self.auto_sweep
        elif key == ord("["):
            self.servo.step(-20)
        elif key == ord("]"):
            self.servo.step(20)
        elif key == ord("c"):
            self.servo.center()
        return True

    def _print_sensor_lines(self) -> None:
        now = time.time()
        if now - self._last_print_ts < 1.0:
            return
        self._last_print_ts = now
        if self._last_temp is not None and "temperature_c" in self._last_temp.get("payload", {}):
            p = self._last_temp["payload"]
            print(f"[TEMP] {p['temperature_c']:.2f} C | [HUM] {p['humidity_rh']:.2f} %RH")
        if self._last_gas is not None and "ppm" in self._last_gas.get("payload", {}):
            print(f"[GAS] {self._last_gas['payload']['ppm']:.2f} ppm")

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

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                loop_start = time.time()
                cam_record = self.webcam.read()
                temp_record = self.temp_sensor.read()
                gas_record = self.gas_sensor.read()
                if temp_record is not None:
                    self._last_temp = temp_record
                if gas_record is not None:
                    self._last_gas = gas_record
                self._print_sensor_lines()

                latest = self.webcam.get_latest_frame()
                if latest is None:
                    time.sleep(0.03)
                    continue

                _, _, frame = latest
                display = frame.copy()
                frame_h, frame_w = display.shape[:2]
                motion_result = self.motion.detect_motion(display)
                self.motion.draw_motion_boxes(display, motion_result["motion_boxes"])
                self.zones.draw_zones(display)

                now = time.time()
                infer_due = (now - self._last_infer_ts) >= self.infer_interval_s
                force_due = (now - self._last_infer_ts) >= self.force_infer_interval_s
                allow_infer = (not self.motion_gate) or motion_result["motion"] or force_due

                detections: List[Dict[str, Any]]
                if self.vision.is_ready() and infer_due and allow_infer:
                    infer_frame = cv2.resize(display, (self.infer_width, self.infer_height))
                    det_small = self.vision.detect_persons(infer_frame)
                    detections = self._rescale_detections(
                        det_small,
                        src_width=self.infer_width,
                        src_height=self.infer_height,
                        dst_width=frame_w,
                        dst_height=frame_h,
                    )
                    detections = self.zones.annotate_detections_with_zone(detections)
                    self._last_detections = detections
                    dt_infer = max(1e-6, now - self._last_infer_ts) if self._last_infer_ts > 0 else self.infer_interval_s
                    inst_infer_fps = 1.0 / dt_infer
                    self._infer_fps_ema = inst_infer_fps if self._infer_fps_ema == 0.0 else (0.85 * self._infer_fps_ema + 0.15 * inst_infer_fps)
                    self._last_infer_ts = now
                    self._inference_reused = False
                else:
                    detections = self._last_detections
                    self._inference_reused = True

                self.vision.draw_detections(display, detections)

                event = self.events.process(display, detections)
                if event:
                    print("INTRUSION EVENT")
                    print(json.dumps(event, indent=2))

                if self.auto_sweep:
                    self.servo.sweep_tick(
                        left_pulse=500,
                        right_pulse=1500,
                        step_pulse=15,
                        tick_interval_s=0.12,
                    )

                now = time.time()
                dt = max(1e-6, now - self._last_loop_ts)
                fps = 1.0 / dt
                self._fps_ema = fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * fps)
                self._last_loop_ts = now

                status = cam_record["status"] if cam_record else "unknown"
                infer_mode = "reuse" if self._inference_reused else "fresh"
                cv2.putText(
                    display,
                    f"Camera: {status} | FPS: {self._fps_ema:.1f} | InferFPS: {self._infer_fps_ema:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Motion: {motion_result['motion']} | Infer: {infer_mode} | Persons: {len(detections)}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if self.show_window:
                    cv2.imshow("Edge Full Pipeline", display)
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key):
                        break

                elapsed = time.time() - loop_start
                sleep_for = self.loop_min_period_s - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full edge pipeline (sensors + motion + zones + intrusion + servo)")
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--servo-id", type=int, default=9)
    parser.add_argument("--servo-mode", choices=["auto", "pwm", "bus"], default="pwm")
    parser.add_argument("--infer-width", type=int, default=320)
    parser.add_argument("--infer-height", type=int, default=240)
    parser.add_argument("--infer-interval", type=float, default=0.25)
    parser.add_argument("--force-infer-interval", type=float, default=1.2)
    parser.add_argument("--max-loop-fps", type=float, default=18.0)
    parser.add_argument("--disable-motion-gate", action="store_true")
    parser.add_argument("--auto-sweep", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()
    app = FullEdgePipelineApp(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model_path,
        confidence=args.confidence,
        servo_id=args.servo_id,
        servo_mode=args.servo_mode,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
        infer_interval_s=args.infer_interval,
        force_infer_interval_s=args.force_infer_interval,
        max_loop_fps=args.max_loop_fps,
        disable_motion_gate=args.disable_motion_gate,
        show_window=not args.headless,
        auto_sweep=args.auto_sweep,
    )

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("run_edge_full_pipeline").info("Received signal %s", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run_forever()


if __name__ == "__main__":
    main()
