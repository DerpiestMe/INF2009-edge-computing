import argparse
import json
import logging
import signal
import sys
import threading
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
from puppypi_movement import PuppyPiMovementController


class FullEdgePipelineApp:
    """Uses all non-empty edge sensor/vision/control modules in one runtime."""

    def __init__(
        self,
        camera_index: int = 2,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        servo_id: int = 9,
        servo_mode: str = "pwm",
        servo_direction: int = -1,
        infer_width: int = 384,
        infer_height: int = 288,
        infer_interval_s: float = 0.12,
        force_infer_interval_s: float = 1.2,
        max_loop_fps: float = 18.0,
        disable_motion_gate: bool = True,
        disable_inference: bool = False,
        disable_overlays: bool = False,
        render_every_n: int = 1,
        profile_perf: bool = False,
        track_person: bool = False,
        track_deadband_px: int = 24,
        track_max_step: int = 18,
        track_interval_s: float = 0.06,
        enable_mobility: bool = False,
        approach_on_detect: bool = False,
        approach_close_bbox_height_px: int = 180,
        teleop_speed_x: float = 5.0,
        teleop_yaw_deg_s: float = 12.0,
        teleop_hold_timeout_s: float = 0.18,
        gait_mode: str = "walk",
        body_height: float = -10.0,
        wrist_servo_id: int = 10,
        wrist_start_pulse: int = 1100,
        wrist_on_start: bool = True,
        show_window: bool = True,
        auto_sweep: bool = False,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self.show_window = show_window
        self.auto_sweep = auto_sweep
        self.servo_direction = -1 if int(servo_direction) < 0 else 1
        self.infer_width = max(64, int(infer_width))
        self.infer_height = max(64, int(infer_height))
        self.infer_interval_s = max(0.05, float(infer_interval_s))
        self.force_infer_interval_s = max(self.infer_interval_s, float(force_infer_interval_s))
        self.motion_gate = not bool(disable_motion_gate)
        self.disable_inference = bool(disable_inference)
        self.disable_overlays = bool(disable_overlays)
        self.render_every_n = max(1, int(render_every_n))
        self.profile_perf = bool(profile_perf)
        self.track_person = bool(track_person)
        self.track_deadband_px = max(0, int(track_deadband_px))
        self.track_max_step = max(1, int(track_max_step))
        self.track_interval_s = max(0.02, float(track_interval_s))
        self.enable_mobility = bool(enable_mobility)
        self.approach_on_detect = bool(approach_on_detect)
        self.approach_close_bbox_height_px = max(60, int(approach_close_bbox_height_px))
        self.teleop_speed_x = float(teleop_speed_x)
        self.teleop_yaw_rate = float(teleop_yaw_deg_s) * (3.141592653589793 / 180.0)
        self.teleop_hold_timeout_s = max(0.05, float(teleop_hold_timeout_s))
        self.gait_mode = str(gait_mode).lower()
        self.body_height = float(body_height)
        self.wrist_servo_id = int(wrist_servo_id)
        self.wrist_start_pulse = int(wrist_start_pulse)
        self.wrist_on_start = bool(wrist_on_start)
        self.max_loop_fps = max(1.0, float(max_loop_fps))
        self.loop_min_period_s = 1.0 / self.max_loop_fps

        self.webcam = WebcamSensor(device_index=camera_index, width=width, height=height, target_fps=fps)
        self.gas_sensor = GasSensor(
            candidate_ports=["/dev/ttyACM0", "/dev/ttyUSB0", "COM3", "COM4"],
            read_timeout=0.02,
        )
        self.temp_sensor = TemperatureHumiditySensor(i2c_bus_index=1, i2c_address=0x38)
        self.motion = MotionDetector(blur_size=15, diff_threshold=24, min_area=1200)
        self.vision = VisionInference(
            model_path=model_path,
            conf_threshold=confidence,
            imgsz=max(self.infer_width, self.infer_height),
        )
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
        self._last_presented_infer_ts = 0.0
        self._last_detections: List[Dict[str, Any]] = []
        self._inference_reused = False
        self._infer_lock = threading.Lock()
        self._infer_input: Optional[Dict[str, Any]] = None
        self._infer_busy = False
        self._infer_stop = False
        self._infer_thread: Optional[threading.Thread] = None
        self._loop_counter = 0
        self._perf_last_log_ts = time.time()
        self._perf_accum = {"capture_ms": 0.0, "sensor_ms": 0.0, "motion_ms": 0.0, "overlay_ms": 0.0}
        self._next_gas_due_ts = 0.0
        self._next_temp_due_ts = 0.0
        self._gas_poll_interval_s = 0.1
        self._temp_poll_interval_s = 1.0
        self._last_track_ts = 0.0
        self._movement = PuppyPiMovementController(
            max_x_cm_s=max(5.0, self.teleop_speed_x),
            max_yaw_rate_rad_s=max(0.2, self.teleop_yaw_rate),
            gait_mode=self.gait_mode,
            body_height=self.body_height,
        )
        self._approach_active = False
        self._teleop_active_x = 0.0
        self._teleop_active_yaw = 0.0
        self._teleop_last_input_ts = 0.0
        self._manual_drive_until_ts = 0.0
        self._last_approach_log_ts = 0.0

    def start(self) -> None:
        self._running = True
        self.webcam.start()
        self.gas_sensor.start()
        self.temp_sensor.start()
        self.servo.center()
        if self.wrist_on_start:
            moved = self.servo.set_pulse_for_id(self.wrist_servo_id, self.wrist_start_pulse, duration_ms=300)
            self._logger.info(
                "Wrist startup pulse: id=%s pulse=%s applied=%s",
                self.wrist_servo_id,
                self.wrist_start_pulse,
                moved,
            )
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
        self._logger.info(
            "Perf flags: disable_inference=%s, disable_overlays=%s, render_every_n=%s",
            self.disable_inference,
            self.disable_overlays,
            self.render_every_n,
        )
        self._logger.info("Snapshot output directory: %s", self.events.snapshot_dir)
        self._logger.info(
            "Tracking: enabled=%s deadband_px=%s max_step=%s interval=%.2fs",
            self.track_person,
            self.track_deadband_px,
            self.track_max_step,
            self.track_interval_s,
        )
        self._logger.info("Servo direction multiplier: %s", self.servo_direction)
        self._logger.info(
            "Mobility: enabled=%s approach_on_detect=%s close_bbox_h=%s teleop_x=%.1f teleop_yaw=%.1fdeg/s",
            self.enable_mobility,
            self.approach_on_detect,
            self.approach_close_bbox_height_px,
            self.teleop_speed_x,
            self.teleop_yaw_rate * 180.0 / 3.141592653589793,
        )
        self._logger.info("Mobility gait mode: %s", self.gait_mode)
        self._logger.info("Mobility body height: %.2f", self._movement.body_height)
        if self.enable_mobility:
            self._logger.info(
                "Mobility keys: hold i/k forward/back, hold j/l turn left/right, <space> stop, r record toggle, p replay, h go_home"
            )
            ok = self._movement.start()
            if not ok:
                self._logger.warning("Mobility enabled but ROS movement stack is unavailable")
        if not self.vision.is_ready() and not self.disable_inference:
            self._logger.warning("Vision model unavailable: %s", self.vision.load_error)
        elif not self.disable_inference:
            self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self._infer_thread.start()

    def stop(self) -> None:
        self._running = False
        self._infer_stop = True
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)
        if self.enable_mobility:
            self._movement.stop_replay()
            self._movement.stop()
        self.webcam.stop()
        self.gas_sensor.stop()
        self.temp_sensor.stop()
        if self.show_window:
            cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return False
        key_ch = ""
        if 0 <= key <= 255:
            key_ch = chr(key).lower()
        now = time.time()
        movement_keys = {"i", "j", "k", "l", "r", "p", "h"}
        if self.enable_mobility and key_ch in movement_keys and not self._movement.is_ready:
            print("[MOBILITY] Movement stack not ready (check ROS sourcing and puppy_control workspace)")
            self._logger.warning("Mobility key '%s' ignored: movement stack not ready", key_ch)
            return True

        if key_ch == "s":
            self.auto_sweep = not self.auto_sweep
        elif key_ch == "[":
            self.servo.step(-20 * self.servo_direction)
        elif key_ch == "]":
            self.servo.step(20 * self.servo_direction)
        elif key_ch == "c":
            self.servo.center()
        elif self.enable_mobility and key_ch == "i":
            self._teleop_active_x, self._teleop_active_yaw = self.teleop_speed_x, 0.0
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + max(0.5, self.teleop_hold_timeout_s * 2.0)
        elif self.enable_mobility and key_ch == "k":
            self._teleop_active_x, self._teleop_active_yaw = -self.teleop_speed_x, 0.0
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + max(0.5, self.teleop_hold_timeout_s * 2.0)
        elif self.enable_mobility and key_ch == "j":
            self._teleop_active_x, self._teleop_active_yaw = 0.0, self.teleop_yaw_rate
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + max(0.5, self.teleop_hold_timeout_s * 2.0)
        elif self.enable_mobility and key_ch == "l":
            self._teleop_active_x, self._teleop_active_yaw = 0.0, -self.teleop_yaw_rate
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + max(0.5, self.teleop_hold_timeout_s * 2.0)
        elif self.enable_mobility and key == ord(" "):
            self._teleop_active_x, self._teleop_active_yaw = 0.0, 0.0
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + 0.1
            self._movement.stop()
        elif self.enable_mobility and key_ch == "r":
            if self._movement.recording:
                self._movement.stop_recording()
                print("[MOBILITY] Recording stopped")
                self._logger.info("Recording stopped")
            else:
                self._movement.start_recording(clear_existing=True)
                print("[MOBILITY] Recording started")
                self._logger.info("Recording started")
        elif self.enable_mobility and key_ch == "p":
            # Prevent stale teleop command from fighting replay thread.
            self._teleop_active_x, self._teleop_active_yaw = 0.0, 0.0
            self._teleop_last_input_ts = now
            self._manual_drive_until_ts = now + 0.1
            print("[MOBILITY] Replay requested")
            self._logger.info("Replay requested")
            self._movement.replay_recording(blocking=False)
        elif self.enable_mobility and key_ch == "h":
            print("[MOBILITY] Go-home requested")
            self._logger.info("Go-home requested")
            self._movement.go_home()
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

    def _track_first_person(self, detections: List[Dict[str, Any]], frame_width: int, force: bool = False) -> None:
        if (not self.track_person and not force) or not detections:
            return
        now = time.time()
        if now - self._last_track_ts < self.track_interval_s:
            return
        self._last_track_ts = now

        target = detections[0]
        x1, _, x2, _ = target["bbox"]
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

    def _inference_worker(self) -> None:
        while not self._infer_stop:
            payload: Optional[Dict[str, Any]] = None
            with self._infer_lock:
                if self._infer_input is not None and not self._infer_busy:
                    payload = self._infer_input
                    self._infer_input = None
                    self._infer_busy = True

            if payload is None:
                time.sleep(0.005)
                continue

            try:
                infer_frame = payload["infer_frame"]
                dst_width = payload["dst_width"]
                dst_height = payload["dst_height"]
                det_small = self.vision.detect_persons(infer_frame)
                detections = self._rescale_detections(
                    det_small,
                    src_width=self.infer_width,
                    src_height=self.infer_height,
                    dst_width=dst_width,
                    dst_height=dst_height,
                )
                detections = self.zones.annotate_detections_with_zone(detections)
                infer_ts = payload["ts"]
                with self._infer_lock:
                    prev_infer_ts = self._last_infer_ts
                dt_infer = max(1e-6, infer_ts - prev_infer_ts) if prev_infer_ts > 0 else self.infer_interval_s
                inst_infer_fps = 1.0 / dt_infer
                with self._infer_lock:
                    self._last_detections = detections
                    self._infer_fps_ema = inst_infer_fps if self._infer_fps_ema == 0.0 else (0.85 * self._infer_fps_ema + 0.15 * inst_infer_fps)
                    self._last_infer_ts = infer_ts
            finally:
                with self._infer_lock:
                    self._infer_busy = False

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                loop_start = time.time()
                t0 = time.time()
                cam_record = self.webcam.read()

                latest = self.webcam.get_latest_frame()
                if latest is None:
                    time.sleep(0.03)
                    continue

                _, _, frame = latest
                clean_frame = frame.copy()
                display = frame.copy()
                frame_h, frame_w = display.shape[:2]
                capture_ms = (time.time() - t0) * 1000.0

                t_sensor = time.time()
                now_sensor = time.time()
                if now_sensor >= self._next_temp_due_ts:
                    temp_record = self.temp_sensor.read()
                    if temp_record is not None:
                        self._last_temp = temp_record
                    self._next_temp_due_ts = now_sensor + self._temp_poll_interval_s

                if now_sensor >= self._next_gas_due_ts:
                    gas_record = self.gas_sensor.read()
                    if gas_record is not None:
                        self._last_gas = gas_record
                    self._next_gas_due_ts = now_sensor + self._gas_poll_interval_s

                self._print_sensor_lines()
                sensor_ms = (time.time() - t_sensor) * 1000.0

                t1 = time.time()
                motion_result = self.motion.detect_motion(display)
                motion_ms = (time.time() - t1) * 1000.0
                if not self.disable_overlays:
                    self.motion.draw_motion_boxes(display, motion_result["motion_boxes"])
                    self.zones.draw_zones(display)

                now = time.time()
                with self._infer_lock:
                    last_infer_ts = self._last_infer_ts
                    infer_busy = self._infer_busy
                infer_due = (now - last_infer_ts) >= self.infer_interval_s
                force_due = (now - last_infer_ts) >= self.force_infer_interval_s
                allow_infer = (not self.motion_gate) or motion_result["motion"] or force_due

                detections: List[Dict[str, Any]]
                if (not self.disable_inference) and self.vision.is_ready() and infer_due and allow_infer and not infer_busy:
                    infer_frame = cv2.resize(clean_frame, (self.infer_width, self.infer_height))
                    with self._infer_lock:
                        self._infer_input = {
                            "infer_frame": infer_frame,
                            "dst_width": frame_w,
                            "dst_height": frame_h,
                            "ts": now,
                        }
                with self._infer_lock:
                    detections = list(self._last_detections)
                    latest_infer_ts = self._last_infer_ts
                if latest_infer_ts > self._last_presented_infer_ts:
                    self._inference_reused = False
                    self._last_presented_infer_ts = latest_infer_ts
                else:
                    self._inference_reused = True

                t2 = time.time()
                if not self.disable_overlays:
                    self.vision.draw_detections(display, detections)
                overlay_ms = (time.time() - t2) * 1000.0

                # Save clean snapshots (no overlays) for downstream cloud inferencing.
                event = self.events.process(clean_frame, detections)
                if event:
                    print("INTRUSION EVENT")
                    print(json.dumps(event, indent=2))

                approach_tracking = self.enable_mobility and self.approach_on_detect and len(detections) > 0
                tracking_active = (self.track_person or approach_tracking) and len(detections) > 0
                if tracking_active:
                    self._track_first_person(detections, frame_width=frame_w, force=approach_tracking)

                manual_override = self.enable_mobility and (self._movement.recording or self._movement.replaying)
                manual_drive_active = self.enable_mobility and (time.time() < self._manual_drive_until_ts)
                if manual_override and self._approach_active:
                    self._logger.info("Approach paused: manual record/replay override active")
                if manual_drive_active and self._approach_active:
                    self._movement.stop()
                    self._approach_active = False
                    self._logger.info("Approach paused: manual teleop override active")
                if self.enable_mobility and self.approach_on_detect and len(detections) > 0 and not manual_override and not manual_drive_active:
                    if not self._approach_active:
                        self._movement.stop_replay()
                        self._movement.stop_recording()
                        self._logger.info("Person detected: switching to approach mode")
                        self._approach_active = True
                    close_enough = self._movement.approach_person(
                        detections[0],
                        frame_width=frame_w,
                        frame_height=frame_h,
                        close_bbox_height_px=self.approach_close_bbox_height_px,
                        max_forward_cm_s=self.teleop_speed_x,
                        servo_current_pulse=self.servo.current_pulse,
                        servo_center_pulse=self.servo.center_pulse,
                        servo_min_pulse=self.servo.min_pulse,
                        servo_max_pulse=self.servo.max_pulse,
                    )
                    now_approach = time.time()
                    if now_approach - self._last_approach_log_ts >= 1.0:
                        bbox_h = int(detections[0]["bbox"][3] - detections[0]["bbox"][1])
                        self._logger.info(
                            "Approach metrics: bbox_h=%s/%s servo_pulse=%s center=%s close=%s",
                            bbox_h,
                            self.approach_close_bbox_height_px,
                            self.servo.current_pulse,
                            self.servo.center_pulse,
                            close_enough,
                        )
                        self._last_approach_log_ts = now_approach
                    if close_enough:
                        self._movement.stop()
                        self._logger.info("Approach complete: target is close enough for face capture")
                elif self.enable_mobility and self._approach_active and (len(detections) == 0 or manual_override or manual_drive_active):
                    self._movement.stop()
                    self._approach_active = False

                if self.enable_mobility and not self._approach_active and not self._movement.replaying:
                    now_teleop = time.time()
                    if now_teleop - self._teleop_last_input_ts > self.teleop_hold_timeout_s:
                        self._teleop_active_x, self._teleop_active_yaw = 0.0, 0.0
                    self._movement.send_velocity(self._teleop_active_x, 0.0, self._teleop_active_yaw)

                if self.auto_sweep and not tracking_active:
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

                self._loop_counter += 1
                if self.show_window:
                    if self._loop_counter % self.render_every_n == 0:
                        cv2.imshow("Edge Full Pipeline", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break

                if self.profile_perf:
                    self._perf_accum["capture_ms"] += capture_ms
                    self._perf_accum["sensor_ms"] += sensor_ms
                    self._perf_accum["motion_ms"] += motion_ms
                    self._perf_accum["overlay_ms"] += overlay_ms
                    now_perf = time.time()
                    if now_perf - self._perf_last_log_ts >= 2.0 and self._loop_counter > 0:
                        n = float(self._loop_counter)
                        self._logger.info(
                            "Perf breakdown avg(ms): capture=%.1f sensor=%.1f motion=%.1f overlay=%.1f",
                            self._perf_accum["capture_ms"] / n,
                            self._perf_accum["sensor_ms"] / n,
                            self._perf_accum["motion_ms"] / n,
                            self._perf_accum["overlay_ms"] / n,
                        )
                        self._loop_counter = 0
                        self._perf_accum = {"capture_ms": 0.0, "sensor_ms": 0.0, "motion_ms": 0.0, "overlay_ms": 0.0}
                        self._perf_last_log_ts = now_perf

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
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--servo-id", type=int, default=9)
    parser.add_argument("--servo-mode", choices=["auto", "pwm", "bus"], default="pwm")
    parser.add_argument("--servo-direction", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--infer-width", type=int, default=384)
    parser.add_argument("--infer-height", type=int, default=288)
    parser.add_argument("--infer-interval", type=float, default=0.12)
    parser.add_argument("--force-infer-interval", type=float, default=0.8)
    parser.add_argument("--max-loop-fps", type=float, default=18.0)
    parser.add_argument("--disable-motion-gate", action="store_true", dest="disable_motion_gate")
    parser.add_argument("--enable-motion-gate", action="store_false", dest="disable_motion_gate")
    parser.add_argument("--disable-inference", action="store_true")
    parser.add_argument("--disable-overlays", action="store_true")
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--profile-perf", action="store_true")
    parser.add_argument("--track-person", action="store_true")
    parser.add_argument("--track-deadband-px", type=int, default=24)
    parser.add_argument("--track-max-step", type=int, default=18)
    parser.add_argument("--track-interval", type=float, default=0.06)
    parser.add_argument("--enable-mobility", action="store_true")
    parser.add_argument("--approach-on-detect", action="store_true")
    parser.add_argument("--approach-close-bbox-height", type=int, default=180)
    parser.add_argument("--teleop-speed-x", type=float, default=5.0)
    parser.add_argument("--teleop-yaw-deg-s", type=float, default=12.0)
    parser.add_argument("--teleop-hold-timeout", type=float, default=0.18)
    parser.add_argument("--gait-mode", choices=["walk", "amble", "trot"], default="walk")
    parser.add_argument("--body-height", type=float, default=-10.0, help="Robot body height for gait pose (typical range: -16..-5)")
    parser.add_argument("--wrist-servo-id", type=int, default=10)
    parser.add_argument("--wrist-start-pulse", type=int, default=1100)
    parser.add_argument("--disable-wrist-on-start", action="store_true")
    parser.add_argument("--auto-sweep", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.set_defaults(disable_motion_gate=True)
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
        servo_direction=args.servo_direction,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
        infer_interval_s=args.infer_interval,
        force_infer_interval_s=args.force_infer_interval,
        max_loop_fps=args.max_loop_fps,
        disable_motion_gate=args.disable_motion_gate,
        disable_inference=args.disable_inference,
        disable_overlays=args.disable_overlays,
        render_every_n=args.render_every_n,
        profile_perf=args.profile_perf,
        track_person=args.track_person,
        track_deadband_px=args.track_deadband_px,
        track_max_step=args.track_max_step,
        track_interval_s=args.track_interval,
        enable_mobility=args.enable_mobility,
        approach_on_detect=args.approach_on_detect,
        approach_close_bbox_height_px=args.approach_close_bbox_height,
        teleop_speed_x=args.teleop_speed_x,
        teleop_yaw_deg_s=args.teleop_yaw_deg_s,
        teleop_hold_timeout_s=args.teleop_hold_timeout,
        gait_mode=args.gait_mode,
        body_height=args.body_height,
        wrist_servo_id=args.wrist_servo_id,
        wrist_start_pulse=args.wrist_start_pulse,
        wrist_on_start=not args.disable_wrist_on_start,
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
