import importlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MovementStep:
    x: float
    y: float
    yaw_rate: float
    dt: float


class PuppyPiMovementController:
    """
    ROS1 movement helper for PuppyPi.

    Supports:
    - velocity publish to /puppy_control/velocity/autogait
    - optional pose/gait initialization from demo defaults
    - command recording + replay
    - person-approach steering helper
    """

    def __init__(
        self,
        node_name: str = "puppypi_movement_controller",
        velocity_topic: str = "/puppy_control/velocity/autogait",
        max_x_cm_s: float = 20.0,
        max_yaw_rate_rad_s: float = math.radians(30.0),
        gait_mode: str = "walk",
    ) -> None:
        self.node_name = node_name
        self.velocity_topic = velocity_topic
        self.max_x_cm_s = float(max_x_cm_s)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self.gait_mode = str(gait_mode).lower()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._rospy = None
        self._Velocity = None
        self._Pose = None
        self._Gait = None
        self._Empty = None
        self._velocity_pub = None
        self._pose_pub = None
        self._gait_pub = None
        self._go_home_srv = None
        self._ready = False

        self._recording = False
        self._recorded_steps: List[MovementStep] = []
        self._last_record_ts = 0.0
        self._replay_thread: Optional[threading.Thread] = None
        self._replay_stop = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def recorded_steps(self) -> List[MovementStep]:
        return list(self._recorded_steps)

    def start(self) -> bool:
        if self._ready:
            return True
        try:
            rospy = importlib.import_module("rospy")
            self._Velocity = importlib.import_module("puppy_control.msg").Velocity
            msgs = importlib.import_module("puppy_control.msg")
            self._Pose = msgs.Pose
            self._Gait = msgs.Gait
            self._Empty = importlib.import_module("std_srvs.srv").Empty
        except Exception as exc:
            self._logger.warning("ROS movement modules unavailable: %s", exc)
            return False

        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node(self.node_name, anonymous=True, disable_signals=True)

        self._velocity_pub = rospy.Publisher(self.velocity_topic, self._Velocity, queue_size=1)
        self._pose_pub = rospy.Publisher("/puppy_control/pose", self._Pose, queue_size=1)
        self._gait_pub = rospy.Publisher("/puppy_control/gait", self._Gait, queue_size=1)
        self._go_home_srv = rospy.ServiceProxy("/puppy_control/go_home", self._Empty)

        rospy.sleep(0.2)
        self._publish_default_pose_and_gait()
        self.stop()
        self._ready = True
        self._logger.info("PuppyPi movement controller ready (topic=%s)", self.velocity_topic)
        return True

    def stop(self) -> None:
        self.send_velocity(0.0, 0.0, 0.0, record=False)

    def go_home(self) -> None:
        if not self._ready:
            return
        try:
            self._go_home_srv()
        except Exception as exc:
            self._logger.warning("go_home service failed: %s", exc)

    def _publish_default_pose_and_gait(self) -> None:
        if not self._ready and self._rospy is None:
            return
        # Base pose (x_shift can be adjusted by gait mode below).
        pose = {
            "stance_x": 0.0,
            "stance_y": 0.0,
            "x_shift": -0.65,
            "height": -10.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "run_time": 500,
        }
        gait_profiles = {
            "walk": {"overlap_time": 0.15, "swing_time": 0.28, "clearance_time": 0.35, "z_clearance": 4.0, "x_shift": -0.65},
            "amble": {"overlap_time": 0.12, "swing_time": 0.22, "clearance_time": 0.12, "z_clearance": 5.0, "x_shift": -0.9},
            "trot": {"overlap_time": 0.20, "swing_time": 0.30, "clearance_time": 0.00, "z_clearance": 6.0, "x_shift": -0.6},
        }
        profile = gait_profiles.get(self.gait_mode, gait_profiles["walk"])
        pose["x_shift"] = profile["x_shift"]
        gait = {
            "overlap_time": profile["overlap_time"],
            "swing_time": profile["swing_time"],
            "clearance_time": profile["clearance_time"],
            "z_clearance": profile["z_clearance"],
        }
        try:
            self._pose_pub.publish(**pose)
            self._rospy.sleep(0.15)
            self._gait_pub.publish(**gait)
            self._rospy.sleep(0.15)
        except Exception as exc:
            self._logger.warning("Failed to publish default pose/gait: %s", exc)

    def set_gait_mode(self, gait_mode: str) -> None:
        self.gait_mode = str(gait_mode).lower()
        if self._ready:
            self._publish_default_pose_and_gait()

    def send_velocity(self, x: float, y: float = 0.0, yaw_rate: float = 0.0, record: bool = True) -> None:
        if not self._ready:
            return
        x = max(-self.max_x_cm_s, min(self.max_x_cm_s, float(x)))
        y = float(y)
        yaw_rate = max(-self.max_yaw_rate_rad_s, min(self.max_yaw_rate_rad_s, float(yaw_rate)))

        try:
            self._velocity_pub.publish(x=x, y=y, yaw_rate=yaw_rate)
        except Exception as exc:
            self._logger.warning("send_velocity failed: %s", exc)
            return

        if record and self._recording:
            now = time.time()
            dt = max(0.01, now - self._last_record_ts) if self._last_record_ts > 0 else 0.1
            self._last_record_ts = now
            self._recorded_steps.append(MovementStep(x=x, y=y, yaw_rate=yaw_rate, dt=dt))

    def start_recording(self, clear_existing: bool = True) -> None:
        if clear_existing:
            self._recorded_steps.clear()
        self._recording = True
        self._last_record_ts = time.time()
        self._logger.info("Movement recording started")

    def stop_recording(self) -> None:
        self._recording = False
        self._logger.info("Movement recording stopped (%s steps)", len(self._recorded_steps))

    def replay_recording(self, blocking: bool = False) -> None:
        if not self._recorded_steps:
            self._logger.info("No recorded movement steps to replay")
            return
        self._replay_stop = False
        if self._replay_thread is not None and self._replay_thread.is_alive():
            self._logger.info("Replay already running")
            return

        def _run_replay() -> None:
            self._logger.info("Replay started (%s steps)", len(self._recorded_steps))
            for step in self._recorded_steps:
                if self._replay_stop:
                    break
                self.send_velocity(step.x, step.y, step.yaw_rate, record=False)
                time.sleep(step.dt)
            self.stop()
            self._logger.info("Replay finished")

        if blocking:
            _run_replay()
            return
        self._replay_thread = threading.Thread(target=_run_replay, daemon=True)
        self._replay_thread.start()

    def stop_replay(self) -> None:
        self._replay_stop = True
        self.stop()

    def approach_person(
        self,
        detection: dict,
        frame_width: int,
        frame_height: int,
        close_bbox_height_px: int = 180,
        max_forward_cm_s: float = 6.0,
    ) -> bool:
        """
        Orient and approach person until close enough.
        Returns True when sufficiently close.
        """
        if not self._ready:
            return False
        x1, y1, x2, y2 = detection["bbox"]
        person_cx = (x1 + x2) / 2.0
        bbox_h = max(1.0, float(y2 - y1))
        frame_cx = frame_width / 2.0
        error_px = person_cx - frame_cx
        error_norm = max(-1.0, min(1.0, error_px / max(1.0, frame_cx)))

        yaw_cmd = error_norm * self.max_yaw_rate_rad_s
        close_enough = bbox_h >= float(close_bbox_height_px)
        if close_enough:
            self.send_velocity(0.0, 0.0, 0.0, record=False)
            return True

        # Slow approach while turning to center target.
        forward = max(1.0, min(max_forward_cm_s, (1.0 - abs(error_norm)) * max_forward_cm_s))
        self.send_velocity(forward, 0.0, yaw_cmd, record=False)
        return False
