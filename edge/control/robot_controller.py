import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_SERVO_ID = 9
DEFAULT_CENTER_PULSE = 1500


class CameraServoController:
    """Best-effort camera servo controller for PuppyPi/HiWonder environments."""

    def __init__(
        self,
        servo_id: int = DEFAULT_SERVO_ID,
        min_pulse: int = 500,
        max_pulse: int = 2500,
        center_pulse: int = DEFAULT_CENTER_PULSE,
        default_duration_ms: int = 200,
    ) -> None:
        self.servo_id = int(servo_id)
        self.min_pulse = int(min_pulse)
        self.max_pulse = int(max_pulse)
        self.center_pulse = int(center_pulse)
        self.default_duration_ms = int(default_duration_ms)

        self._driver_name: Optional[str] = None
        self._driver: Any = None
        self._set_pulse_fn = None
        self._mode = "none"
        self._current_pulse = self.center_pulse
        self._sweep_direction = 1
        self._last_sweep_ts = 0.0
        self._module_attempts = []
        self._path_injections = []
        self._load_driver()

    @staticmethod
    def _safe_import(module_name: str) -> tuple[Optional[Any], Optional[str]]:
        try:
            return importlib.import_module(module_name), None
        except Exception as exc:
            return None, str(exc)

    def _inject_candidate_paths(self) -> None:
        candidate_paths = [
            "/home/pi/PuppyPi",
            "/home/pi/PuppyPi/HiwonderSDK",
            "/home/pi/HiwonderSDK",
            "/home/pi/puppypi",
            "/home/pi/puppypi/HiwonderSDK",
        ]
        env_extra = os.getenv("PUPPYPI_SDK_PATH")
        if env_extra:
            candidate_paths.insert(0, env_extra)

        for candidate in candidate_paths:
            path = Path(candidate)
            if not path.exists():
                continue
            resolved = str(path.resolve())
            if resolved in sys.path:
                continue
            sys.path.insert(0, resolved)
            self._path_injections.append(resolved)

    def _load_driver(self) -> None:
        self._inject_candidate_paths()
        candidates: Sequence[str] = (
            "Board",
            "HiwonderSDK.Board",
            "hiwonder.Board",
            "SDK.Board",
        )
        for module_name in candidates:
            module, error = self._safe_import(module_name)
            self._module_attempts.append((module_name, error))
            if module is None:
                continue

            # Prefer bus-servo API since user typically refers to servo IDs (e.g., ID9).
            if hasattr(module, "setBusServoPulse"):
                self._driver = module
                self._driver_name = module_name
                self._set_pulse_fn = module.setBusServoPulse
                self._mode = "bus_servo"
                return

            # Fallback to PWM-servo API if that's what the environment exposes.
            if hasattr(module, "setPWMServoPulse"):
                self._driver = module
                self._driver_name = module_name
                self._set_pulse_fn = module.setPWMServoPulse
                self._mode = "pwm_servo"
                return

    @property
    def is_available(self) -> bool:
        return self._driver is not None and self._set_pulse_fn is not None

    @property
    def current_pulse(self) -> int:
        return self._current_pulse

    def _clamp_pulse(self, pulse: int) -> int:
        return max(self.min_pulse, min(self.max_pulse, int(pulse)))

    def set_pulse(self, pulse: int, duration_ms: Optional[int] = None) -> bool:
        if not self.is_available:
            return False

        pulse = self._clamp_pulse(pulse)
        duration_ms = self.default_duration_ms if duration_ms is None else int(duration_ms)
        self._set_pulse_fn(self.servo_id, pulse, duration_ms)
        self._current_pulse = pulse
        return True

    def center(self, duration_ms: Optional[int] = None) -> bool:
        return self.set_pulse(self.center_pulse, duration_ms=duration_ms)

    def step(self, delta_pulse: int, duration_ms: Optional[int] = None) -> bool:
        return self.set_pulse(self._current_pulse + int(delta_pulse), duration_ms=duration_ms)

    def sweep_tick(
        self,
        left_pulse: int,
        right_pulse: int,
        step_pulse: int = 30,
        tick_interval_s: float = 0.08,
        duration_ms: Optional[int] = None,
    ) -> bool:
        """
        Advance one non-blocking sweep step.
        Returns True if a pulse command was sent.
        """
        if not self.is_available:
            return False

        now = time.time()
        if now - self._last_sweep_ts < tick_interval_s:
            return False
        self._last_sweep_ts = now

        left = self._clamp_pulse(min(left_pulse, right_pulse))
        right = self._clamp_pulse(max(left_pulse, right_pulse))
        step = max(1, abs(int(step_pulse)))
        next_pulse = self._current_pulse + (self._sweep_direction * step)

        if next_pulse >= right:
            next_pulse = right
            self._sweep_direction = -1
        elif next_pulse <= left:
            next_pulse = left
            self._sweep_direction = 1

        return self.set_pulse(next_pulse, duration_ms=duration_ms)

    def describe(self) -> str:
        if not self.is_available:
            return "No supported PuppyPi servo driver found"
        return f"{self._driver_name} ({self._mode}, id={self.servo_id})"

    def diagnostics(self) -> dict:
        return {
            "driver": self.describe(),
            "injected_paths": list(self._path_injections),
            "module_attempts": [
                {
                    "module": module_name,
                    "error": error,
                }
                for module_name, error in self._module_attempts
            ],
        }
