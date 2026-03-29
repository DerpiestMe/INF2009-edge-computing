import logging
import importlib
import re
import time
import statistics
from collections import deque
from typing import Any, Deque, Dict, List, Optional

try:
    serial = importlib.import_module("serial")
    SerialException = serial.SerialException
except ModuleNotFoundError:  # pragma: no cover - hardware/runtime dependency
    serial = None  # type: ignore[assignment]

    class SerialException(Exception):
        pass


GAS_LINE_PATTERN = re.compile(r"^G:(-?\d+(?:\.\d+)?)$")


class GasSensor:
    """Owns raw Pico serial I/O for gas sensor PPM readings."""

    def __init__(
        self,
        sensor_id: str = "gas-01",
        candidate_ports: Optional[List[str]] = None,
        baudrate: int = 115200,
        read_timeout: float = 1.0,
        stale_seconds: float = 5.0,
        reconnect_backoff: float = 2.0,
        min_ppm: float = 0.0,
        max_ppm: float = 10000.0,
        buffer_size: int = 100,
        alert_warning_ppm: float = 1000.0,
        alert_critical_ppm: float = 2000.0,
        alert_emergency_ppm: float = 5000.0,
        spike_window_seconds: float = 120.0,
        spike_warning_delta_ppm: float = 300.0,
        spike_critical_delta_ppm: float = 600.0,
        history_size: int = 600,
    ) -> None:
        self.sensor_type = "gas"
        self.sensor_id = sensor_id
        self.candidate_ports = candidate_ports or ["/dev/ttyACM0", "/dev/ttyUSB0", "COM3", "COM4"]
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        self.stale_seconds = stale_seconds
        self.reconnect_backoff = reconnect_backoff
        self.min_ppm = min_ppm
        self.max_ppm = max_ppm
        self.alert_warning_ppm = alert_warning_ppm
        self.alert_critical_ppm = alert_critical_ppm
        self.alert_emergency_ppm = alert_emergency_ppm
        self.spike_window_seconds = spike_window_seconds
        self.spike_warning_delta_ppm = spike_warning_delta_ppm
        self.spike_critical_delta_ppm = spike_critical_delta_ppm

        self._logger = logging.getLogger(self.__class__.__name__)
        self._serial = None
        self._active_port: Optional[str] = None
        self._last_read_ts = 0.0
        self._last_ok_ts = 0.0
        self._last_reconnect_attempt_ts = 0.0
        self._error_count = 0
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._history: Deque[Dict[str, float]] = deque(maxlen=history_size)

    def start(self) -> bool:
        if serial is None:
            self._logger.error("pyserial is not installed. Install pyserial.")
            return False

        for port in self.candidate_ports:
            try:
                self._serial = serial.Serial(port, self.baudrate, timeout=self.read_timeout)
                self._active_port = port
                self._logger.info("Gas sensor connected on %s", port)
                return True
            except (SerialException, OSError):
                continue

        self._serial = None
        self._active_port = None
        self._logger.warning("Unable to connect to gas sensor on candidate ports: %s", self.candidate_ports)
        return False

    def stop(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._active_port = None

    def _status(self) -> str:
        now = time.time()
        if self._serial is None:
            return "offline"
        if self._last_ok_ts == 0.0:
            return "degraded"
        if now - self._last_ok_ts > self.stale_seconds:
            return "degraded"
        return "ok"

    def _health(self) -> Dict[str, Any]:
        now = time.time()
        is_stale = self._last_ok_ts == 0.0 or (now - self._last_ok_ts > self.stale_seconds)
        return {
            "error_count": self._error_count,
            "last_read_ts": self._last_read_ts,
            "last_ok_ts": self._last_ok_ts,
            "is_stale": is_stale,
            "port": self._active_port,
        }

    def _build_record(self, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sensor_type": self.sensor_type,
            "sensor_id": self.sensor_id,
            "timestamp": time.time(),
            "status": status,
            "health": self._health(),
            "payload": payload,
        }

    def _maybe_reconnect(self) -> None:
        now = time.time()
        if now - self._last_reconnect_attempt_ts < self.reconnect_backoff:
            return
        self._last_reconnect_attempt_ts = now
        self.stop()
        self.start()

    def _parse_ppm(self, line: str) -> Optional[float]:
        match = GAS_LINE_PATTERN.match(line)
        if not match:
            return None
        ppm = float(match.group(1))
        if ppm < self.min_ppm or ppm > self.max_ppm:
            return None
        return ppm

    def _make_alert(
        self,
        now_ts: float,
        code: str,
        severity: str,
        metric: str,
        value: float,
        threshold: float,
        comparison: str,
        message: str,
        baseline: Optional[float] = None,
        delta: Optional[float] = None,
        window_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        alert = {
            "schema": "edge.sensor_alert.v1",
            "timestamp": now_ts,
            "sensor_type": self.sensor_type,
            "sensor_id": self.sensor_id,
            "alert_code": code,
            "severity": severity,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "message": message,
        }
        if baseline is not None:
            alert["baseline"] = baseline
        if delta is not None:
            alert["delta"] = delta
        if window_s is not None:
            alert["window_s"] = window_s
        return alert

    def _evaluate_alerts(self, ppm: float, now_ts: float) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        # NOTE: Thresholds are CO2/IAQ-oriented defaults and should be tuned if your
        # sensor is calibrated to another gas profile.
        if ppm >= self.alert_emergency_ppm:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="gas_ppm_emergency",
                    severity="critical",
                    metric="ppm",
                    value=ppm,
                    threshold=self.alert_emergency_ppm,
                    comparison=">=",
                    message="Gas ppm reached emergency threshold",
                )
            )
        elif ppm >= self.alert_critical_ppm:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="gas_ppm_high",
                    severity="critical",
                    metric="ppm",
                    value=ppm,
                    threshold=self.alert_critical_ppm,
                    comparison=">=",
                    message="Gas ppm reached critical threshold",
                )
            )
        elif ppm >= self.alert_warning_ppm:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="gas_ppm_elevated",
                    severity="warning",
                    metric="ppm",
                    value=ppm,
                    threshold=self.alert_warning_ppm,
                    comparison=">=",
                    message="Gas ppm exceeded recommended indoor threshold",
                )
            )

        recent = [h["ppm"] for h in self._history if now_ts - h["ts"] <= self.spike_window_seconds]
        if len(recent) >= 5:
            baseline = float(statistics.median(recent))
            delta = float(ppm - baseline)
            if delta >= self.spike_critical_delta_ppm:
                alerts.append(
                    self._make_alert(
                        now_ts=now_ts,
                        code="gas_ppm_spike",
                        severity="critical",
                        metric="ppm",
                        value=ppm,
                        threshold=self.spike_critical_delta_ppm,
                        comparison="delta>=",
                        message="Gas ppm spiked sharply versus short-term baseline",
                        baseline=baseline,
                        delta=delta,
                        window_s=self.spike_window_seconds,
                    )
                )
            elif delta >= self.spike_warning_delta_ppm:
                alerts.append(
                    self._make_alert(
                        now_ts=now_ts,
                        code="gas_ppm_spike",
                        severity="warning",
                        metric="ppm",
                        value=ppm,
                        threshold=self.spike_warning_delta_ppm,
                        comparison="delta>=",
                        message="Gas ppm rose quickly versus short-term baseline",
                        baseline=baseline,
                        delta=delta,
                        window_s=self.spike_window_seconds,
                    )
                )

        return alerts

    def read(self) -> Optional[Dict[str, Any]]:
        self._last_read_ts = time.time()

        if self._serial is None and not self.start():
            self._error_count += 1
            return self._build_record(status="offline", payload={"reason": "serial_unavailable"})

        try:
            raw = self._serial.readline()
        except (SerialException, OSError):
            self._error_count += 1
            self._maybe_reconnect()
            return self._build_record(status="degraded", payload={"reason": "serial_read_failed"})

        if not raw:
            return None

        line = raw.decode("utf-8", errors="replace").strip()
        ppm = self._parse_ppm(line)
        if ppm is None:
            self._error_count += 1
            return self._build_record(status="degraded", payload={"reason": "invalid_payload", "raw": line})

        now_ts = time.time()
        self._last_ok_ts = now_ts
        alerts = self._evaluate_alerts(ppm=ppm, now_ts=now_ts)
        record = self._build_record(
            status=self._status(),
            payload={
                "ppm": ppm,
                "unit": "ppm",
                "raw": line,
                "port": self._active_port,
                "anomaly": len(alerts) > 0,
                "alerts": alerts,
            },
        )
        self._history.append({"ts": now_ts, "ppm": ppm})
        self._buffer.append(record)
        return record

    def get_buffered_readings(self) -> list:
        return list(self._buffer)

    def health_snapshot(self) -> Dict[str, Any]:
        return self._build_record(status=self._status(), payload={"heartbeat": True})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sensor = GasSensor()
    try:
        while True:
            record = sensor.read()
            if record is not None:
                print(record)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()
