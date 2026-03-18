import logging
import importlib
import re
import time
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

        self._logger = logging.getLogger(self.__class__.__name__)
        self._serial = None
        self._active_port: Optional[str] = None
        self._last_read_ts = 0.0
        self._last_ok_ts = 0.0
        self._last_reconnect_attempt_ts = 0.0
        self._error_count = 0
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)

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

        self._last_ok_ts = time.time()
        record = self._build_record(
            status=self._status(),
            payload={
                "ppm": ppm,
                "unit": "ppm",
                "raw": line,
                "port": self._active_port,
            },
        )
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