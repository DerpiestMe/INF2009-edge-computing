import importlib
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

try:
    smbus2 = importlib.import_module("smbus2")
    SMBus = smbus2.SMBus
    i2c_msg = smbus2.i2c_msg
except ModuleNotFoundError:  # pragma: no cover - hardware/runtime dependency
    try:
        SMBus = importlib.import_module("smbus").SMBus
        i2c_msg = None
    except ModuleNotFoundError:
        SMBus = None
        i2c_msg = None


class TemperatureHumiditySensor:
    """Owns raw I2C reads for HiWonder temperature/humidity sensor."""

    INIT_CMD = [0xBE, 0x08, 0x00]
    MEASURE_CMD = [0xAC, 0x33, 0x00]

    def __init__(
        self,
        sensor_id: str = "temp-01",
        i2c_bus_index: int = 1,
        i2c_address: int = 0x38,
        stale_seconds: float = 5.0,
        read_wait_seconds: float = 0.08,
        max_busy_retries: int = 10,
        initial_wait_seconds: float = 0.1,
        buffer_size: int = 100,
        temp_low_warning_c: float = 18.0,
        temp_low_critical_c: float = 15.0,
        temp_high_warning_c: float = 30.0,
        temp_high_critical_c: float = 35.0,
        temp_change_window_seconds: float = 300.0,
        temp_change_warning_c: float = 1.0,
        temp_change_critical_c: float = 2.0,
        history_size: int = 600,
    ) -> None:
        self.sensor_type = "temperature_humidity"
        self.sensor_id = sensor_id
        self.i2c_bus_index = i2c_bus_index
        self.i2c_address = i2c_address
        self.stale_seconds = stale_seconds
        self.read_wait_seconds = read_wait_seconds
        self.max_busy_retries = max_busy_retries
        self.initial_wait_seconds = initial_wait_seconds

        self.temp_low_warning_c = temp_low_warning_c
        self.temp_low_critical_c = temp_low_critical_c
        self.temp_high_warning_c = temp_high_warning_c
        self.temp_high_critical_c = temp_high_critical_c
        self.temp_change_window_seconds = temp_change_window_seconds
        self.temp_change_warning_c = temp_change_warning_c
        self.temp_change_critical_c = temp_change_critical_c

        self._logger = logging.getLogger(self.__class__.__name__)
        self._bus = None
        self._last_read_ts = 0.0
        self._last_ok_ts = 0.0
        self._error_count = 0
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._history: Deque[Dict[str, float]] = deque(maxlen=history_size)
        self._read_mode = "block_data"

    @staticmethod
    def _i2c_error_payload(reason: str, error: Exception, stage: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reason": reason,
            "stage": stage,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
        }
        errno = getattr(error, "errno", None)
        if errno is not None:
            payload["errno"] = errno
        return payload

    def start(self) -> bool:
        if SMBus is None:
            self._logger.error("smbus2 is not installed. Install smbus2.")
            return False

        try:
            self._bus = SMBus(self.i2c_bus_index)
            self._logger.info(
                "Temperature sensor started on I2C bus %s, address 0x%02X",
                self.i2c_bus_index,
                self.i2c_address,
            )
            return True
        except OSError:
            self._bus = None
            self._logger.warning("Unable to open I2C bus %s", self.i2c_bus_index)
            return False

    def stop(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _status(self) -> str:
        now = time.time()
        if self._bus is None:
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
            "i2c_bus": self.i2c_bus_index,
            "i2c_address": f"0x{self.i2c_address:02X}",
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

    def _read_raw(self) -> Optional[list]:
        if self._bus is None:
            return None

        if self._read_mode == "rdwr" and i2c_msg is not None and hasattr(self._bus, "i2c_rdwr"):
            read = i2c_msg.read(self.i2c_address, 7)
            self._bus.i2c_rdwr(read)
            return list(read)

        try:
            raw = self._bus.read_i2c_block_data(self.i2c_address, 0x00, 7)
            return raw
        except OSError:
            if i2c_msg is None or not hasattr(self._bus, "i2c_rdwr"):
                raise
            read = i2c_msg.read(self.i2c_address, 7)
            self._bus.i2c_rdwr(read)
            self._read_mode = "rdwr"
            self._logger.info("Temperature sensor switched to i2c_rdwr read mode")
            return list(read)

    def _ensure_calibrated(self) -> None:
        if self._bus is None:
            return

        try:
            raw = self._read_raw()
        except OSError:
            return

        if raw is None or len(raw) < 1:
            return

        is_calibrated = (raw[0] & 0x08) != 0
        if is_calibrated:
            return

        self._bus.write_i2c_block_data(self.i2c_address, self.INIT_CMD[0], self.INIT_CMD[1:])
        time.sleep(0.02)

    def _trigger_measurement(self) -> None:
        if self._bus is None:
            return
        self._bus.write_i2c_block_data(self.i2c_address, self.MEASURE_CMD[0], self.MEASURE_CMD[1:])

    @staticmethod
    def _parse_aht20(raw: list) -> Optional[Dict[str, Any]]:
        if len(raw) < 7:
            return None

        busy = (raw[0] & 0x80) != 0
        if busy:
            return {"busy": True}

        humidity_raw = ((raw[1] << 16) | (raw[2] << 8) | raw[3]) >> 4
        temperature_raw = ((raw[3] & 0x0F) << 16) | (raw[4] << 8) | raw[5]

        humidity = (humidity_raw * 100.0) / 1048576.0
        temperature = (temperature_raw * 200.0) / 1048576.0 - 50.0

        return {
            "busy": False,
            "temperature_c": round(temperature, 2),
            "humidity_rh": round(humidity, 2),
        }

    def _make_alert(
        self,
        now_ts: float,
        code: str,
        severity: str,
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
            "metric": "temperature_c",
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

    def _evaluate_temp_alerts(self, temperature_c: float, now_ts: float) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        # Low temperature alerts (18C minimum indoor comfort-health baseline often cited in public health guidance).
        if temperature_c <= self.temp_low_critical_c:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="temperature_low",
                    severity="critical",
                    value=temperature_c,
                    threshold=self.temp_low_critical_c,
                    comparison="<=",
                    message="Temperature dropped to critical low range",
                )
            )
        elif temperature_c <= self.temp_low_warning_c:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="temperature_low",
                    severity="warning",
                    value=temperature_c,
                    threshold=self.temp_low_warning_c,
                    comparison="<=",
                    message="Temperature is lower than recommended indoor range",
                )
            )

        # High temperature alerts (operational comfort/safety defaults; tune for deployment context).
        if temperature_c >= self.temp_high_critical_c:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="temperature_high",
                    severity="critical",
                    value=temperature_c,
                    threshold=self.temp_high_critical_c,
                    comparison=">=",
                    message="Temperature reached critical high range",
                )
            )
        elif temperature_c >= self.temp_high_warning_c:
            alerts.append(
                self._make_alert(
                    now_ts=now_ts,
                    code="temperature_high",
                    severity="warning",
                    value=temperature_c,
                    threshold=self.temp_high_warning_c,
                    comparison=">=",
                    message="Temperature is higher than normal indoor range",
                )
            )

        recent = [h for h in self._history if now_ts - h["ts"] <= self.temp_change_window_seconds]
        if len(recent) >= 2:
            baseline = float(recent[0]["temperature_c"])
            delta = float(temperature_c - baseline)
            abs_delta = abs(delta)
            if abs_delta >= self.temp_change_critical_c:
                alerts.append(
                    self._make_alert(
                        now_ts=now_ts,
                        code="temperature_change_spike",
                        severity="critical",
                        value=temperature_c,
                        threshold=self.temp_change_critical_c,
                        comparison="abs(delta)>=",
                        message="Temperature changed sharply in short time window",
                        baseline=baseline,
                        delta=delta,
                        window_s=self.temp_change_window_seconds,
                    )
                )
            elif abs_delta >= self.temp_change_warning_c:
                alerts.append(
                    self._make_alert(
                        now_ts=now_ts,
                        code="temperature_change_spike",
                        severity="warning",
                        value=temperature_c,
                        threshold=self.temp_change_warning_c,
                        comparison="abs(delta)>=",
                        message="Temperature changed significantly in short time window",
                        baseline=baseline,
                        delta=delta,
                        window_s=self.temp_change_window_seconds,
                    )
                )

        return alerts

    def read(self) -> Optional[Dict[str, Any]]:
        self._last_read_ts = time.time()

        if self._bus is None and not self.start():
            self._error_count += 1
            return self._build_record(status="offline", payload={"reason": "i2c_unavailable"})

        try:
            self._ensure_calibrated()
            self._trigger_measurement()
        except OSError as error:
            self._error_count += 1
            return self._build_record(
                status="degraded",
                payload=self._i2c_error_payload("i2c_write_failed", error, stage="trigger_measurement"),
            )

        last_raw = None
        parsed = None
        for attempt in range(self.max_busy_retries):
            try:
                raw = self._read_raw()
            except OSError as error:
                self._error_count += 1
                return self._build_record(
                    status="degraded",
                    payload=self._i2c_error_payload("i2c_read_failed", error, stage=f"read_attempt_{attempt + 1}"),
                )

            if raw is None:
                return None

            last_raw = raw
            parsed = self._parse_aht20(raw)
            if parsed is None:
                break
            if not parsed.get("busy", False):
                break

            if attempt == 0:
                time.sleep(self.initial_wait_seconds)
            else:
                time.sleep(self.read_wait_seconds)

        if parsed is None:
            self._error_count += 1
            return self._build_record(status="degraded", payload={"reason": "invalid_payload", "raw": last_raw})

        if parsed.get("busy", False):
            self._error_count += 1
            return self._build_record(
                status="degraded",
                payload={
                    "reason": "sensor_busy_timeout",
                    "raw": last_raw,
                    "retries": self.max_busy_retries,
                },
            )

        now_ts = time.time()
        self._last_ok_ts = now_ts
        temperature_c = float(parsed["temperature_c"])
        alerts = self._evaluate_temp_alerts(temperature_c=temperature_c, now_ts=now_ts)
        record = self._build_record(
            status=self._status(),
            payload={
                "temperature_c": parsed["temperature_c"],
                "humidity_rh": parsed["humidity_rh"],
                "temperature_unit": "C",
                "humidity_unit": "%RH",
                "raw": last_raw,
                "read_mode": self._read_mode,
                "anomaly": len(alerts) > 0,
                "alerts": alerts,
            },
        )
        self._history.append({"ts": now_ts, "temperature_c": temperature_c})
        self._buffer.append(record)
        return record

    def get_buffered_readings(self) -> list:
        return list(self._buffer)

    def health_snapshot(self) -> Dict[str, Any]:
        return self._build_record(status=self._status(), payload={"heartbeat": True})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sensor = TemperatureHumiditySensor()
    try:
        while True:
            record = sensor.read()
            if record is not None:
                print(record)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()

