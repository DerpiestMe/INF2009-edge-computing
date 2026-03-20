import logging
import importlib
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

try:
	SMBus = importlib.import_module("smbus2").SMBus
except ModuleNotFoundError:  # pragma: no cover - hardware/runtime dependency
	try:
		SMBus = importlib.import_module("smbus").SMBus
	except ModuleNotFoundError:
		SMBus = None


class TemperatureHumiditySensor:
	"""Owns raw I2C reads for HiWonder temperature/humidity sensor."""

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
	) -> None:
		self.sensor_type = "temperature_humidity"
		self.sensor_id = sensor_id
		self.i2c_bus_index = i2c_bus_index
		self.i2c_address = i2c_address
		self.stale_seconds = stale_seconds
		self.read_wait_seconds = read_wait_seconds
		self.max_busy_retries = max_busy_retries
		self.initial_wait_seconds = initial_wait_seconds

		self._logger = logging.getLogger(self.__class__.__name__)
		self._bus = None
		self._last_read_ts = 0.0
		self._last_ok_ts = 0.0
		self._error_count = 0
		self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)

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

		# AHT20-compatible measurement command used by many 0x38 temp/humidity modules.
		self._bus.write_i2c_block_data(self.i2c_address, self.MEASURE_CMD[0], self.MEASURE_CMD[1:])
		time.sleep(self.read_wait_seconds)
		raw = self._bus.read_i2c_block_data(self.i2c_address, 0x00, 7)
		return raw

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

	def read(self) -> Optional[Dict[str, Any]]:
		self._last_read_ts = time.time()

		if self._bus is None and not self.start():
			self._error_count += 1
			return self._build_record(status="offline", payload={"reason": "i2c_unavailable"})

		last_raw = None
		parsed = None
		for attempt in range(self.max_busy_retries):
			try:
				raw = self._read_raw()
			except OSError:
				self._error_count += 1
				return self._build_record(status="degraded", payload={"reason": "i2c_read_failed"})

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

		self._last_ok_ts = time.time()
		record = self._build_record(
			status=self._status(),
			payload={
				"temperature_c": parsed["temperature_c"],
				"humidity_rh": parsed["humidity_rh"],
				"temperature_unit": "C",
				"humidity_unit": "%RH",
				"raw": last_raw,
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
