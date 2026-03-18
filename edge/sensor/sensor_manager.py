import logging
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from sensor_gas import GasSensor
from sensor_temp import TemperatureHumiditySensor
from sensor_webcam import WebcamSensor


@dataclass
class SensorSchedule:
	sensor: Any
	poll_interval: float
	next_due_ts: float


class SensorManager:
	"""Orchestrates raw sensor I/O, normalization, buffering, and health watchdogs."""

	def __init__(self, buffer_size: int = 500, health_log_interval: float = 5.0) -> None:
		self._logger = logging.getLogger(self.__class__.__name__)
		self._running = False
		self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
		self._latest_by_sensor: Dict[str, Dict[str, Any]] = {}
		self._health_log_interval = health_log_interval
		self._last_health_log_ts = 0.0

		now = time.time()
		self._schedules: Dict[str, SensorSchedule] = {
			"webcam": SensorSchedule(
				sensor=WebcamSensor(
					sensor_id="webcam-01",
					device_index=0,
					width=640,
					height=480,
					target_fps=10.0,
					stale_seconds=3.0,
					buffer_size=30,
				),
				poll_interval=0.1,
				next_due_ts=now,
			),
			"gas": SensorSchedule(
				sensor=GasSensor(
					sensor_id="gas-01",
					candidate_ports=["/dev/ttyACM0", "/dev/ttyUSB0", "COM3", "COM4"],
					stale_seconds=5.0,
					buffer_size=100,
				),
				poll_interval=0.1,
				next_due_ts=now,
			),
			"temperature_humidity": SensorSchedule(
				sensor=TemperatureHumiditySensor(
					sensor_id="temp-01",
					i2c_bus_index=1,
					i2c_address=0x38,
					stale_seconds=5.0,
					buffer_size=100,
				),
				poll_interval=1.0,
				next_due_ts=now,
			),
		}

	def start(self) -> None:
		self._running = True
		self._logger.info("Starting sensor manager")
		for schedule in self._schedules.values():
			schedule.sensor.start()

	def stop(self) -> None:
		self._running = False
		self._logger.info("Stopping sensor manager")
		for schedule in self._schedules.values():
			schedule.sensor.stop()

	def _ingest_record(self, record: Dict[str, Any]) -> None:
		key = f"{record['sensor_type']}:{record['sensor_id']}"
		self._buffer.append(record)
		self._latest_by_sensor[key] = record

	def get_latest_snapshot(self) -> Dict[str, Dict[str, Any]]:
		return dict(self._latest_by_sensor)

	def get_buffered_records(self) -> list:
		return list(self._buffer)

	def _poll_sensor(self, schedule: SensorSchedule) -> None:
		record = schedule.sensor.read()
		if record is not None:
			self._ingest_record(record)

	def _log_health_heartbeat(self) -> None:
		now = time.time()
		if now - self._last_health_log_ts < self._health_log_interval:
			return

		self._last_health_log_ts = now
		health_summary = {}
		for name, schedule in self._schedules.items():
			health_record = schedule.sensor.health_snapshot()
			health_summary[name] = {
				"status": health_record["status"],
				"health": health_record["health"],
			}
		self._logger.info("Sensor health: %s", health_summary)

	def run_forever(self) -> None:
		self.start()
		try:
			while self._running:
				now = time.time()
				earliest_next_due = now + 0.5

				for schedule in self._schedules.values():
					if now >= schedule.next_due_ts:
						self._poll_sensor(schedule)
						schedule.next_due_ts = now + schedule.poll_interval
					earliest_next_due = min(earliest_next_due, schedule.next_due_ts)

				self._log_health_heartbeat()

				sleep_for = max(0.01, earliest_next_due - time.time())
				time.sleep(min(0.1, sleep_for))
		finally:
			self.stop()


def _configure_logging() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
	_configure_logging()
	manager = SensorManager()

	def _handle_shutdown(signum: int, _frame: Optional[Any]) -> None:
		logging.getLogger("SensorManager").info("Received signal %s, shutting down", signum)
		manager.stop()

	signal.signal(signal.SIGINT, _handle_shutdown)
	signal.signal(signal.SIGTERM, _handle_shutdown)

	manager.run_forever()


if __name__ == "__main__":
	main()
