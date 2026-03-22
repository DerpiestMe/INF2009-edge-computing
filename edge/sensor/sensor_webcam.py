import logging
import os
import time
from pathlib import Path
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
	import cv2
except ImportError:  # pragma: no cover - hardware/runtime dependency
	cv2 = None


class WebcamSensor:
	"""Owns raw webcam I/O and frame buffering."""

	def __init__(
		self,
		sensor_id: str = "webcam-01",
		device_index: Optional[int] = None,
		candidate_indices: Optional[List[int]] = None,
		width: int = 640,
		height: int = 480,
		target_fps: float = 10.0,
		stale_seconds: float = 3.0,
		buffer_size: int = 30,
		reconnect_backoff: float = 2.0,
	) -> None:
		self.sensor_type = "webcam"
		self.sensor_id = sensor_id
		self.device_index = device_index
		self.candidate_indices = candidate_indices or [0, 1, 2, 3, 4, 5]
		self.width = width
		self.height = height
		self.target_fps = target_fps
		self.stale_seconds = stale_seconds
		self.reconnect_backoff = reconnect_backoff

		self._logger = logging.getLogger(self.__class__.__name__)
		self._cap = None
		self._running = False
		self._sequence = 0
		self._last_read_ts = 0.0
		self._last_ok_ts = 0.0
		self._error_count = 0
		self._last_reconnect_attempt_ts = 0.0
		self._active_device_index: Optional[int] = None

		self._metadata_buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
		self._frame_buffer: Deque[Tuple[int, float, Any]] = deque(maxlen=buffer_size)

	def _candidate_device_indices(self) -> List[int]:
		if self.device_index is not None:
			return [self.device_index]

		indices: List[int] = []
		for dev in sorted(Path("/dev").glob("video*")):
			suffix = dev.name.replace("video", "")
			if suffix.isdigit():
				indices.append(int(suffix))

		if indices:
			return indices
		return list(self.candidate_indices)

	def _open_capture(self, index: int) -> Optional[Any]:
		if os.name != "nt":
			cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
		else:
			cap = cv2.VideoCapture(index)
		if not cap.isOpened():
			cap.release()
			return None

		# Demo-like low-latency settings for USB webcams on Linux.
		if hasattr(cv2, "CAP_PROP_FOURCC"):
			cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
		if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
		cap.set(cv2.CAP_PROP_FPS, self.target_fps)
		return cap

	def start(self) -> bool:
		if cv2 is None:
			self._logger.error("OpenCV is not installed. Install opencv-python.")
			return False

		for index in self._candidate_device_indices():
			cap = self._open_capture(index)
			if cap is None:
				continue

			self._cap = cap
			self._active_device_index = index
			self._running = True
			self._logger.info("Webcam started on device %s", index)
			return True

		self._cap = None
		self._active_device_index = None
		self._running = False
		self._logger.warning("Failed to open webcam on candidate devices: %s", self._candidate_device_indices())
		return False

	def stop(self) -> None:
		self._running = False
		if self._cap is not None:
			self._cap.release()
			self._cap = None
		self._active_device_index = None

	def _status(self) -> str:
		now = time.time()
		if not self._running:
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

	def read(self) -> Optional[Dict[str, Any]]:
		self._last_read_ts = time.time()

		if not self._running or self._cap is None:
			if not self.start():
				self._error_count += 1
				return self._build_record(
					status="offline",
					payload={"reason": "camera_unavailable", "device_index": self._active_device_index},
				)

		ok, frame = self._cap.read()
		if not ok or frame is None:
			self._error_count += 1
			self._maybe_reconnect()
			return self._build_record(
				status="degraded",
				payload={"reason": "frame_read_failed", "device_index": self._active_device_index},
			)

		self._sequence += 1
		self._last_ok_ts = time.time()
		height, width = frame.shape[:2]
		channels = frame.shape[2] if len(frame.shape) > 2 else 1

		metadata = self._build_record(
			status=self._status(),
			payload={
				"frame_seq": self._sequence,
				"width": width,
				"height": height,
				"channels": channels,
				"device_index": self._active_device_index,
			},
		)

		self._metadata_buffer.append(metadata)
		self._frame_buffer.append((self._sequence, self._last_ok_ts, frame))
		return metadata

	def get_latest_frame(self) -> Optional[Tuple[int, float, Any]]:
		if not self._frame_buffer:
			return None
		return self._frame_buffer[-1]

	def get_buffered_metadata(self) -> list:
		return list(self._metadata_buffer)

	def health_snapshot(self) -> Dict[str, Any]:
		return self._build_record(status=self._status(), payload={"heartbeat": True})


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	sensor = WebcamSensor()
	window_name = "USB Webcam Live"
	display_enabled = cv2 is not None

	if display_enabled:
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	try:
		while True:
			record = sensor.read()
			if record is not None:
				print(record)

			if display_enabled:
				latest = sensor.get_latest_frame()
				if latest is not None:
					_, _, frame = latest
					cv2.imshow(window_name, frame)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
			time.sleep(0.1)
	except KeyboardInterrupt:
		pass
	finally:
		if display_enabled:
			cv2.destroyAllWindows()
		sensor.stop()
