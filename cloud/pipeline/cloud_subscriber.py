"""
cloud/pipeline/cloud_subscriber.py
====================================
Runs on YOUR LAPTOP (cloud side).
Subscribes to the MQTT broker running on your laptop and handles:
  1. Storing telemetry to InfluxDB (already in docker-compose)
  2. Running YOLOv8-Large re-ID on intrusion snapshots
  3. Routing email/SMS alerts via alert_router.py

MQTT broker is on your laptop (see docker-compose).
Run this on the laptop:
  python3 cloud/pipeline/cloud_subscriber.py
"""

import json
import base64
import time
import logging
import tempfile
import threading
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty

import paho.mqtt.client as mqtt
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Import your existing cloud modules (already written)
from alert_router import AlertRouter

log = logging.getLogger("cloud_subscriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
# MQTT broker runs on the laptop (cloud). Default to localhost.
# Override with env if you run the broker elsewhere.
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))

# InfluxDB running on YOUR LAPTOP via docker-compose
INFLUX_URL    = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN", "your-influxdb-token")   # set during InfluxDB setup
INFLUX_ORG    = os.getenv("INFLUX_ORG", "puppypi")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "puppypi-data")

# Where to store snapshots received from edge (base64 over MQTT)
CLOUD_SNAPSHOT_DIR = Path(os.getenv("CLOUD_SNAPSHOT_DIR", "cloud_snapshots"))
CLOUD_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Simple alert history persistence for dashboard
ALERTS_DB_PATH = Path(os.getenv("ALERTS_DB_PATH", "cloud/data/alerts.jsonl"))
ALERTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Whitelist re-identification settings
WHITELIST_DIR = Path(os.getenv("WHITELIST_DIR", "cloud/whitelist"))
REID_THRESHOLD = float(os.getenv("REID_THRESHOLD", "0.6"))

MQTT_PUBLISH_CLIENT = None

TOPICS = [
    ("puppypi/sensors/telemetry", 0),
    ("puppypi/events/intrusion",  1),
    ("puppypi/status/heartbeat",  0),
]

# ── InfluxDB Writer ───────────────────────────────────────────────────────────
class InfluxWriter:
    """Writes telemetry and events to InfluxDB for Grafana dashboards."""

    def __init__(self):
        self._client   = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def write_telemetry(self, payload: dict):
        """Write gas + temp readings as InfluxDB measurements."""
        ts = payload.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ"))

        if payload.get("gas_ppm") is not None:
            point = (
                Point("gas_reading")
                .tag("device_id", payload.get("device_id", "puppypi-01"))
                .field("ppm", float(payload["gas_ppm"]))
                .field("severity", payload.get("gas_severity", "NORMAL"))
                .time(ts, WritePrecision.S)
            )
            self._write_api.write(bucket=INFLUX_BUCKET, record=point)

        if payload.get("temp_c") is not None:
            point = (
                Point("temperature_reading")
                .tag("device_id", payload.get("device_id", "puppypi-01"))
                .field("temp_c",   float(payload["temp_c"]))
                .field("humidity", float(payload.get("humidity", 0)))
                .time(ts, WritePrecision.S)
            )
            self._write_api.write(bucket=INFLUX_BUCKET, record=point)

        log.debug(f"Telemetry written to InfluxDB: {payload.get('gas_ppm')} ppm, {payload.get('temp_c')}C")

    def write_intrusion_event(self, payload: dict, reid_result: dict = None, snapshot_path: str = None):
        """Write an intrusion event to InfluxDB."""
        point = (
            Point("intrusion_event")
            .tag("device_id", payload.get("device_id", "puppypi-01"))
            .tag("severity",  payload.get("severity", "INTRUDER"))
            .field("event_id",    payload.get("event_id", ""))
            .field("gas_ppm",     float(payload.get("gas_ppm") or 0))
            .field("temp_c",      float(payload.get("temp_c") or 0))
        )
        if snapshot_path:
            point = point.field("snapshot_path", snapshot_path)
        if reid_result:
            point = point.field("person_name",  reid_result.get("name", "Unknown"))
            point = point.field("reid_matched", reid_result.get("matched", False))
        self._write_api.write(bucket=INFLUX_BUCKET, record=point)


def _parse_ts_epoch(value) -> int:
    if value is None:
        return int(time.time())
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return int(time.time())
    return int(time.time())


def _append_alert_record(record: dict) -> None:
    try:
        record = dict(record)
        record.setdefault("id", record.get("event_id") or f"{record.get('type','event')}_{int(record.get('ts') or time.time())}")
        with open(ALERTS_DB_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.error(f"Failed to persist alert record: {e}")


# ── Whitelist Re-ID ───────────────────────────────────────────────────────────
class WhitelistReID:
    """Simple whitelist matcher for face images."""

    def __init__(self, whitelist_dir: Path, threshold: float = 0.6):
        self.whitelist_dir = Path(whitelist_dir)
        self.threshold = float(threshold)
        self._use_face_recognition = False
        self._face_recognition = None
        self._embeddings = []
        self._load_backend()
        self._load_whitelist()

    def _load_backend(self):
        try:
            import face_recognition  # type: ignore
            self._face_recognition = face_recognition
            self._use_face_recognition = True
            log.info("Using face_recognition backend for whitelist matching")
        except Exception:
            self._use_face_recognition = False
            log.warning("face_recognition not available; using fallback embedding")

    def _embed(self, img_bgr):
        import cv2

        if self._use_face_recognition and self._face_recognition:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes = self._face_recognition.face_locations(rgb)
            encodings = self._face_recognition.face_encodings(rgb, boxes)
            if not encodings:
                return None, []
            return np.array(encodings[0], dtype=np.float32), boxes

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        vec = small.astype(np.float32).flatten()
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm, []

    def _load_whitelist(self):
        self._embeddings = []
        if not self.whitelist_dir.exists():
            log.warning("Whitelist directory missing: %s", self.whitelist_dir)
            return
        for path in sorted(self.whitelist_dir.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is None:
                    continue
                emb = self._embed(img)
                if emb is None:
                    log.warning("No face found in whitelist image: %s", path.name)
                    continue
                self._embeddings.append((path.stem, emb))
            except Exception as e:
                log.warning("Failed to load whitelist image %s: %s", path.name, e)
        log.info("Whitelist loaded: %s identities", len(self._embeddings))

    def match(self, img_bgr):
        if not self._embeddings:
            return {"matched": False, "name": "Unknown", "score": 0.0}

        emb, boxes = self._embed(img_bgr)
        if emb is None:
            return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": boxes}

        best_name = "Unknown"
        best_score = 0.0

        if self._use_face_recognition and self._face_recognition is not None:
            known = [e for _, e in self._embeddings]
            distances = self._face_recognition.face_distance(known, emb)
            if len(distances) == 0:
                return {"matched": False, "name": "Unknown", "score": 0.0, "face_locations": boxes}
            best_idx = int(np.argmin(distances))
            best_name = self._embeddings[best_idx][0]
            best_score = 1.0 - float(distances[best_idx])
            matched = distances[best_idx] <= self.threshold
            return {"matched": matched, "name": best_name if matched else "Unknown", "score": best_score, "face_locations": boxes}

        # Fallback cosine similarity
        for name, known in self._embeddings:
            score = float(np.dot(emb, known))
            if score > best_score:
                best_score = score
                best_name = name
        matched = best_score >= max(0.85, self.threshold)
        return {"matched": matched, "name": best_name if matched else "Unknown", "score": best_score, "face_locations": boxes}


class ReIDProcessor:
    """Processes snapshots and classifies them as authorized/unauthorized."""

    def __init__(self, whitelist_dir: Path, threshold: float, on_result):
        self._job_queue = Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._reid = WhitelistReID(whitelist_dir=whitelist_dir, threshold=threshold)
        self._on_result = on_result

    def start(self):
        self._thread.start()
        log.info("ReIDProcessor started")

    def enqueue(self, event_id: str, snapshot_b64: str, payload: dict, snapshot_path: str):
        self._job_queue.put((event_id, snapshot_b64, payload, snapshot_path))

    def _worker(self):
        while True:
            try:
                event_id, snapshot_b64, payload, snapshot_path = self._job_queue.get(timeout=5)
                self._process(event_id, snapshot_b64, payload, snapshot_path)
                self._job_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                log.exception(f"ReID worker error: {e}")

    def _process(self, event_id: str, snapshot_b64: str, payload: dict, snapshot_path: str) -> dict:
        import cv2

        try:
            img_bytes = base64.b64decode(snapshot_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            result = self._reid.match(img_bgr)
            result.update({"event_id": event_id, "snapshot_path": snapshot_path})
            self._on_result(payload, result)
            return result
        except Exception as e:
            log.error(f"ReID processing failed for {event_id}: {e}")
            return {}


# ── Message Handlers ──────────────────────────────────────────────────────────
alert_router   = AlertRouter()
influx_writer  = InfluxWriter()
reid_proc      = ReIDProcessor(whitelist_dir=WHITELIST_DIR, threshold=REID_THRESHOLD, on_result=None)

def handle_telemetry(payload: dict):
    """Store telemetry to InfluxDB. Check gas threshold for alerts."""
    influx_writer.write_telemetry(payload)

    # Alert on gas threshold breach — uses your existing alert_router unchanged
    gas_ppm  = payload.get("gas_ppm", 0) or 0
    severity = payload.get("gas_severity", "NORMAL")

    if severity == "CRITICAL":
        _append_alert_record({
            "type": "GAS_HIGH",
            "severity": "critical",
            "ts": _parse_ts_epoch(payload.get("timestamp")),
            "msg": f"Gas reading {gas_ppm:.1f} ppm — CRITICAL",
        })
        alert_router.send_alert(
            alert_type = "HAZARD",
            subject    = "☣️ CRITICAL Gas Level — PuppyPi",
            message    = (
                f"Gas level: {gas_ppm:.1f} PPM (CRITICAL)\n"
                f"Temperature: {payload.get('temp_c')}°C\n"
                f"Time: {payload.get('timestamp')}"
            ),
            severity   = "CRITICAL",
            dedup_key  = "gas-critical",
        )
    elif severity == "WARNING":
        _append_alert_record({
            "type": "GAS_HIGH",
            "severity": "warning",
            "ts": _parse_ts_epoch(payload.get("timestamp")),
            "msg": f"Gas reading {gas_ppm:.1f} ppm — WARNING",
        })
        alert_router.send_alert(
            alert_type = "HAZARD",
            subject    = "⚠️ Gas Warning — PuppyPi",
            message    = f"Gas level: {gas_ppm:.1f} PPM (WARNING)\nTime: {payload.get('timestamp')}",
            severity   = "MEDIUM",
            dedup_key  = "gas-warning",
        )


def _publish_reid_result(event: dict) -> None:
    if MQTT_PUBLISH_CLIENT is None:
        return
    try:
        MQTT_PUBLISH_CLIENT.publish("puppypi/events/reid", json.dumps(event), qos=0)
    except Exception as e:
        log.warning(f"Failed to publish reid result: {e}")


def _handle_reid_result(payload: dict, result: dict) -> None:
    matched = bool(result.get("matched"))
    name = result.get("name", "Unknown")
    score = float(result.get("score", 0.0))
    face_locations = result.get("face_locations") or []
    event_id = result.get("event_id") or payload.get("event_id", "unknown")
    snapshot_path = result.get("snapshot_path")
    ts = _parse_ts_epoch(payload.get("timestamp"))

    influx_writer.write_intrusion_event(payload, reid_result=result, snapshot_path=snapshot_path)

    alert_type = "AUTHORIZED" if matched else "UNAUTHORIZED"
    severity = "info" if matched else "critical"
    msg = f"{alert_type}: {name} (score: {score:.2f})"
    _append_alert_record({
        "type": alert_type,
        "severity": severity,
        "ts": ts,
        "msg": msg,
        "event_id": event_id,
        "snapshot_path": snapshot_path,
        "snapshot_filename": Path(snapshot_path).name if snapshot_path else None,
        "face_locations": face_locations,
    })

    _publish_reid_result({
        "event_id": event_id,
        "authorized": matched,
        "name": name,
        "score": score,
        "timestamp": payload.get("timestamp"),
        "snapshot_path": snapshot_path,
        "face_locations": face_locations,
    })

    if not matched:
        alert_router.send_alert(
            alert_type = "INTRUDER",
            subject    = "🚨 Unauthorized Person Detected — PuppyPi",
            message    = (
                f"Intrusion event ID: {event_id}\n"
                f"Matched: {name}\n"
                f"Score: {score:.2f}\n"
                f"Time: {payload.get('timestamp')}"
            ),
            severity   = "HIGH",
            dedup_key  = event_id,
        )


def handle_intrusion(payload: dict):
    """
    Handle intrusion event:
    1. Store to InfluxDB
    2. Enqueue for YOLOv8-Large processing (non-blocking)
    3. Send immediate alert
    """
    event_id     = payload.get("event_id", "unknown")
    snapshot_b64 = payload.get("snapshot_b64")
    snapshot_path = None

    if snapshot_b64:
        safe_event_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(event_id))
        filename = f"{safe_event_id}_{int(time.time())}.jpg"
        snapshot_path = str(CLOUD_SNAPSHOT_DIR / filename)
        try:
            img_bytes = base64.b64decode(snapshot_b64)
            with open(snapshot_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            log.error(f"Failed to save snapshot for {event_id}: {e}")
            snapshot_path = None

    # Enqueue for whitelist re-identification (runs in background thread)
    if snapshot_b64:
        reid_proc.enqueue(event_id, snapshot_b64, payload, snapshot_path)
    else:
        log.warning(f"Motion event {event_id} has no snapshot attached")


def handle_heartbeat(payload: dict):
    log.info(f"Heartbeat: state={payload.get('state')}, uptime={payload.get('uptime_sec')}s")


TOPIC_HANDLERS = {
    "puppypi/sensors/telemetry": handle_telemetry,
    "puppypi/events/intrusion":  handle_intrusion,
    "puppypi/status/heartbeat":  handle_heartbeat,
}


# ── MQTT Callbacks ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"Connected to PuppyPi Mosquitto at {MQTT_BROKER_HOST}")
        for topic, qos in TOPICS:
            client.subscribe(topic, qos)
            log.info(f"Subscribed: {topic}")
    else:
        log.error(f"Connection failed rc={rc} — is your laptop on the PuppyPi hotspot?")

def on_message(client, userdata, msg):
    try:
        payload  = json.loads(msg.payload.decode("utf-8"))
        handler  = TOPIC_HANDLERS.get(msg.topic)
        if handler:
            handler(payload)
        else:
            log.warning(f"No handler for topic: {msg.topic}")
    except Exception as e:
        log.exception(f"Error handling {msg.topic}: {e}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.warning(f"Disconnected from broker (rc={rc}), reconnecting...")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MQTT_PUBLISH_CLIENT
    reid_proc._on_result = _handle_reid_result
    reid_proc.start()

    client = mqtt.Client(client_id="cloud-subscriber")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect
    MQTT_PUBLISH_CLIENT = client

    log.info(f"Connecting to PuppyPi Mosquitto at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    log.info("Make sure your laptop is connected to the PuppyPi hotspot!")

    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
