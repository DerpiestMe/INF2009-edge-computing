"""
cloud/pipeline/cloud_subscriber.py
====================================
Runs on YOUR LAPTOP (connected to PuppyPi hotspot).
Subscribes to the PuppyPi's local Mosquitto broker and handles:
  1. Storing telemetry to InfluxDB (already in docker-compose)
  2. Running YOLOv8-Large re-ID on intrusion snapshots
  3. Routing email/SMS alerts via alert_router.py

MQTT broker is the PuppyPi's own Mosquitto.
Connect your laptop to PuppyPi hotspot first, then run:
  python3 cloud/pipeline/cloud_subscriber.py

Default PuppyPi hotspot IP: 192.168.149.1
"""

import json
import base64
import time
import logging
import tempfile
import threading
from pathlib import Path
from queue import Queue, Empty

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Import your existing cloud modules (already written)
from alert_router import AlertRouter

log = logging.getLogger("cloud_subscriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
# PuppyPi hotspot IP — your laptop connects to PuppyPi WiFi
MQTT_BROKER_HOST = "192.168.149.1"   # PuppyPi default hotspot IP
MQTT_BROKER_PORT = 1883

# InfluxDB running on YOUR LAPTOP via docker-compose
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "your-influxdb-token"   # set during InfluxDB setup
INFLUX_ORG    = "puppypi"
INFLUX_BUCKET = "puppypi-data"

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
                .time(ts, WritePrecision.SECONDS)
            )
            self._write_api.write(bucket=INFLUX_BUCKET, record=point)

        if payload.get("temp_c") is not None:
            point = (
                Point("temperature_reading")
                .tag("device_id", payload.get("device_id", "puppypi-01"))
                .field("temp_c",   float(payload["temp_c"]))
                .field("humidity", float(payload.get("humidity", 0)))
                .time(ts, WritePrecision.SECONDS)
            )
            self._write_api.write(bucket=INFLUX_BUCKET, record=point)

        log.debug(f"Telemetry written to InfluxDB: {payload.get('gas_ppm')} ppm, {payload.get('temp_c')}C")

    def write_intrusion_event(self, payload: dict, reid_result: dict = None):
        """Write an intrusion event to InfluxDB."""
        point = (
            Point("intrusion_event")
            .tag("device_id", payload.get("device_id", "puppypi-01"))
            .tag("severity",  payload.get("severity", "INTRUDER"))
            .field("event_id",    payload.get("event_id", ""))
            .field("gas_ppm",     float(payload.get("gas_ppm") or 0))
            .field("temp_c",      float(payload.get("temp_c") or 0))
        )
        if reid_result:
            point = point.field("person_name",  reid_result.get("name", "Unknown"))
            point = point.field("reid_matched", reid_result.get("matched", False))
        self._write_api.write(bucket=INFLUX_BUCKET, record=point)


# ── YOLOv8 Large Re-ID ────────────────────────────────────────────────────────
class CloudVisionProcessor:
    """
    Processes intrusion snapshots with YOLOv8-Large for better accuracy.
    Runs in a background thread off the main MQTT loop to avoid blocking.
    """

    def __init__(self):
        self._job_queue = Queue()
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._model     = None

    def start(self):
        self._thread.start()
        log.info("CloudVisionProcessor started")

    def enqueue(self, event_id: str, snapshot_b64: str, payload: dict):
        """Add an intrusion event to the vision processing queue."""
        self._job_queue.put((event_id, snapshot_b64, payload))

    def _load_model(self):
        """Lazy-load YOLOv8-Large on first use."""
        if self._model is None:
            from ultralytics import YOLO
            log.info("Loading YOLOv8-Large model (first use)...")
            self._model = YOLO("yolov8l.pt")
            log.info("YOLOv8-Large ready")
        return self._model

    def _worker(self):
        """Background worker — processes one vision job at a time."""
        while True:
            try:
                event_id, snapshot_b64, payload = self._job_queue.get(timeout=5)
                self._process(event_id, snapshot_b64, payload)
                self._job_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                log.exception(f"Vision worker error: {e}")

    def _process(self, event_id: str, snapshot_b64: str, payload: dict) -> dict:
        """
        Decode snapshot, run YOLOv8-Large, return detection results.
        Uses the same snapshot that intrusion_events.py already saved cleanly.
        """
        import cv2
        import numpy as np

        try:
            img_bytes  = base64.b64decode(snapshot_b64)
            img_array  = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr    = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            model      = self._load_model()
            results    = model.predict(img_bgr, conf=0.4, classes=[0], verbose=False)

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": float(box.conf[0]),
                    })

            log.info(f"YOLOv8-Large: {len(detections)} person(s) in event {event_id}")

            # Save annotated image locally for dashboard
            annotated = img_bgr.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
                cv2.putText(annotated, f"{det['confidence']:.0%}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)

            out_path = Path(f"cloud_annotated/{event_id}_annotated.jpg")
            out_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out_path), annotated)

            return {"event_id": event_id, "detections": detections, "annotated": str(out_path)}

        except Exception as e:
            log.error(f"Vision processing failed for {event_id}: {e}")
            return {}


# ── Message Handlers ──────────────────────────────────────────────────────────
alert_router   = AlertRouter()
influx_writer  = InfluxWriter()
vision_proc    = CloudVisionProcessor()

def handle_telemetry(payload: dict):
    """Store telemetry to InfluxDB. Check gas threshold for alerts."""
    influx_writer.write_telemetry(payload)

    # Alert on gas threshold breach — uses your existing alert_router unchanged
    gas_ppm  = payload.get("gas_ppm", 0) or 0
    severity = payload.get("gas_severity", "NORMAL")

    if severity == "CRITICAL":
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
        alert_router.send_alert(
            alert_type = "HAZARD",
            subject    = "⚠️ Gas Warning — PuppyPi",
            message    = f"Gas level: {gas_ppm:.1f} PPM (WARNING)\nTime: {payload.get('timestamp')}",
            severity   = "MEDIUM",
            dedup_key  = "gas-warning",
        )


def handle_intrusion(payload: dict):
    """
    Handle intrusion event:
    1. Store to InfluxDB
    2. Enqueue for YOLOv8-Large processing (non-blocking)
    3. Send immediate alert
    """
    event_id    = payload.get("event_id", "unknown")
    snapshot_b64 = payload.get("snapshot_b64")

    # Store immediately to InfluxDB
    influx_writer.write_intrusion_event(payload)

    # Enqueue for heavy vision processing (runs in background thread)
    if snapshot_b64:
        vision_proc.enqueue(event_id, snapshot_b64, payload)
    else:
        log.warning(f"Intrusion event {event_id} has no snapshot attached")

    # Send alert immediately — don't wait for YOLOv8 result
    alert_router.send_alert(
        alert_type = "INTRUDER",
        subject    = "🚨 Intruder Detected — PuppyPi",
        message    = (
            f"Intrusion event ID: {event_id}\n"
            f"Time: {payload.get('timestamp')}\n"
            f"Gas at time of event: {payload.get('gas_ppm')} PPM\n"
            f"Temperature: {payload.get('temp_c')}°C"
        ),
        severity   = "HIGH",
        dedup_key  = event_id,
    )


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
    vision_proc.start()

    client = mqtt.Client(client_id="cloud-subscriber")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    log.info(f"Connecting to PuppyPi Mosquitto at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    log.info("Make sure your laptop is connected to the PuppyPi hotspot!")

    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
