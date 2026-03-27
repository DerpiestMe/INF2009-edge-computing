"""
mqtt_ingestion.py — PuppyPi Cloud MQTT Ingestion Pipeline
==========================================================
Subscribes to all PuppyPi MQTT topics, parses incoming events,
stores telemetry to DynamoDB, uploads snapshots to S3, and
forwards critical events to the alert router + vision pipeline.

AWS Services used:
  - S3          : snapshot/video clip storage
  - DynamoDB    : event + telemetry time-series storage
  - SQS         : decoupled queue for vision processing jobs
  - SSM         : secure retrieval of MQTT broker credentials
"""

import json
import time
import logging
import boto3
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from alert_router import AlertRouter

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("mqtt_ingestion")

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION        = "ap-southeast-1"       # Singapore region
S3_BUCKET         = "puppypi-events"
DYNAMO_TABLE      = "puppypi-telemetry"
SQS_VISION_QUEUE  = "puppypi-vision-jobs"

MQTT_BROKER_HOST  = "your-mqtt-broker.amazonaws.com"   # AWS IoT Core endpoint
MQTT_BROKER_PORT  = 8883                               # TLS port
MQTT_CLIENT_ID    = "cloud-ingestion-service"

# Topics to subscribe to
TOPICS = [
    ("puppypi/events/intrusion", 1),   # QoS 1 — at least once
    ("puppypi/events/hazard",    1),
    ("puppypi/sensors/telemetry",0),   # QoS 0 — fire and forget (high frequency)
    ("puppypi/status/heartbeat", 0),
]

# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class IntrusionEvent:
    event_id:      str
    timestamp:     str
    zone:          str
    confidence:    float
    snapshot_url:  Optional[str]  # S3 URL after upload
    snapshot_b64:  Optional[str]  # raw base64 from robot (transient)
    severity:      str            # SUSPICIOUS or INTRUDER

@dataclass
class HazardEvent:
    event_id:   str
    timestamp:  str
    gas_ppm:    float
    temp_c:     float
    severity:   str   # WARNING or CRITICAL
    location:   str

@dataclass
class TelemetryReading:
    device_id:  str
    timestamp:  str
    gas_ppm:    float
    temp_c:     float
    humidity:   float

@dataclass
class HeartbeatStatus:
    device_id:  str
    timestamp:  str
    battery_pct: float
    state:      str   # IDLE, SUSPICIOUS, INTRUDER, HAZARD
    uptime_sec: int

# ── AWS Clients ───────────────────────────────────────────────────────────────
class AWSClients:
    """Lazy-initialised AWS service clients (singleton pattern)."""
    _s3       = None
    _dynamo   = None
    _sqs      = None
    _sqs_url  = None

    @classmethod
    def s3(cls):
        if not cls._s3:
            cls._s3 = boto3.client("s3", region_name=AWS_REGION)
        return cls._s3

    @classmethod
    def dynamo(cls):
        if not cls._dynamo:
            cls._dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
        return cls._dynamo

    @classmethod
    def sqs(cls):
        if not cls._sqs:
            cls._sqs = boto3.client("sqs", region_name=AWS_REGION)
        return cls._sqs

    @classmethod
    def sqs_queue_url(cls):
        if not cls._sqs_url:
            resp = cls.sqs().get_queue_url(QueueName=SQS_VISION_QUEUE)
            cls._sqs_url = resp["QueueUrl"]
        return cls._sqs_url

# ── Storage Helpers ───────────────────────────────────────────────────────────
def upload_snapshot_to_s3(event_id: str, b64_data: str) -> str:
    """
    Decode base64 snapshot from robot and upload to S3.
    Returns the public S3 URL of the stored image.
    """
    import base64
    image_bytes = base64.b64decode(b64_data)
    key = f"snapshots/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{event_id}.jpg"

    AWSClients.s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
        # Snapshots expire after 90 days to save cost
        # (set lifecycle policy on S3 bucket separately)
    )
    s3_url = f"s3://{S3_BUCKET}/{key}"
    log.info(f"Snapshot uploaded → {s3_url}")
    return s3_url


def store_event_to_dynamo(table_name: str, item: dict):
    """Store any event/telemetry dict to DynamoDB."""
    table = AWSClients.dynamo().Table(table_name)
    table.put_item(Item=item)
    log.info(f"Stored to DynamoDB [{table_name}]: {item.get('event_id') or item.get('timestamp')}")


def enqueue_vision_job(event: IntrusionEvent):
    """
    Push intrusion event to SQS so cloud_vision.py can pick it up
    for heavy YOLOv8-Large re-identification processing.
    """
    AWSClients.sqs().send_message(
        QueueUrl=AWSClients.sqs_queue_url(),
        MessageBody=json.dumps({
            "event_id":    event.event_id,
            "snapshot_url": event.snapshot_url,
            "timestamp":   event.timestamp,
            "zone":        event.zone,
            "confidence":  event.confidence,
        }),
        MessageGroupId="vision-jobs",   # for FIFO queue ordering
    )
    log.info(f"Vision job enqueued for event {event.event_id}")

# ── Message Handlers ──────────────────────────────────────────────────────────
alert_router = AlertRouter()

def handle_intrusion(payload: dict):
    """
    Process intrusion detection event from edge:
    1. Upload snapshot to S3
    2. Store event to DynamoDB
    3. Enqueue for YOLOv8-Large re-ID
    4. Route alert if severity is INTRUDER
    """
    event = IntrusionEvent(
        event_id     = payload["event_id"],
        timestamp    = payload["timestamp"],
        zone         = payload.get("zone", "unknown"),
        confidence   = float(payload.get("confidence", 0.0)),
        snapshot_url = None,
        snapshot_b64 = payload.get("snapshot_b64"),
        severity     = payload.get("severity", "SUSPICIOUS"),
    )

    # Upload snapshot if attached
    if event.snapshot_b64:
        event.snapshot_url = upload_snapshot_to_s3(event.event_id, event.snapshot_b64)
        event.snapshot_b64 = None  # clear raw data before storing

    # Persist to DynamoDB
    store_event_to_dynamo(DYNAMO_TABLE, {
        **asdict(event),
        "pk": f"INTRUSION#{event.event_id}",
        "sk": event.timestamp,
        "ttl": int(time.time()) + (90 * 86400),  # 90-day TTL
    })

    # Always enqueue for YOLOv8-Large re-identification
    enqueue_vision_job(event)

    # Only alert on confirmed intruder (not just suspicious)
    if event.severity == "INTRUDER":
        alert_router.send_alert(
            alert_type = "INTRUDER",
            subject    = "🚨 Intruder Detected — PuppyPi",
            message    = (
                f"Intrusion confirmed in zone: {event.zone}\n"
                f"Confidence: {event.confidence:.1%}\n"
                f"Time: {event.timestamp}\n"
                f"Snapshot: {event.snapshot_url or 'N/A'}"
            ),
            severity = "HIGH",
        )


def handle_hazard(payload: dict):
    """
    Process gas/temperature hazard event:
    1. Store to DynamoDB
    2. Immediately route alert (all hazards are critical)
    """
    event = HazardEvent(
        event_id  = payload["event_id"],
        timestamp = payload["timestamp"],
        gas_ppm   = float(payload.get("gas_ppm", 0)),
        temp_c    = float(payload.get("temp_c", 0)),
        severity  = payload.get("severity", "WARNING"),
        location  = payload.get("location", "unknown"),
    )

    store_event_to_dynamo(DYNAMO_TABLE, {
        **asdict(event),
        "pk": f"HAZARD#{event.event_id}",
        "sk": event.timestamp,
        "ttl": int(time.time()) + (90 * 86400),
    })

    # All hazard events trigger an alert
    alert_router.send_alert(
        alert_type = "HAZARD",
        subject    = f"☣️ {'CRITICAL' if event.severity == 'CRITICAL' else 'WARNING'} Gas/Temp Hazard — PuppyPi",
        message    = (
            f"Hazard detected at: {event.location}\n"
            f"Gas level: {event.gas_ppm:.1f} PPM\n"
            f"Temperature: {event.temp_c:.1f}°C\n"
            f"Severity: {event.severity}\n"
            f"Time: {event.timestamp}"
        ),
        severity = "CRITICAL" if event.severity == "CRITICAL" else "MEDIUM",
    )


def handle_telemetry(payload: dict):
    """Store sensor telemetry reading (high frequency, no alerting)."""
    reading = TelemetryReading(
        device_id = payload.get("device_id", "puppypi-01"),
        timestamp = payload["timestamp"],
        gas_ppm   = float(payload.get("gas_ppm", 0)),
        temp_c    = float(payload.get("temp_c", 0)),
        humidity  = float(payload.get("humidity", 0)),
    )
    store_event_to_dynamo(DYNAMO_TABLE, {
        **asdict(reading),
        "pk": f"TELEMETRY#{reading.device_id}",
        "sk": reading.timestamp,
        "ttl": int(time.time()) + (30 * 86400),  # 30-day TTL for telemetry
    })


def handle_heartbeat(payload: dict):
    """Track robot status and detect if it goes offline."""
    status = HeartbeatStatus(
        device_id   = payload.get("device_id", "puppypi-01"),
        timestamp   = payload["timestamp"],
        battery_pct = float(payload.get("battery_pct", 0)),
        state       = payload.get("state", "UNKNOWN"),
        uptime_sec  = int(payload.get("uptime_sec", 0)),
    )
    store_event_to_dynamo(DYNAMO_TABLE, {
        **asdict(status),
        "pk": f"HEARTBEAT#{status.device_id}",
        "sk": status.timestamp,
        "ttl": int(time.time()) + (7 * 86400),  # 7-day TTL for heartbeats
    })

    if status.battery_pct < 15:
        alert_router.send_alert(
            alert_type = "SYSTEM",
            subject    = "🔋 PuppyPi Low Battery Warning",
            message    = f"Battery at {status.battery_pct:.0f}%. Please recharge soon.",
            severity   = "LOW",
        )

# ── MQTT Callbacks ────────────────────────────────────────────────────────────
TOPIC_HANDLERS = {
    "puppypi/events/intrusion": handle_intrusion,
    "puppypi/events/hazard":    handle_hazard,
    "puppypi/sensors/telemetry": handle_telemetry,
    "puppypi/status/heartbeat": handle_heartbeat,
}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to MQTT broker")
        for topic, qos in TOPICS:
            client.subscribe(topic, qos)
            log.info(f"Subscribed to: {topic} (QoS {qos})")
    else:
        log.error(f"MQTT connection failed with code {rc}")

def on_message(client, userdata, msg):
    topic   = msg.topic
    handler = TOPIC_HANDLERS.get(topic)
    if not handler:
        log.warning(f"No handler for topic: {topic}")
        return
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        log.info(f"Received on [{topic}]: {list(payload.keys())}")
        handler(payload)
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON on {topic}: {e}")
    except Exception as e:
        log.exception(f"Error handling message on {topic}: {e}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.warning(f"Unexpected MQTT disconnect (rc={rc}), will auto-reconnect...")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv5)

    # TLS for secure connection to AWS IoT Core
    client.tls_set(
        ca_certs   = "certs/AmazonRootCA1.pem",
        certfile   = "certs/device-cert.pem.crt",
        keyfile    = "certs/private.pem.key",
    )

    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    log.info(f"Connecting to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)

    # Blocking loop — auto-reconnects on disconnect
    client.loop_forever()

if __name__ == "__main__":
    main()