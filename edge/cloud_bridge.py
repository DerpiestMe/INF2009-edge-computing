"""
edge/cloud_bridge.py
====================
Wraps PuppyPiStartupApp with MQTT publishing.
Does NOT modify any existing functions.

Publishes to the LOCAL Mosquitto broker running on the PuppyPi (port 1883).
The cloud subscriber (on your laptop) connects to this same broker
via the PuppyPi hotspot IP (192.168.149.1:1883).

Run instead of main_puppypi_startup.py:
  python3 edge/cloud_bridge.py --camera-index 2 --headless
"""

import json
import time
import uuid
import base64
import logging
import threading
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paho.mqtt.client as mqtt
from main_puppypi_startup import PuppyPiStartupApp, parse_args

log = logging.getLogger("cloud_bridge")

# ── Config ────────────────────────────────────────────────────────────────────
# This is the local Mosquitto broker running ON THE PUPPYPI itself
MQTT_BROKER_HOST     = "localhost"   # we're running ON the PuppyPi
MQTT_BROKER_PORT     = 1883
TELEMETRY_INTERVAL_S = 5.0          # publish sensor readings every 5s
SNAPSHOT_DIR         = Path("snapshots")  # where intrusion_events saves clean frames

TOPIC_TELEMETRY  = "puppypi/sensors/telemetry"
TOPIC_INTRUSION  = "puppypi/events/intrusion"
TOPIC_HEARTBEAT  = "puppypi/status/heartbeat"

# ── MQTT Publisher Thread ─────────────────────────────────────────────────────
class EdgeMQTTPublisher:
    """
    Background thread that reads data from a running PuppyPiStartupApp
    and publishes it to the local Mosquitto broker.

    Does NOT modify any function in PuppyPiStartupApp — only reads
    its public/semi-public attributes.
    """

    def __init__(self, app: PuppyPiStartupApp):
        self._app     = app
        self._client  = mqtt.Client(client_id="puppypi-edge-publisher")
        self._running = False
        self._thread  = None
        self._seen_snapshots: set = set()   # track already-published snapshots
        self._start_time = time.time()

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info("EdgeMQTTPublisher connected to local Mosquitto broker")
        else:
            log.error(f"MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            log.warning("MQTT disconnected unexpectedly, will retry...")

    def start(self):
        try:
            self._client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
            self._client.loop_start()   # non-blocking network loop
        except Exception as e:
            log.error(f"Cannot connect to local MQTT broker: {e}")
            log.error("Is Mosquitto running? Check: docker-compose up mosquitto")
            return

        self._running = True
        self._thread  = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        log.info("EdgeMQTTPublisher started")

    def stop(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()

    def _publish_loop(self):
        last_telemetry = 0.0
        last_heartbeat = 0.0

        while self._running:
            now = time.time()

            # ── Telemetry (every 5s) ──────────────────────────────────────────
            if now - last_telemetry >= TELEMETRY_INTERVAL_S:
                self._publish_telemetry()
                last_telemetry = now

            # ── Heartbeat (every 30s) ─────────────────────────────────────────
            if now - last_heartbeat >= 30.0:
                self._publish_heartbeat()
                last_heartbeat = now

            # ── Intrusion snapshots (poll snapshot dir for new files) ─────────
            self._check_and_publish_snapshots()

            time.sleep(1.0)

    def _publish_telemetry(self):
        """
        Read _last_temp_record and _last_gas_record from the running app
        and publish as a single telemetry message.
        """
        temp = self._app._last_temp_record
        gas  = self._app._last_gas_record

        payload = {
            "event_id":  str(uuid.uuid4()),
            "device_id": "puppypi-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Extract temp fields if available
        if temp and "payload" in temp:
            tp = temp["payload"]
            payload["temp_c"]    = tp.get("temperature_c")
            payload["humidity"]  = tp.get("humidity_rh")

        # Extract gas fields if available
        if gas and "payload" in gas:
            gp = gas["payload"]
            payload["gas_ppm"]   = gp.get("ppm")

            # Basic threshold classification (edge-side only)
            ppm = gp.get("ppm", 0)
            if ppm > 500:
                payload["gas_severity"] = "CRITICAL"
            elif ppm > 200:
                payload["gas_severity"] = "WARNING"
            else:
                payload["gas_severity"] = "NORMAL"

        self._client.publish(
            TOPIC_TELEMETRY,
            json.dumps(payload),
            qos=0,   # fire-and-forget for high-freq telemetry
        )
        log.debug(f"Telemetry published: gas={payload.get('gas_ppm')} ppm, temp={payload.get('temp_c')}C")

    def _check_and_publish_snapshots(self):
        """
        Poll the snapshot directory for new .jpg files saved by intrusion_events.py.
        For each new file, publish an intrusion event with the image as base64.

        This hooks into intrusion_events output WITHOUT modifying that module.
        """
        if not SNAPSHOT_DIR.exists():
            return

        for snap_path in sorted(SNAPSHOT_DIR.glob("*.jpg")):
            if snap_path.name in self._seen_snapshots:
                continue

            self._seen_snapshots.add(snap_path.name)

            try:
                with open(snap_path, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")

                event_id = snap_path.stem   # filename without .jpg is the event ID

                payload = {
                    "event_id":    event_id,
                    "device_id":   "puppypi-01",
                    "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "severity":    "INTRUDER",
                    "snapshot_b64": b64_image,
                    "snapshot_path": str(snap_path),
                    # Include latest sensor readings at time of intrusion
                    "gas_ppm":  self._app._last_gas_record.get("payload", {}).get("ppm") if self._app._last_gas_record else None,
                    "temp_c":   self._app._last_temp_record.get("payload", {}).get("temperature_c") if self._app._last_temp_record else None,
                }

                self._client.publish(
                    TOPIC_INTRUSION,
                    json.dumps(payload),
                    qos=1,   # at-least-once for critical events
                )
                log.info(f"Intrusion event published: {event_id}")

            except Exception as e:
                log.error(f"Failed to publish snapshot {snap_path}: {e}")

    def _publish_heartbeat(self):
        """Publish robot status heartbeat."""
        payload = {
            "device_id":  "puppypi-01",
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state":      "RUNNING",
            "uptime_sec": int(time.time() - self._start_time),
        }
        self._client.publish(TOPIC_HEARTBEAT, json.dumps(payload), qos=0)


# ── Main ──────────────────────────────────────────────────────────────────────
class CloudBridgeApp(PuppyPiStartupApp):
    """
    PuppyPiStartupApp + MQTT publishing side-car.
    Inherits everything from PuppyPiStartupApp unchanged.
    Only adds: start/stop the EdgeMQTTPublisher alongside the existing lifecycle.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._publisher = EdgeMQTTPublisher(app=self)

    def start(self) -> None:
        super().start()             # starts sensors, servo — unchanged
        self._publisher.start()     # also start MQTT publisher

    def stop(self) -> None:
        self._publisher.stop()      # stop publisher first
        super().stop()              # then stop sensors — unchanged


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    app = CloudBridgeApp(
        camera_index=args.camera_index,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        model_path=args.model_path,
        person_confidence=args.confidence,
        gas_ports=args.gas_ports,
        servo_channel=args.servo_channel,
        camera_start_servo_pulse=args.camera_start_servo_pulse,
        show_window=not args.headless,
    )

    app.run_forever()


if __name__ == "__main__":
    main()