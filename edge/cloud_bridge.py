"""
edge/cloud_bridge.py
====================
Wraps FullEdgePipelineApp with MQTT publishing.
Does NOT modify any existing functions.

Publishes to the Mosquitto broker running on your LAPTOP (cloud side).
Set MQTT_BROKER_HOST to the laptop's hotspot/LAN IP if needed.

Run on the PuppyPi instead of edge/run_edge_full_pipeline.py:
  MQTT_BROKER_HOST=192.168.x.x python3 edge/cloud_bridge.py --camera-index 2 --headless
"""

import json
import time
import uuid
import base64
import logging
import threading
import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paho.mqtt.client as mqtt
from edge.run_edge_full_pipeline import FullEdgePipelineApp, parse_args

log = logging.getLogger("cloud_bridge")

# ── Config ────────────────────────────────────────────────────────────────────
# Mosquitto broker runs on the laptop (cloud). Default to hotspot IP.
# Override with env if your laptop uses a different address.
MQTT_BROKER_HOST     = os.getenv("MQTT_BROKER_HOST", "192.168.149.1")
MQTT_BROKER_PORT     = int(os.getenv("MQTT_BROKER_PORT", "1883"))
TELEMETRY_INTERVAL_S = 5.0          # publish sensor readings every 5s
SNAPSHOT_DIR         = Path("snapshots")  # where intrusion_events saves clean frames
DELETE_SNAPSHOT_AFTER_PUBLISH = os.getenv("DELETE_SNAPSHOT_AFTER_PUBLISH", "false").lower() in ("1", "true", "yes")

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

    def __init__(self, app: FullEdgePipelineApp):
        self._app     = app
        self._client  = mqtt.Client(client_id="puppypi-edge-publisher")
        self._running = False
        self._thread  = None
        self._seen_snapshots: set = set()   # track already-published snapshots
        self._start_time = time.time()
        self._cpu_prev_total = None
        self._cpu_prev_idle = None

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
        Read _last_temp and _last_gas from the running app
        and publish as a single telemetry message.
        """
        temp = self._app._last_temp
        gas = self._app._last_gas

        payload = {
            "event_id":  str(uuid.uuid4()),
            "device_id": "puppypi-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Extract temp fields if available
        temp_alerts = []
        if temp and "payload" in temp:
            tp = temp["payload"]
            payload["temp_c"]    = tp.get("temperature_c")
            payload["humidity"]  = tp.get("humidity_rh")
            payload["temp_anomaly"] = bool(tp.get("anomaly", False))
            temp_alerts = list(tp.get("alerts") or [])
            payload["temp_alerts"] = temp_alerts

        # Extract gas fields if available
        gas_alerts = []
        if gas and "payload" in gas:
            gp = gas["payload"]
            payload["gas_ppm"]   = gp.get("ppm")
            payload["gas_anomaly"] = bool(gp.get("anomaly", False))
            gas_alerts = list(gp.get("alerts") or [])
            payload["gas_alerts"] = gas_alerts

            # Backward-compatible severity derived from anomaly alerts (not raw ppm threshold).
            severities = {str(a.get("severity", "")).lower() for a in gas_alerts}
            if "critical" in severities:
                payload["gas_severity"] = "CRITICAL"
            elif "warning" in severities:
                payload["gas_severity"] = "WARNING"
            else:
                payload["gas_severity"] = "NORMAL"

        payload["sensor_alerts"] = gas_alerts + temp_alerts

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
                    "severity":    "MOTION",
                    "event_type":  "MOTION",
                    "snapshot_b64": b64_image,
                    "snapshot_path": str(snap_path),
                    # Include latest sensor readings at time of intrusion
                    "gas_ppm":  self._app._last_gas.get("payload", {}).get("ppm") if self._app._last_gas else None,
                    "temp_c":   self._app._last_temp.get("payload", {}).get("temperature_c") if self._app._last_temp else None,
        }

                self._client.publish(
                    TOPIC_INTRUSION,
                    json.dumps(payload),
                    qos=1,   # at-least-once for critical events
                )
                log.info(f"Intrusion event published: {event_id}")
                if DELETE_SNAPSHOT_AFTER_PUBLISH:
                    try:
                        snap_path.unlink()
                    except Exception as e:
                        log.warning(f"Failed to delete snapshot {snap_path}: {e}")

            except Exception as e:
                log.error(f"Failed to publish snapshot {snap_path}: {e}")

    def _publish_heartbeat(self):
        """Publish robot status heartbeat."""
        cpu = self._read_cpu_percent()
        ram = self._read_ram_percent()
        fps = float(getattr(self._app, "_fps_ema", 0.0) or 0.0)
        payload = {
            "device_id":  "puppypi-01",
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state":      "RUNNING",
            "uptime_sec": int(time.time() - self._start_time),
            "cpu":        cpu,
            "ram":        ram,
            "fps":        round(fps, 1),
        }
        self._client.publish(TOPIC_HEARTBEAT, json.dumps(payload), qos=0)

    def _read_cpu_percent(self) -> float:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline()
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "cpu":
                return 0.0
            nums = [int(p) for p in parts[1:]]
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
            total = sum(nums)
            if self._cpu_prev_total is None:
                self._cpu_prev_total = total
                self._cpu_prev_idle = idle
                return 0.0
            total_delta = total - self._cpu_prev_total
            idle_delta = idle - (self._cpu_prev_idle or 0)
            self._cpu_prev_total = total
            self._cpu_prev_idle = idle
            if total_delta <= 0:
                return 0.0
            usage = (1.0 - (idle_delta / total_delta)) * 100.0
            return max(0.0, min(100.0, round(usage, 1)))
        except Exception:
            return 0.0

    def _read_ram_percent(self) -> float:
        try:
            mem_total = None
            mem_available = None
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])
                    if mem_total is not None and mem_available is not None:
                        break
            if not mem_total or mem_available is None:
                return 0.0
            used = mem_total - mem_available
            usage = (used / mem_total) * 100.0
            return max(0.0, min(100.0, round(usage, 1)))
        except Exception:
            return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
class CloudBridgeApp(FullEdgePipelineApp):
    """
    FullEdgePipelineApp + MQTT publishing side-car.
    Inherits everything from FullEdgePipelineApp unchanged.
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
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model_path,
        confidence=args.confidence,
        servo_id=args.servo_id,
        servo_mode=args.servo_mode,
        servo_direction=args.servo_direction,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
        infer_interval_s=args.infer_interval,
        force_infer_interval_s=args.force_infer_interval,
        max_loop_fps=args.max_loop_fps,
        disable_motion_gate=args.disable_motion_gate,
        disable_inference=args.disable_inference,
        disable_overlays=args.disable_overlays,
        render_every_n=args.render_every_n,
        profile_perf=args.profile_perf,
        track_person=args.track_person,
        track_deadband_px=args.track_deadband_px,
        track_max_step=args.track_max_step,
        track_interval_s=args.track_interval,
        enable_mobility=args.enable_mobility,
        approach_on_detect=args.approach_on_detect,
        approach_close_bbox_height_px=args.approach_close_bbox_height,
        teleop_speed_x=args.teleop_speed_x,
        teleop_yaw_deg_s=args.teleop_yaw_deg_s,
        teleop_hold_timeout_s=args.teleop_hold_timeout,
        gait_mode=args.gait_mode,
        wrist_servo_id=args.wrist_servo_id,
        wrist_start_pulse=args.wrist_start_pulse,
        wrist_on_start=not args.disable_wrist_on_start,
        show_window=not args.headless,
        auto_sweep=args.auto_sweep,
    )

    app.run_forever()


if __name__ == "__main__":
    main()
