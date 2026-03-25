#!/usr/bin/env python3
"""
mqtt_test_publisher.py
======================
Simulates the PuppyPi robot publishing sensor + alert data to MQTT.
Use this to test your dashboard WITHOUT needing real hardware.

Run:  python mqtt_test_publisher.py
      python mqtt_test_publisher.py --host 192.168.x.x   (remote broker)
      python mqtt_test_publisher.py --gas-spike           (trigger gas alert)

Dependencies: pip install paho-mqtt
"""

import argparse
import json
import math
import random
import time

import paho.mqtt.client as mqtt

parser = argparse.ArgumentParser()
parser.add_argument("--host",      default="localhost")
parser.add_argument("--port",      type=int, default=1883)
parser.add_argument("--robot-id",  default="robot/01")
parser.add_argument("--gas-spike", action="store_true",
                    help="Send a critical gas reading immediately")
args = parser.parse_args()

ROBOT = args.robot_id

client = mqtt.Client(client_id="test-publisher")
client.connect(args.host, args.port)
client.loop_start()
print(f"Publishing to {args.host}:{args.port} as {ROBOT}")
print("Press Ctrl+C to stop\n")

start    = time.time()
tick     = 0
cpu_base = 55

try:
    while True:
        t = time.time()
        ts = int(t)

        # ── Gas sensor (MQ2) ────────────────────────────────────────────────
        if args.gas_spike and tick == 0:
            gas_val = 850.0
        else:
            # Slow drift with small noise
            gas_val = 180 + 60 * math.sin(t / 60) + random.gauss(0, 8)
            gas_val = max(0, gas_val)

        client.publish(f"{ROBOT}/gas", json.dumps({
            "value": round(gas_val, 1),
            "unit": "ppm",
            "ts": ts,
        }))

        # ── Temperature sensor ───────────────────────────────────────────────
        temp_val = 26 + 4 * math.sin(t / 120) + random.gauss(0, 0.3)
        client.publish(f"{ROBOT}/temp", json.dumps({
            "value": round(temp_val, 1),
            "unit": "C",
            "ts": ts,
        }))

        # ── Person detection (random, ~10% chance per cycle) ─────────────────
        if random.random() < 0.10:
            conf = round(random.uniform(0.65, 0.97), 2)
            client.publish(f"{ROBOT}/person", json.dumps({
                "detected": True,
                "confidence": conf,
                "bbox": [120, 80, 280, 360],
                "ts": ts,
            }))
            print(f"  [person]  confidence={conf}")

        # ── Gas alert when threshold exceeded ────────────────────────────────
        if gas_val > 400:
            client.publish(f"{ROBOT}/alert", json.dumps({
                "type": "GAS_HIGH",
                "severity": "critical",
                "message": f"Gas {gas_val:.0f} ppm exceeds threshold (400 ppm)",
                "ts": ts,
            }))
            print(f"  [ALERT]   GAS_HIGH  {gas_val:.0f} ppm")

        # ── System status (every 5 ticks) ────────────────────────────────────
        if tick % 5 == 0:
            uptime = int(t - start)
            cpu = cpu_base + random.randint(-5, 15)
            ram = 68 + random.randint(-3, 8)
            fps = round(random.uniform(8, 12), 1)
            client.publish(f"{ROBOT}/status", json.dumps({
                "robot_id": ROBOT,
                "cpu":    min(cpu, 100),
                "ram":    min(ram, 100),
                "uptime": uptime,
                "fps":    fps,
                "ts":     ts,
            }))
            print(f"  [status]  cpu={cpu}%  ram={ram}%  fps={fps}  uptime={uptime}s")

        tick += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped.")
    client.loop_stop()
    client.disconnect()