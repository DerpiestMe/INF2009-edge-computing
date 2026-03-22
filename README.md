# INF2009-edge-computing
PuppyPi edge sensing, vision, and servo-control project.

## Repository Structure
- `edge/sensor/`: webcam, gas, and temperature/humidity sensor adapters.
- `edge/vision/`: motion detection, YOLO person detection, zone logic, and intrusion event generation.
- `edge/control/`: servo controller and control-layer placeholders.
- `cloud/`: cloud pipeline and dashboard placeholders.
- `shared/`: shared schema placeholders.

## Core Edge Modules
- `edge/sensor/sensor_webcam.py`
- `edge/sensor/sensor_gas.py`
- `edge/sensor/sensor_temp.py`
- `edge/sensor/sensor_manager.py`
- `edge/vision/motion_detector.py`
- `edge/vision/vision_inference.py`
- `edge/vision/zone_manager.py`
- `edge/vision/intrusion_events.py`
- `edge/control/robot_controller.py`

## Main Runners

### 1) Optimized sensor + vision + servo runner
File: `edge/run_sensor_vision_servo.py`

What it does:
- Reads all 3 sensors.
- Runs person detection.
- Displays live camera feed.
- Controls/sweeps webcam servo (ID/channel configurable).
- Includes FPS and performance controls.

Example:
```bash
python3 edge/run_sensor_vision_servo.py --camera-index 2 --servo-id 9 --servo-mode pwm --auto-sweep
```

### 2) Full pipeline runner (recommended for integration tests)
File: `edge/run_edge_full_pipeline.py`

What it does:
- Integrates all non-empty edge sensor/vision/control modules.
- Adds zone drawing, intrusion confirmation, and snapshot saving.
- Uses async inference + quantized inference scheduling for smoother loop/servo behavior.
- Saves **clean snapshots without bounding boxes/overlays** for cloud-side re-inference.

Example:
```bash
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --auto-sweep
```

## Dependencies
Install on PuppyPi:
```bash
pip install opencv-python pyserial smbus2 ultralytics
```

## Hardware Checks (PuppyPi)
```bash
ls /dev/video*
ls /dev/ttyACM* /dev/ttyUSB*
ls -l /dev/serial0 /dev/ttyAMA0
i2cdetect -y 1
```

Expected temperature/humidity I2C address: `0x38`.

## Servo Control Notes
- Current servo defaults are tuned for webcam movement:
  - Clamp: `500..1500`
  - Center: `1000`
- `robot_controller.py` supports:
  - Common HiWonder `Board` module APIs.
  - `ros_robot_controller_sdk.py` fallback.
- For ROS SDK, default mode should be `pwm` for camera servo channel/ID usage that matches your tests.

## Performance / Quantization Features
Both optimized runners support performance-oriented controls:
- Temporal quantization: run inference every `--infer-interval`.
- Spatial quantization: infer at `--infer-width x --infer-height`.
- Motion-gated inference with forced refresh (`--force-infer-interval`).
- Detection reuse between inference updates.
- Loop FPS capping (`--max-loop-fps`).
- On-screen telemetry (`FPS`, `InferFPS`, inference freshness).

Useful tuning examples:
```bash
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --infer-width 256 --infer-height 192 --infer-interval 0.35 --max-loop-fps 15 --auto-sweep
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --disable-motion-gate --infer-interval 0.2
```

## If FPS Is Very Low (for example ~1 FPS)
Most likely causes are local compute/runtime contention, not network.

Check in this order:
1. Model load/perf
- Try smaller/lighter model or lower infer resolution.
- Increase `--infer-interval` and lower `--max-loop-fps`.

2. Display stack issues
- If Qt/Wayland warnings appear, try `--headless` to isolate display overhead.

3. CPU contention
- Ensure no other heavy process is running YOLO/camera loops.
- Keep only one process accessing `/dev/video*` and servo UART.

4. Servo/UART contention
- Make sure `/dev/ttyAMA0` is not held by other long-running processes while testing.

5. Camera index
- Use explicit `--camera-index` that works on your PuppyPi (for example `2`).

## Clean Snapshot Behavior
`edge/run_edge_full_pipeline.py` now sends a raw frame to `IntrusionEventManager`, so saved images are clean and suitable for downstream cloud whitelist inference.

## Existing Placeholders
These files are currently empty placeholders:
- `edge/control/fusion_engine.py`
- `edge/control/threat_fsm.py`
- `edge/control/speaker_alert.py`
- `shared/mqtt_schema.py`
- `puppypi_movement.py`

They are reserved for future development and not yet wired with logic.
