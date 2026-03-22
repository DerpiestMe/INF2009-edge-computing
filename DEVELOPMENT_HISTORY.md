# Development History and Methodology (Poster Reference)

This document records the technical evolution of the PuppyPi edge pipeline, including architecture decisions, parameter values, measured outcomes, and lessons learned.

## 1) Project Objective
- Build a local, real-time edge monitoring pipeline on PuppyPi that:
  - Reads 3 sensors (webcam, gas, temperature/humidity).
  - Detects humans in camera frames.
  - Supports servo camera sweep/control.
  - Generates intrusion events and snapshots for cloud-side follow-up inference.

## 2) Initial Integration Milestone

### Implemented
- Integrated sensor stack:
  - `edge/sensor/sensor_webcam.py`
  - `edge/sensor/sensor_gas.py`
  - `edge/sensor/sensor_temp.py`
- Added integrated runner:
  - `edge/run_sensor_vision_servo.py`
- Added servo controller abstraction:
  - `edge/control/robot_controller.py`

### Outcome
- Webcam, gas, and temp/humidity successfully initialized.
- Human detection and overlays displayed.
- Servo initially not controllable due to missing SDK/module import path.

## 3) Servo Integration and Driver Compatibility Work

### Problem
- No importable `Board` / `HiwonderSDK` module on target environment.

### Actions
- Added multi-path driver discovery and diagnostics in `robot_controller.py`.
- Added fallback support for `ros_robot_controller_sdk.py`.
- Added mode selection:
  - `servo_mode = auto | pwm | bus`
- Added env-configurable serial settings:
  - `ROS_ROBOT_CONTROLLER_PORT`
  - `ROS_ROBOT_CONTROLLER_BAUD`
  - `ROS_ROBOT_CONTROLLER_TIMEOUT`

### Key finding
- Camera servo movement worked using:
  - `ros_robot_controller_sdk` + `pwm_servo_set_position`
  - Channel/ID 9 in PWM mode.

### Final servo defaults
- Clamp range: `500..1500`
- Center: `1000`
- Sweep default:
  - left `500`
  - right `1500`
  - step `15`
  - interval `0.12s`

## 4) Vision/Inference Optimization Phase

### Performance methodologies added
1. Temporal quantization:
   - Inference every `infer_interval` (default `0.25s`) instead of every frame.
2. Spatial quantization:
   - Infer at reduced resolution (`infer_width x infer_height`, default `320x240`).
3. Motion gating:
   - Run inference mostly when motion is present.
4. Forced refresh:
   - Inference still forced every `force_infer_interval` (default `1.2s`).
5. Detection reuse:
   - Reuse last detections between fresh inference cycles.
6. Loop pacing:
   - Loop capped with `max_loop_fps` (default `18`).
7. Telemetry overlays:
   - Display `FPS`, `InferFPS`, and inference mode (`fresh/reuse`).

### YOLO call constraints added
- `classes=[person]`
- fixed `imgsz`
- `max_det=6`

## 5) Full Pipeline Integration Milestone

### New file
- `edge/run_edge_full_pipeline.py`

### Added module integrations beyond basic runner
- `edge/vision/zone_manager.py`
- `edge/vision/intrusion_events.py`
- `edge/vision/motion_detector.py`

### Functional differences vs basic runner
- Draws restricted zone(s).
- Tags detections as inside/outside zone.
- Confirms intrusion using consecutive-frame + cooldown logic.
- Saves event snapshots.

## 6) Snapshot Data Integrity Decision

### Requirement
- Cloud-side whitelist inference should receive unaltered images.

### Change
- Intrusion snapshots now use `clean_frame` (no boxes, no zone lines, no motion overlays):
  - `events.process(clean_frame, detections)`

### Rationale
- Overlays can bias downstream visual models or post-processors.
- Clean frames are preferred for second-stage cloud inference.

## 7) Performance Debugging and Measured Metrics

### Field-reported profiling metrics
- Step 1 (inference off, overlays off):  
  `capture=988.6ms, motion=5.1ms, overlay=0.0ms`
- Step 2 (inference off, overlays on):  
  `capture=937.9ms, motion=4.3ms, overlay=0.0ms`
- Step 3 (inference on):  
  `capture=996.5ms, motion=4.9ms, overlay=0.0ms`

### Interpretation
- Bottleneck was not inference or overlay.
- Main delay concentrated in "capture stage" around ~1 second.

## 8) Root-Cause-Oriented Pipeline Rethink

### Important discovery
- Previous "capture stage" timing included camera + sensor polling in same block.
- Gas sensor `readline()` timeout can block the loop and mimic low camera FPS.

### Structural improvements added
1. Asynchronous inference worker thread.
2. Sensor polling decoupled from frame cadence:
   - gas poll interval `0.1s`
   - temp poll interval `1.0s`
3. Gas serial timeout reduced in full pipeline:
   - `read_timeout=0.02`
4. Perf profiling split into:
   - `capture_ms`
   - `sensor_ms`
   - `motion_ms`
   - `overlay_ms`

## 9) Camera Path Tuning (Demo-style Mimic Attempt)

### Camera stack updates
- `CAP_V4L2` backend on Linux.
- FourCC forced to `MJPG`.
- Capture buffer size set to `1` for low latency.
- Full pipeline default camera FPS set to `30`.

### Verified camera capability
- Device: Logitech C270 (`uvcvideo`).
- Supports MJPG 640x480 @ 30 FPS.
- Manual command succeeded:
  - `v4l2-ctl --set-fmt-video=...MJPG --set-parm=30`

## 10) Operational Commands Used for Validation

### Core run commands
- Basic optimized runner:
```bash
python3 edge/run_sensor_vision_servo.py --camera-index 2 --servo-id 9 --servo-mode pwm --auto-sweep
```

- Full pipeline runner:
```bash
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --auto-sweep
```

### Profiling mode
```bash
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --profile-perf --auto-sweep
```

### Isolation toggles
```bash
--disable-inference
--disable-overlays
--render-every-n 2
--headless
```

## 11) Current State Summary
- Sensor ingestion: functional.
- Servo control: functional in ROS SDK PWM mode (ID/channel 9).
- Vision inference: functional with quantization controls.
- Intrusion events + snapshots: functional.
- Snapshot cleanliness for cloud inference: implemented.
- Remaining challenge: validating real-time smoothness after decoupled sensor polling and updated profiling buckets.

## 12) Key Methodologies (Poster-friendly)
- Edge-first architecture (local sensing + local inference + local actuation).
- Iterative bottleneck isolation using instrumentation.
- Progressive fallback and compatibility strategy for hardware SDK discovery.
- Quantized inference scheduling for energy/performance balance.
- Clean-data design for multi-stage inference pipelines (edge -> cloud).
- Modular integration to maximize reusability of existing repo components.

## 13) Suggested Poster Sections
1. Problem and constraints (PuppyPi compute + real-time requirements).
2. Architecture diagram (Sensors -> Motion gate -> Inference -> Zone/Intrusion -> Servo/Cloud snapshot).
3. Timeline of engineering decisions.
4. Metrics table (before vs after each optimization phase).
5. Lessons learned (blocking I/O in control loops, SDK portability, clean snapshot strategy).
