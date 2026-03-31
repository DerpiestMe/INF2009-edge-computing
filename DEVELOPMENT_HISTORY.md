# Development History and Methodology (Current State)

This document reflects the repository state as of **March 31, 2026**.

## 1) Project Objective
- Build an edge-first PuppyPi system that combines:
  - webcam person detection
  - gas + temperature/humidity sensing
  - camera-servo sweep/tracking
  - intrusion snapshot/event generation
  - optional robot mobility (teleop, record/replay, approach-on-detect)
  - cloud/dashboard telemetry + alerting

## 2) Core Integration Milestone
- Integrated edge modules into runnable pipelines:
  - `edge/run_sensor_vision_servo.py`
  - `edge/run_edge_full_pipeline.py`
  - `edge/run_vision_servo_tracking_only.py`
- Sensor modules used:
  - `edge/sensor/sensor_webcam.py`
  - `edge/sensor/sensor_gas.py`
  - `edge/sensor/sensor_temp.py`
- Vision/event modules used:
  - `edge/vision/vision_inference.py`
  - `edge/vision/motion_detector.py`
  - `edge/vision/zone_manager.py`
  - `edge/vision/intrusion_events.py`

## 3) Servo Driver Compatibility and Control
### Challenge
- PuppyPi SDK import paths were inconsistent across environments (`Board`, `HiwonderSDK`, ROS SDK variants).

### Mitigation
- Added fallback detection and diagnostics in:
  - `edge/control/robot_controller.py`
- Implemented selectable camera servo backend:
  - `auto | pwm | bus`

### Result
- Camera servo control stabilized on PuppyPi via `ros_robot_controller_sdk` PWM mode.

## 4) Performance Optimization and Runtime Smoothness
### Challenge
- Initial integrated pipeline was near/sub-1 FPS (capture bottleneck dominant).

### Mitigation
- Added inference quantization and pacing:
  - `--infer-width/--infer-height`
  - `--infer-interval`
  - `--force-infer-interval`
  - `--max-loop-fps`
  - detection reuse between infer cycles
- Decoupled sensor polling cadence from vision loop:
  - gas poll at 0.1s cadence
  - temp poll at 1.0s cadence

### Result
- Field-tested runtime improved to about **17-18 FPS** in latest smooth configuration.

## 5) Snapshot/Event Pipeline Evolution
### Earlier behavior
- Snapshot trigger depended on restricted-zone intrusion logic.

### Current behavior
- Snapshot trigger is now any detected person in full pipeline mode:
  - `IntrusionEventManager(trigger_on_any_person=True, confirm_frames=1)`
- Default snapshot cooldown is now **5 seconds** (`--snapshot-cooldown-seconds` default `5.0`).
- Snapshots use clean frame path (no overlay boxes) for downstream cloud inference quality.

### Result
- More useful face/subject captures while reducing burst spam with cooldown.

## 6) Mobility, Record/Replay, and Approach Control
### Challenge
- Conflicts between teleop, replay, and auto-approach behavior.

### Current control rules
- Manual recording now overrides approach.
- Replay/patrol can be interrupted by person detection and approach can take over.
- `approach_on_detect` no longer auto-stops recording; recording must be stopped by operator.

### Result
- Behavior now matches intended workflow:
  - record route manually without forced approach takeover
  - approach detected person during replay/patrol operations

## 7) Person Tracking and Servo Direction Fixes
### Challenge
- Initial servo direction mapping was reversed (left/right key and tracking response opposite).

### Mitigation
- Added servo direction multiplier and corrected directional behavior in tracking/control path.

### Result
- Manual sweep and person-centering now align with expected direction semantics.

## 8) Sensor Anomaly Alerting on Edge
### Gas anomaly model (`edge/sensor/sensor_gas.py`)
- Absolute thresholds:
  - warning: `>= 1000 ppm`
  - critical: `>= 2000 ppm`
  - emergency: `>= 5000 ppm`
- Spike detection vs rolling median baseline:
  - warning delta: `>= 300 ppm`
  - critical delta: `>= 600 ppm`
  - window: `120 s`

### Temperature anomaly model (`edge/sensor/sensor_temp.py`)
- Absolute low/high thresholds:
  - low warning: `<= 18C`, low critical: `<= 15C`
  - high warning: `>= 30C`, high critical: `>= 35C`
- Change/spike thresholds over window:
  - window: `300 s` (5 min)
  - warning delta: **`>= 1.0C`** (updated from 2.0C)
  - critical delta: **`>= 2.0C`** (updated from 4.0C)

### Output format
- Sensor payload includes:
  - `anomaly` boolean
  - `alerts` array (`edge.sensor_alert.v1`)

## 9) Cloud + Dashboard Integration (Latest Updates)
### Challenge
- Dashboard showed frequent `GAS_HIGH` warnings at low-300 ppm due legacy generic threshold path.

### Root cause
- Legacy cloud/dashboard logic used telemetry `gas_severity` bands, not sensor anomaly alerts.

### Mitigation implemented
- `edge/cloud_bridge.py` telemetry now forwards:
  - `gas_alerts`, `temp_alerts`, `sensor_alerts`
  - `gas_anomaly`, `temp_anomaly`
- `cloud/pipeline/cloud_subscriber.py` telemetry handler now:
  - consumes anomaly alert arrays
  - writes structured anomaly records (`GAS_ANOMALY`, `TEMP_ANOMALY`, etc.)
  - stops creating `GAS_HIGH` from raw ppm bands
- `cloud/dashboard/src/hooks/useRobotData.js` now:
  - stops generating `GAS_HIGH` from telemetry severity
  - creates dashboard alerts only from anomaly alert arrays
  - filters legacy `GAS_HIGH` entries from local storage/history load

### Result
- Dashboard sensor alerts now align with edge anomaly logic instead of noisy baseline ppm bands.

## 10) Cloud/Edge Data Flow (Current)
1. Edge pipeline reads sensors + camera and performs local person detection.
2. Intrusion snapshots are captured on edge and published via MQTT.
3. Edge telemetry includes raw readings + anomaly alert arrays.
4. Cloud subscriber stores telemetry/events, runs whitelist re-ID, and persists alert history.
5. Websocket server forwards MQTT topics to frontend.
6. Dashboard renders live telemetry, event stream, and anomaly-driven alerts.

## 11) Main Challenges and Mitigations Summary
1. SDK/environment mismatch:
   - fallback servo backend loading + diagnostics.
2. Very low FPS/jitter:
   - quantized inference + loop pacing + decoupled sensor cadence.
3. Snapshot usefulness and spam:
   - clean-frame capture + any-person trigger + cooldown throttling.
4. Control-priority conflicts:
   - explicit override rules between recording/replay/approach/teleop.
5. Alert noise:
   - moved from generic ppm warning bands to structured sensor anomaly alerts.

## 12) Current Feature Set (Latest Version)
- Full integrated edge runtime:
  - sensors + vision + snapshots + servo tracking/sweep + optional mobility.
- Person-approach mode with configurable center/distance parameters.
- Structured anomaly alerts from gas/temp sensors suitable for MQTT/cloud.
- Cloud ingestion with alert history persistence and dashboard live updates.
- Whitelist face re-identification flow on intrusion snapshots.

## 13) Poster-Friendly Methodology Points
- Edge-first architecture with modular fallbacks.
- Iterative bottleneck isolation with instrumentation metrics.
- Parameterized control design for rapid field tuning.
- Structured event contracts for cloud interoperability.
- Continuous feedback loop: observe -> tune -> validate -> document.
