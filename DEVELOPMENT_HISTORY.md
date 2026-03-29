# Development History and Methodology (Current State)

This document reflects the repository state as of the latest updates in this branch.

## 1) Objective
- Build an edge-first PuppyPi pipeline that combines:
  - Multi-sensor ingestion (webcam, gas, temperature/humidity)
  - Local person detection
  - Camera-servo tracking/sweep control
  - Intrusion event generation + snapshot capture
  - Optional robot mobility with manual teleop and approach behavior

## 2) Early Integration Milestone
- Integrated sensor modules:
  - `edge/sensor/sensor_webcam.py`
  - `edge/sensor/sensor_gas.py`
  - `edge/sensor/sensor_temp.py`
- Added integrated run scripts:
  - `edge/run_sensor_vision_servo.py`
  - `edge/run_edge_full_pipeline.py`
- Added vision stack:
  - `edge/vision/vision_inference.py`
  - `edge/vision/motion_detector.py`
  - `edge/vision/zone_manager.py`
  - `edge/vision/intrusion_events.py`

## 3) Servo Driver Compatibility Phase
### Problem
- PuppyPi SDK imports differed across environments (missing `Board`, `HiwonderSDK`, etc.).

### Mitigation
- Implemented fallback/diagnostic servo controller:
  - `edge/control/robot_controller.py`
  - multi-path module loading
  - fallback to `ros_robot_controller_sdk`
  - selectable mode: `auto | pwm | bus`

### Outcome
- Camera servo control stabilized with PWM mode on servo/channel ID 9 in target runtime.

## 4) Performance and Smoothness Work
### Problem
- Initial runtime exhibited poor FPS and jitter under integrated load.

### Mitigation
- Added inference scheduling and loop pacing in full pipeline:
  - temporal quantization (`--infer-interval`)
  - spatial quantization (`--infer-width/--infer-height`)
  - optional motion-gated inference
  - forced refresh interval
  - detection reuse between inference updates
  - loop FPS cap (`--max-loop-fps`)
- Reduced blocking effects:
  - decoupled sensor polling cadence
  - shorter gas serial timeout in integrated runner

### Outcome
- Runtime became significantly smoother than initial <1 FPS behavior.

## 5) Snapshot Integrity and Event Robustness
### Problem
- Need cloud-ready snapshots without overlays; snapshot saves needed clearer reliability.

### Mitigation
- Snapshots are generated from clean frame path:
  - `events.process(clean_frame, detections)`
- Improved snapshot manager:
  - parent directory creation with `parents=True`
  - write-success check for `cv2.imwrite`
  - explicit `snapshot_saved` flag in event payload
  - unique event sequence in filenames
  - configurable cooldown in full pipeline (`--snapshot-cooldown-seconds`)

### Outcome
- Cleaner downstream inference inputs and lower snapshot spam rate.

## 6) Mobility, Teleop, and Replay Iterations
### Problem
- Key handling and replay behavior were inconsistent in some runtime paths.

### Mitigation
- Updated full pipeline control loop:
  - key handling on every loop (not only rendered frames)
  - explicit terminal feedback for `r/p/h`
  - replay no longer fights teleop velocity stream
  - manual teleop can temporarily override auto-approach
- Updated movement controller:
  - stronger visibility logs for recording/replay lifecycle
  - body height parameterization for stance tuning (`--body-height`)

### Outcome
- Manual movement + record/replay now provide observable feedback and improved behavior.

## 7) Approach-on-Detect Evolution
### Problem
- Robot could spend too long aligning or stop too early due simplistic closeness gating.

### Mitigation
- Improved approach logic in `puppypi_movement.py` and full pipeline:
  - body heading uses blend of:
    - person center error in frame
    - camera servo offset from center
  - optional align-first rotation behavior
  - closeness now checks both:
    - proximity metric
    - centering tolerance
- Added tunable CLI controls:
  - `--approach-center-tolerance`
  - `--approach-servo-center-tolerance`
  - `--approach-target-distance-m`
  - `--approach-distance-ref-m`
  - `--approach-distance-ref-bbox-height`
  - `--approach-close-bbox-height` (legacy pixel-threshold mode)
- Added approach state/metric logs for debugging:
  - detection count
  - replay/recording override status
  - estimated distance
  - close/not-close status

### Outcome
- Approach behavior is now more tunable and diagnosable in field testing.

## 8) Sensor Anomaly Alerting (Dashboard-Ready Payloads)
### Problem
- Needed actionable alerts (not only raw readings) for cloud/dashboard consumption.

### Mitigation
- Added structured anomaly alerts directly in sensor payloads:
  - schema: `edge.sensor_alert.v1`
  - fields include severity, metric, threshold, comparison, message, optional baseline/delta/window
- Gas sensor (`edge/sensor/sensor_gas.py`):
  - absolute threshold alerts (warning/critical/emergency)
  - short-window spike alerts against rolling median baseline
- Temperature sensor (`edge/sensor/sensor_temp.py`):
  - low/high absolute alerts
  - rapid-change alerts over time window (up or down)
- Full pipeline now emits throttled alert envelopes:
  - schema: `edge.alert_event.v1`
  - includes sensor metadata, health, reading context, and alert object
- Sensor manager also logs structured alert JSON entries.

### Outcome
- Alert semantics are now available for MQTT/dashboard publishing and cloud event ingestion.

## 9) Current Runtime Characteristics (What Is True Now)
- Full runner: `edge/run_edge_full_pipeline.py`
  - sensor ingestion, person detection, zone intrusion, clean snapshots
  - servo tracking/sweep controls
  - optional mobility + teleop + record/replay
  - optional approach behavior with tunable center/distance heuristics
  - snapshot cooldown control
  - sensor alert event emission
- Additional runners available:
  - `edge/run_sensor_vision_servo.py`
  - `edge/run_vision_servo_tracking_only.py`

## 10) Key Challenges and How They Were Mitigated
1. Hardware SDK variability:
   - Mitigation: robust import fallback and diagnostics in servo controller.
2. Low FPS / jitter:
   - Mitigation: inference quantization, loop pacing, and non-blocking sensor strategy.
3. Replay/teleop conflicts:
   - Mitigation: prevent concurrent command stream conflicts, add explicit control feedback.
4. Premature or indecisive approach behavior:
   - Mitigation: blended heading logic + configurable tolerances + metric logging.
5. Alert-less raw telemetry:
   - Mitigation: structured anomaly events with schema and severity.
6. Snapshot uncertainty:
   - Mitigation: save-result checks, cooldown control, and clean-frame capture policy.

## 11) Poster-Friendly Methodology Summary
- Edge-first modular architecture
- Instrumentation-driven bottleneck isolation
- Compatibility-first hardware integration
- Parameterized control for field tuning
- Structured event design for cloud interoperability
- Iterative close-the-loop validation with runtime diagnostics

