# INF2009-edge-computing
PuppyPi Project

## Edge Sensor Ingestion

The sensor ingestion layer is implemented in:

- `edge/sensor/sensor_webcam.py`
- `edge/sensor/sensor_gas.py`
- `edge/sensor/sensor_temp.py`
- `edge/sensor/sensor_manager.py`

### What it does

- Owns raw sensor I/O for webcam, gas (Pico serial), and temperature/humidity (I2C).
- Normalizes records into a shared structure:
	- `sensor_type`
	- `sensor_id`
	- `timestamp`
	- `status` (`ok`, `degraded`, `offline`)
	- `health` (watchdog fields)
	- `payload` (sensor values/metadata)
- Buffers recent records in memory.
- Includes watchdog behavior and reconnect attempts for unstable devices.

### Install dependencies

```bash
pip install opencv-python pyserial smbus2
```

### Run all sensors via manager

```bash
python edge/sensor/sensor_manager.py
```

### Run sensors individually

```bash
python edge/sensor/sensor_webcam.py
python edge/sensor/sensor_gas.py
python edge/sensor/sensor_temp.py
```

### Hardware checks on PuppyPi

```bash
ls /dev/video*
ls /dev/ttyACM* /dev/ttyUSB*
i2cdetect -y 1
```

Expected temperature/humidity sensor I2C address: `0x38`.

### Boot startup (systemd suggestion)

Use a systemd service on the PuppyPi host to start `sensor_manager.py` on power-up.
Set `Restart=on-failure` so sensor ingestion recovers after crashes/reboots.


## PuppyPi startup main script

A ready-to-run startup entrypoint is now provided at `main_puppypi_startup.py`.

### What it integrates

- USB webcam capture and live OpenCV display using `edge/sensor/sensor_webcam.py`.
- HiWonder temperature/humidity readings over I2C using `edge/sensor/sensor_temp.py`.
- Gas readings from a Raspberry Pi Pico over USB serial using `edge/sensor/sensor_gas.py`.
- Person-only object detection overlays using `edge/vision/vision_inference.py`.
- A configurable startup servo pulse for the PuppyPi camera/claw servo.

### Compatibility notes

- The webcam, gas, and temperature readers are reused directly from the existing repository.
- `edge/vision/vision_inference.py` performs person detection and draws bounding boxes.
- `edge/vision/tempmain.py` references an `IntrusionEventManager` that is not currently present in the repository, so `main_puppypi_startup.py` avoids that dependency and only uses modules that already exist.
- Servo movement is best-effort and will activate automatically when one of the common PuppyPi/HiWonder `Board.setPWMServoPulse(...)` modules is installed on the robot.

### Install dependencies on PuppyPi

```bash
pip install opencv-python pyserial smbus2 ultralytics
```

### Run the startup script manually

```bash
python main_puppypi_startup.py --camera-start-servo-pulse 1500
```

Optional flags:

```bash
python main_puppypi_startup.py --print-compatibility
python main_puppypi_startup.py --gas-port /dev/ttyACM0 --gas-port /dev/ttyUSB0
python main_puppypi_startup.py --headless
```
