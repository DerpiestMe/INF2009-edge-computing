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
