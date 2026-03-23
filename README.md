# INF2009-edge-computing
PuppyPi edge sensing, vision, and servo-control project.

## WiFi Connectivity Set Up
At the moment, PuppyPi is in AP Connection mode, i.e. you directly connect YOUR device to the PUPPYPI's hotspot.

To change it into LAN mode, i.e. connect the PUPPYPI to your LAPTOP's hotspot, follow the next steps. Ensure you have the WonderPi application on your phone

1) Check the 3 blue LEDs next to the PuppyPi's Power Switch.
    - If middle LED is blinking slowly, PuppyPi is in AP connection mode (Giving off Hotspot)
    - If middle LED is **not** blinking, PuppyPi is trying to connect to a WiFi (LAN Mode)
      -  If it is in LAN mode, hold down the button to the LEDs' left (K1), until the middle LED is blinking periodically
2) Once PuppyPi is in AP mode, connect your Laptop to a 2.4GHz WiFi / Hotspot, and turn on your Laptop's hotspot.
3) Connect your phone to the Laptop's hotspot, and go into WonderPi app
    - Select Standard -> PuppyPi -> Add device button on bottom right -> LAN Mode
    - Enter your hotspot password, and it will prompt you to connect to the PuppyPi's hotspot (HW-XXXXXXX)
    - After connecting to PuppyPi's hotspot, go back to the app, it will now try to connect the PuppyPi to laptop's hotspot. On the PuppyPi, you will see the middle LED start **rapidly** blinking, which means its trying to connect.
    - Once it goes from rapidly blinking to NOT blinking, it means it has connected successfully, you should be able to see the connection and IP on your laptop's hotspot settings
4) Enter RealVNC, File -> New connection -> Enter IP Address
    - username: pi
    - password: raspberrypi

 

## Quick Start From Scratch (PuppyPi + ROS Noetic Docker)

### 1) Clone repository on host (Raspberry Pi OS side)
```bash
cd /home/pi
git clone https://github.com/<your-org-or-user>/INF2009-edge-computing.git
cd INF2009-edge-computing
```

### 2) Copy code into existing ROS container (if needed)
If your ROS container is already running (example: `82df027dddb8`):
```bash
docker cp /home/pi/INF2009-edge-computing/. 82df027dddb8:/home/ubuntu/INF2009-edge-computing
```

Notes:
- Run `docker cp` from the **host shell**, not from inside container.
- If you use bind mount (recommended), you do not need repeated `docker cp`.

### 3) Start a ROS Noetic container with bind mount (recommended workflow)
```bash
docker run -it --name puppypi-dev \
  -u ubuntu \
  -w /home/ubuntu/INF2009-edge-computing \
  -v /home/pi/INF2009-edge-computing:/home/ubuntu/INF2009-edge-computing \
  ros:noetic \
  /bin/bash
```

If container already exists:
```bash
docker start -ai puppypi-dev
```

### 4) Inside container: source ROS environments
Run these every new shell session:
```bash
source /opt/ros/noetic/setup.bash
source /home/ubuntu/puppypi/devel/setup.bash
```

Verify:
```bash
python3 -c "import rospy; print('rospy ok')"
rospack find puppy_control
```

### 5) Install Python dependencies (inside container)
```bash
pip3 install opencv-python pyserial smbus2 ultralytics
```

### 6) Run scripts

Full edge pipeline (recommended):
```bash
python3 edge/run_edge_full_pipeline.py --camera-index 2 --servo-id 9 --servo-mode pwm --track-person
```

Vision + servo tracking only (no leg movement):
```bash
python3 edge/run_vision_servo_tracking_only.py --camera-index 2 --servo-id 9 --servo-mode pwm
```

Sensor + vision + servo (earlier integrated runner):
```bash
python3 edge/run_sensor_vision_servo.py --camera-index 2 --servo-id 9 --servo-mode pwm
```

Movement sanity tests:
```bash
python3 edge/control/test_forward_movement.py --speed 8 --duration 1.5
python3 edge/control/test_ijkl_movement.py --speed 10 --yaw-deg 25
```

## Relevant Scripts And What They Do

- `edge/run_edge_full_pipeline.py`
  - Main integration script.
  - Reads webcam + gas + temperature/humidity.
  - Runs person detection + zone/intrusion logic.
  - Tracks person with camera servo (ID 9, PWM mode supported).
  - Supports mobility mode (`IJKL`) when ROS movement stack is available.
  - Saves clean snapshots (without drawn boxes) for cloud-side inference.

- `edge/run_vision_servo_tracking_only.py`
  - Lightweight script for person detection + servo tracking only.
  - Useful to debug vision and camera servo behavior without walking control.

- `edge/run_sensor_vision_servo.py`
  - Reads all sensors and performs local vision with servo controls.
  - Useful for quick integrated edge tests before full mobility flow.

- `edge/control/robot_controller.py`
  - Camera servo abstraction.
  - Supports multiple driver backends (including `ros_robot_controller_sdk` fallback).
  - Includes pulse clamp/center behavior used by tracking and manual controls.

- `puppypi_movement.py`
  - ROS movement controller wrapper.
  - Handles teleop commands, record/playback flow, gait profile, and approach logic.

- `edge/sensor/sensor_manager.py`
  - Starts and health-checks all available edge sensors together.

## Hardware Checks
```bash
ls /dev/video*
ls /dev/ttyACM* /dev/ttyUSB*
ls -l /dev/serial0 /dev/ttyAMA0
i2cdetect -y 1
```
Expected temperature/humidity I2C address: `0x38`.

## Known Operational Notes
- If `ultralytics` warning appears, install it in the same environment where script runs.
- If movement works only after `source .../setup.bash`, that is expected for ROS packages.
- If using `docker cp` workflow, recopy when host files change.
- If using bind mount workflow, edits on host appear in container immediately.
