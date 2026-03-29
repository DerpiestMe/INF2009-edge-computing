# websocket_server.py
import asyncio
import json
import os
import paho.mqtt.client as mqtt
import websockets

connected_clients = set()

def on_mqtt_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    event = {"topic": msg.topic, "data": payload}
    asyncio.run(broadcast(json.dumps(event)))

async def broadcast(message):
    for ws in list(connected_clients):
        try:
            await ws.send(message)
        except:
            connected_clients.discard(ws)

async def ws_handler(ws):
    connected_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        connected_clients.discard(ws)

async def main():
    mqttc = mqtt.Client()
    mqttc.on_message = on_mqtt_message
    mqtt_host = os.getenv("MQTT_BROKER_HOST", "localhost")
    mqtt_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    mqttc.connect(mqtt_host, mqtt_port)
    mqttc.subscribe("puppypi/#")
    mqttc.loop_start()
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        await asyncio.Future()

asyncio.run(main())
