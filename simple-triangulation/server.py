import os
import json
import time
import threading
from position import Triangulator
from position import WebsocketServer

SOCKET_IP = os.getenv("SOCKET_IP", "localhost")
SOCKET_PORT = int(os.getenv("SOCKET_PORT", "8765"))

boards = {}

def run_background(interval: float):
    try:
        while True:
            if len(boards) > 2:
                position = Triangulator.estimate_position(list(boards.values()))
                print("Estimated BLE source position:", position)
            else:
                print("Not enough data to estimate position", len(boards))
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

def message_handler(websocket, data: str):
    json_data = json.loads(data)
    boards[websocket.id] = json_data

async def run_server(interval: float):
    server_thread = threading.Thread(target=run_background, args=(interval,), daemon=True)
    server_thread.start()

    ws = WebsocketServer(SOCKET_IP, SOCKET_PORT, message_handler)
    await ws.start()

        