import os
import json
import asyncio
from position import SerialReader
from position import WebsocketClient

COM_PORT = os.getenv("COM_PORT", "") 
BAUD_RATE = int(os.getenv("BAUD_RATE", "115200"))

BOARD_POSITION_X = float(os.getenv("BOARD_POSITION_X", "0"))
BOARD_POSITION_Y = float(os.getenv("BOARD_POSITION_Y", "0"))
BOARD_POSITION_Z = float(os.getenv("BOARD_POSITION_Z", "0"))

BOARD_ROTATION_X = float(os.getenv("BOARD_ROTATION_X", "0"))
BOARD_ROTATION_Y = float(os.getenv("BOARD_ROTATION_Y", "0"))
BOARD_ROTATION_Z = float(os.getenv("BOARD_ROTATION_Z", "0"))

SOCKET_IP = os.getenv("SOCKET_IP", "localhost")
SOCKET_PORT = int(os.getenv("SOCKET_PORT", "8765"))

board = {
    "position": (BOARD_POSITION_X, BOARD_POSITION_Y, BOARD_POSITION_Z),
    "rotation": (BOARD_ROTATION_X, BOARD_ROTATION_Y, BOARD_ROTATION_Z),
    "angle": (0, 0),
}

def serial_handler(bytes: bytes):
    data = bytes.split(b":")

    if (data[0] == b"+UUDF"):
        info = data[1].split(b",")
        azimuth = -float(info[2])
        elevation = float(info[3])    
        board["angle"] = (elevation, azimuth)

async def run_client(interval: float):
    if not COM_PORT:
        print("COM_PORT is not defined")
        return
    
    reader = SerialReader(COM_PORT, BAUD_RATE, callback=serial_handler)
    reader.start()
    
    ws = WebsocketClient(f'wss://{SOCKET_IP}')
    await ws.connect()
    
    while True:
        await ws.send(json.dumps(board))
        await asyncio.sleep(interval)

