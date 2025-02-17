import time
import argparse
import position
import math
import threading

import asyncio
import websockets
import json

from pprint import pprint

WEBSOCKET_PORT = 8765

BAUD_RATE = 115200 # All serial ports should have the same baud rate

"""
7/////////////////////5
///////////////////////
///////////////////////
//////////////////////4
"""

PORTS = [ "COM4"]

# Right-handed Cartesian coordinates
# Anchor positions in meters
ANCHOR_POSITIONS = [
    (1.2, 0.25, 0.6),   
    (1.2, 0.25, 0),
    (0, 0.25, 0)
]

# Anchor rotations (r_x, r_y, r_z) in radians
# (0, 0, 0) is facing the positive z-axis
ANCHOR_ROTATIONS = [
    (0, math.radians(-90), 0),
    (0, math.radians(-90), 0),
    (0, math.radians(90), 0)
]


ble_position = (0, 0, 0)

def update_ble_position(interval: float) -> None:
    global ble_position
    latest_serial_data = position.serial.get_latest_data_from_ports(PORTS, BAUD_RATE)

    # try:
    #     while True:
    #         if any([len(data) == 0 for data in latest_serial_data]):
    #             continue

    #         # angles = [(math.radians(-float(data[2])), math.radians(-float(data[3]))) for data in latest_serial_data]
    #         # ble_position = position.triangulation.calculate_source_position(ANCHOR_POSITIONS, ANCHOR_ROTATIONS, angles)
            
    #         # Debug
    #         # print([data for data in latest_serial_data])
    #         # print("Estimated BLE source position:", ble_position)
            
    #         # tracker.update(ble_position[0], ble_position[2], ble_position[1])
    #         time.sleep(interval)
    # except KeyboardInterrupt:
    #     print("\n❌ 프로그램 종료됨.")

async def send_ble_position(websocket):
    global ble_position

    try:
        while True:
            position_data = json.dumps({"x": ble_position[0], "y": ble_position[1], "z": ble_position[2]})
            await websocket.send(position_data)
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket 연결이 종료되었습니다.")

async def websocket_server():
    server = await websockets.serve(send_ble_position, "0.0.0.0", WEBSOCKET_PORT)
    # print(f"🚀 WebSocket 서버가 ws://localhost:{WEBSOCKET_PORT} 에서 실행 중...")
    await server.wait_closed()


def main(interval: float) -> None:
    # tracker = position.gui.BLETracker3D()

    ble_thread = threading.Thread(target=update_ble_position, args=(interval,))
    ble_thread.daemon = True
    ble_thread.start()

    # tracker.show()
    asyncio.run(websocket_server())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line argument parser")
    parser.add_argument("interval", type=float, nargs="?", default=1, help="Interval value (seconds)")

    args = parser.parse_args()
    main(args.interval if args.interval else 1)