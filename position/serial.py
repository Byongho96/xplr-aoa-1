from typing import List
import serial
import threading

def read_and_update_serial_data(port: str, baud_rate: int, latest_data: List, idx: int) -> None:
    """
    Read and update the latest data from the serial port.
    """
    ser = serial.Serial(port, baud_rate, timeout=1)    
    try:
        while True:
            data = ser.readline().decode('utf-8').rstrip()
            print(f'{data}')
            # if (data[0] == b"+UUDFP"):
            #     # print(f'{data}')
            #     latest_data[idx] = data[1].split(b",")
    except KeyboardInterrupt:
        print(f"\n❌ {port} Closed.")
    finally:
        ser.close()

def get_latest_data_from_ports(ports: List[str], baud_rate: int) -> List[List[bytes]]:
    """
    Get the latest data from the serial ports.
    """
    latest_data = [[] for _ in range(len(ports))]

    threads = []
    for idx, port in enumerate(ports):
        thread = threading.Thread(target=read_and_update_serial_data, args=(port, baud_rate, latest_data, idx))
        thread.daemon = True  # Stop threads when the program exits
        threads.append(thread)
        thread.start()

    return latest_data
