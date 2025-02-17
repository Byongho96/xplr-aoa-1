import serial
import threading

class SerialReader:
    def __init__(self, port, baudrate=115200, timeout=1, callback=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.callback = callback
        self.serial_connection = None
        self._stop_event = threading.Event() # 스레드 종료 이벤트
        self.thread = None

    def start(self):
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"{self.port} port is connected")
        except serial.SerialException as e:
            print("Port connectio failed:", e)
            return

        self._stop_event.clear()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                if self.serial_connection.in_waiting:
                    data = self.serial_connection.readline().rstrip()
                    if data:
                        if self.callback:
                            self.callback(data)
                        else:
                            print(f"Read from {self.port}:", data)
            except Exception as e:
                print(f"Error from {self.port}:", e)
                break

    def stop(self):
        self._stop_event.set()  # Stop the loop
        if self.thread:
            self.thread.join()  # Stop the thread
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print(f"{self.port} port is disconnected")
