import serial

class SerialReader:
    def __init__(self, port, baudrate=11520, timeout=1):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def read_line(self):
        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            return line
        except Exception as e:
            print("Read Error:", e)
            return None

    def close(self):
        self.ser.close()