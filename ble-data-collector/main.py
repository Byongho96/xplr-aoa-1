import csv
import yaml
import argparse
from datetime import datetime
from packages import SerialReader
        
def main(name, port, baudrate):
    reader = SerialReader(port, baudrate)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f'{(name and name + "-")}{timestamp}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamp', 'Data'])
        
        try:
            while True:
                data = reader.read_line()
                if data and data.startswith('+UUDF:'):
                    timestamp = datetime.now().isoformat()
                    csvwriter.writerow([timestamp, data ])
        except KeyboardInterrupt:
            print("Terminated by user")
        finally:
            reader.close()


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description="Command-line argument parser")
    parser.add_argument("name", type=str, nargs="?", default="", help="CSV name")
    parser.add_argument("port", type=str, nargs="?", default=config['port'], help="Serial port")
    parser.add_argument("baudrate", type=int, nargs="?", default=config['baudrate'], help="Braud rate")

    args = parser.parse_args()
    main(args.name, args.port, args.baudrate)