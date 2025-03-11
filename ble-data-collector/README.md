# Overview

This code reads data through the serial port and saves it to a CSV file along with a timestamp. It is designed for collecting BLE AoA data.

## How to Run
1. Create a virtual environment  
  ```sh
  python -m venv venv
  ```
2. Activate the virtual environment  
  ```sh
  # Windows
  source venv/Scripts/activate

  # Linux
  source venv/bin/activate
  ```
3. Install dependencies  
  ```sh
  pip install -r requirements.txt
  ```
4. Run the script  
  ```sh
  python main.py 

  # Optional parameters
  python main.py --port COM3 --baudrate 115200
  ```
