# Overview

This project refactors the Python code from the [corresponding paper](https://ieeexplore.ieee.org/abstract/document/10843768) while studying it. The original Python code can be found [here](https://zenodo.org/records/14078963).

The dataset used in this project was downloaded from the [corresponding paper](https://ieeexplore.ieee.org/document/10201845/references#references) at [Zenodo](https://zenodo.org/records/7759557).

Since the logic follows the original code exactly, it produces the same results while improving readability and reducing runtime.

## Calibration

BLE signals are transmitted from 119 static points, and RSSI values are collected from four receivers. Using this data, the path-loss coefficient for each anchor is calculated based on the RSSI values.

## Mobility


## How to Run
1. Create a virtual environment  
  ```sh
  python -m venv venv
  ```
2. Activate the virtual environment  
  ```sh
  source venv/Scripts/activate
  ```
3. Install dependencies  
  ```sh
  pip install -r requirements.txt
  ```
4. Run the script  
  ```sh
  python main.py
  ```
