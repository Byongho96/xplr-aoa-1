# BLE AoA Position Tracking

This project reads AoA values from multiple C211 anchors, transmits them to a server via WebSocket, and tracks real-time positions using a simple LMS linear algebra algorithm.

## Output Data
- The result follows the Right-hand Cartesian coordinate system.  
- See the diagram for reference.  
- This is a basic implementation and does not consider sensor or system noise, so errors may be significant.

## Environment Variables

Create a `.env` file in the root directory and set the following variables.

### Client
```ini
ROLE=Client

BOARD_POSITION_X=0  # Anchor X position
BOARD_POSITION_Y=0  # Anchor Y position
BOARD_POSITION_Z=0     # Anchor Z position

BOARD_ROTATION_X=0   # Anchor X Euler rotation (degree)
BOARD_ROTATION_Y=0 # Anchor Y Euler rotation (degree)
BOARD_ROTATION_Z=0   # Anchor Z Euler rotation (degree)

COM_PORT=COM5        # C211 serial port
BAUD_RATE=115200     # C211 baud rate

SOCKET_IP=caec-163-152-3-131.ngrok-free.app  # Server IP
SOCKET_PORT=8765  # Server port
```

### Server
```ini
ROLE=Server

SOCKET_PORT=8765  # Server port
```

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
