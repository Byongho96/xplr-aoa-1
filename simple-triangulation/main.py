import os
import argparse
import asyncio
from dotenv import load_dotenv

from client import run_client
from server import run_server

load_dotenv()

ROLE = os.getenv("ROLE", "Client")

def main(interval: float) -> None:
    if ROLE == "Server":
        asyncio.run(run_server(interval))
    elif ROLE == "Client":
        asyncio.run(run_client(interval))
    else:
        print("Invalid role")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line argument parser")
    parser.add_argument("interval", type=float, nargs="?", default=1, help="Interval value (seconds)")

    args = parser.parse_args()
    main(args.interval if args.interval else 1)