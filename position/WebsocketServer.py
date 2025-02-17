import asyncio
import websockets

class WebsocketServer:
    def __init__(self, host='localhost', port=8765, message_handler=None):
        self.host = host
        self.port = port
        self.message_handler = message_handler
        self.clients = set()
        self.server = None

    async def _handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                if self.message_handler:
                    await self.message_handler(websocket, message)
        except websockets.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)

    async def start(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        print(f"Server started: ws://{self.host}:{self.port}")
        await asyncio.Future() 

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("Server stopped")
