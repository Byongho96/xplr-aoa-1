import asyncio
import websockets

class WebsocketClient:
    def __init__(self, uri, message_handler=None):
        self.uri = uri
        self.websocket = None
        self.message_handler = message_handler

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        asyncio.create_task(self._receive())

    async def _receive(self):
        try:
            async for message in self.websocket:
                if self.message_handler:
                    await self.message_handler(message)
                else:
                    print("Received:", message)
        except websockets.ConnectionClosed:
            print("Server disconnected")

    async def send(self, message):
        if self.websocket:
            await self.websocket.send(message)
            print("Sent:", message)

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            print("Disconnected")