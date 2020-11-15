import asyncio
import time
import socketio
import logging

loop = asyncio.get_event_loop()
sio = socketio.AsyncClient()
start_timer = None
logger = logging.getLogger(__name__)

class Blackboard():

    def connect_to_bb(self):
        loop.run_until_complete(self.start_server())

    def notify_bb(self, triples):
        print(triples)
        loop.run_until_complete(self.sendMessage(triples))

    async def start_server(self):
        await sio.connect('http://localhost:5000')
        return

    @sio.event
    async def sendMessage(self, data):
        await sio.emit('sendMessage', data)
        return 0