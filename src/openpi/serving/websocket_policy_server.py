# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import asyncio
import logging
import traceback
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        if hasattr(self._policy, 'start'):
            self._policy.start()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                ## TODO: avoid creating is_thinking False inside the policy server 
                if 'state' in obs:
                    batch_size = obs['state'].shape[0]

                infer_task = asyncio.create_task(asyncio.to_thread(self._policy.infer, obs))
                ack_msg = {}
                while True:
                    if infer_task.done():
                        action = await infer_task
                        await websocket.send(packer.pack(action))
                        break
                    # Check if policy is thinking - handle both single boolean and list of booleans
                    is_thinking = getattr(self._policy, 'is_thinking', False)
                    if isinstance(is_thinking, list):
                        # For batch policy: check if all element is True
        
                        if all(is_thinking):
                       
                            await websocket.send(packer.pack({"isthinking": is_thinking}))
                            # Check if the received message is a reset signal
                            ack_msg = msgpack_numpy.unpackb(await websocket.recv())
                        if 'reset_thinking' in ack_msg and ack_msg['reset_thinking']:
                            # Cancel the current inference task and process reset
                       
                            infer_task.cancel()
                            ## also send back the self._policy.thought
                            try:
                                await infer_task
                            except asyncio.CancelledError:
                                pass
                            # Process the reset directly
                            reset_response = await asyncio.to_thread(self._policy.infer, ack_msg)
                           
                            await websocket.send(packer.pack(reset_response))
                            break  # Exit the thinking loop, not the handler
                       # Exit the think
                                
                    else:
                        # For single policy: check the boolean directly
                        if is_thinking:
                            await websocket.send(packer.pack({"isthinking": np.True_}))
                            # Check if the received message is a reset signal
                            ack_msg = msgpack_numpy.unpackb(await websocket.recv())
                            if 'reset_thinking' in ack_msg and ack_msg['reset_thinking']:
                                # Cancel the current inference task and process reset
                                infer_task.cancel()
                                try:
                                    await infer_task
                                except asyncio.CancelledError:
                                    pass
                                # Process the reset directly
                                reset_response = await asyncio.to_thread(self._policy.infer, ack_msg)
                                await websocket.send(packer.pack(reset_response))
                                break  # Exit the thinking loop, not the handler
                    await asyncio.sleep(0.02)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
