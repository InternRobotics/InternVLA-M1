import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from tools.model_interface import QwenpiPolicyInterfence


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: QwenpiPolicyInterfence,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy  #
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

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
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

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        route rules:
        - compatible with two styles:
          1) explicit type: msg = {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          2) old version implicit key: contains "device" as init, contains "reset" as reset, otherwise infer
        return: unified dictionary, at least contains {"status": "ok"|"error"}, and include "ok"/"type"/"request_id"
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "default")  # default is infer
        payload = msg.get("payload", msg)  # when no payload, use top level

        # 1) explicit type routing
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "pong", "request_id": req_id}

        if mtype == "init":
            ok = bool(self._policy.init_infer(payload))
            if ok:
                return {"status": "ok", "ok": True, "type": "init_result", "request_id": req_id}
            return {
                "status": "error",
                "ok": False,
                "type": "init_result",
                "request_id": req_id,
                "message": "Failed to initialize device",
            }

        if mtype == "reset":
            # compatible with different field names
            instr = payload.get("instruction") or payload.get("task_description")
            self._policy.reset(instr)
            return {"status": "ok", "ok": True, "type": "reset_result", "request_id": req_id}

        if mtype == "infer":
            data = self._policy.step(payload)
            return {"status": "ok", "ok": True, "type": "inference_result", "request_id": req_id, "data": data}

        # 2) compatible with old version implicit key routing
        if "device" in msg:
            ok = bool(self._policy.init_infer(msg))
            if ok:
                return {"status": "ok", "ok": True, "type": "init_result", "request_id": req_id}
            return {
                "status": "error",
                "ok": False,
                "type": "init_result",
                "request_id": req_id,
                "message": "Failed to initialize device",
            }

        if "reset" in msg:
            instr = msg.get("instruction") or msg.get("task_description")
            self._policy.reset(instr)
            return {"status": "ok", "ok": True, "type": "reset_result", "request_id": req_id}

        raw_action = self._policy.step(**msg)
        data = {"raw_action": raw_action}
        return {"status": "ok", "ok": True, "type": "inference_result", "request_id": req_id, "data": data}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=8765)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
