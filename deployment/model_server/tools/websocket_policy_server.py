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
        self._policy = policy # 
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

        while True: # TODO 这里定义了有什么服务， 但是看起来服务非映射还做的很不高兴
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg)  # 路由消息
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
    # 路由逻辑： 识别 client 端发过来的 request
    def _route_message(self, msg: dict) -> dict:
        """
        路由规则：
        - 兼容两种风格：
          1) 显式 type：msg = {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          2) 旧版隐式键：包含 "device" 视为 init，包含 "reset" 视为 reset，否则 infer
        返回：统一字典，至少包含 {"status": "ok"|"error"}，并尽量附带 "ok"/"type"/"request_id"
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "default")  # 默认为 infer
        payload = msg.get("payload", msg)  # 无 payload 时直接用顶层

        # 1) 显式类型路由
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "pong", "request_id": req_id}

        if mtype == "init":
            ok = bool(self._policy.init_infer(payload))
            if ok:
                return {"status": "ok", "ok": True, "type": "init_result", "request_id": req_id}
            return {"status": "error", "ok": False, "type": "init_result", "request_id": req_id,
                    "message": "Failed to initialize device"}

        if mtype == "reset":
            # 兼容不同字段名
            instr = payload.get("instruction") or payload.get("task_description")
            self._policy.reset(instr)
            return {"status": "ok", "ok": True, "type": "reset_result", "request_id": req_id}

        if mtype == "infer":
            data = self._policy.step(payload)
            return {"status": "ok", "ok": True, "type": "inference_result", "request_id": req_id, "data": data}

        # 2) 兼容旧版隐式键路由
        if "device" in msg:
            ok = bool(self._policy.init_infer(msg))
            if ok:
                return {"status": "ok", "ok": True, "type": "init_result", "request_id": req_id}
            return {"status": "error", "ok": False, "type": "init_result", "request_id": req_id,
                    "message": "Failed to initialize device"}

        if "reset" in msg:
            instr = msg.get("instruction") or msg.get("task_description")
            self._policy.reset(instr)
            return {"status": "ok", "ok": True, "type": "reset_result", "request_id": req_id}

        # 默认：推理 --> 消息转发就不要改动任何key-value 了
        # 借口会因为模型和 其他不一样而发生变化
        # 消息不能在这里比
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