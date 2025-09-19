import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple
import numpy as np

from typing_extensions import override

from tools.websocket_policy_client import WebsocketClientPolicy

# class WebsocketClientPolicy:
#     """Implements the Policy interface by communicating with a server over websocket."""
#     def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
#         # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
#         self._uri = f"ws://{host}"
#         if port is not None:
#             self._uri += f":{port}"
#         self._packer = msgpack_numpy.Packer()
#         self._api_key = api_key
#         self._ws, self._server_metadata = self._wait_for_server()

#     def get_server_metadata(self) -> Dict:
#         return self._server_metadata

#     def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
#         logging.info(f"Waiting for server at {self._uri}...")
#         # avoid any proxy interference
#         for k in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"):
#             os.environ.pop(k, None)
#         while True:
#             try:
#                 headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
#                 conn = websockets.sync.client.connect(
#                     self._uri, compression=None, max_size=None, additional_headers=headers
#                 )
#                 # server first send metadata (msgpack binary)
#                 metadata = msgpack_numpy.unpackb(conn.recv())
#                 return conn, metadata
#             except ConnectionRefusedError:
#                 logging.info("Still waiting for server...")
#                 time.sleep(2)

#     def init_device(self, device: str = "cuda") -> Dict:
#         """send one device initialization message, verify protocol and service availability"""
#         payload = {"device": device}
#         self._ws.send(self._packer.pack(payload))
#         resp = self._ws.recv()
#         if isinstance(resp, str):
#             raise RuntimeError(f"Server error (init_device):\n{resp}")
#         return msgpack_numpy.unpackb(resp)

#     @override
#     def infer(self, obs: Dict) -> Dict:  # noqa: UP006
#         data = self._packer.pack(obs)
#         self._ws.send(data)
#         response = self._ws.recv()
#         if isinstance(response, str):
#             # server will send text stack when exception, here directly throw
#             raise RuntimeError(f"Error in inference server:\n{response}")
#         return msgpack_numpy.unpackb(response)

#     @override
#     def reset(self, instruction) -> None:
#         payload = {"instruction": instruction, "reset": True}
#         self._ws.send(self._packer.pack(payload))
#         resp = self._ws.recv()
#         pass

#     def close(self) -> None:
#         try:
#             self._ws.close()
#         except Exception:
#             pass


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional: API key for authentication")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="initialize device")
    ap.add_argument(
        "--test", choices=["init", "infer"], default="infer", help="test mode: only initialize, or try simple inference"
    )
    ap.add_argument("--log_level", default="INFO")
    return ap


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    # 1) device initialization
    init_ret = client.init_device(args.device)  # here to set some things on the server
    logging.info("Init device resp: %s", init_ret)

    # 2) optional: try one simple inference
    if args.test == "infer":
        try:
            # build observation aligned with model API
            H, W = 224, 224
            img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            wrist_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            state = np.zeros((7,), dtype=np.float32)  # [x,y,z, ax,ay,az, gripper]

            observation = {  # key to align with model API
                "request_id": "smoke-test",
                "observation.primary": np.expand_dims(img, axis=0),  # (1,H,W,C), uint8 0-255
                "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (1,H,W,C)
                "observation.state": np.expand_dims(state, axis=0),  # (1,7), float32
                "instruction": ["debug: pick up the red block"],  # single element list
            }

            obs = {
                "request_id": "smoke-test",
                "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                "task_description": observation["instruction"][0],  # assume only one task description
            }

            infer_ret = client.infer(obs)
            logging.info("Infer resp: %s", infer_ret)
        except Exception as e:
            logging.error("Infer error (this still proves transport OK): %s", e)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    _main()
