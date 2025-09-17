import logging, argparse
import time, os
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        ## avoid any proxy interference --> only for cluster
        for k in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"):
            os.environ.pop(k, None)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                    open_timeout=150,  # adjust handshake timeout
                    ping_interval=20,
                    ping_timeout=20
                )
                # server first send metadata (msgpack binary)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(2)

    def init_device(self, device: str = "cuda") -> Dict:
        """send one device initialization message, verify protocol and service availability"""
        payload = {"device": device}
        self._ws.send(self._packer.pack(payload))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Server error (init_device):\n{resp}")
        return msgpack_numpy.unpackb(resp)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, instruction) -> None:
        payload = {"instruction": instruction, "reset": True}
        self._ws.send(self._packer.pack(payload))
        resp = self._ws.recv()
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def _build_argparser():
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional: API key for authentication")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="initialize device")
    ap.add_argument("--test", choices=["init", "infer"], default="infer", help="test mode: only initialize, or try simple inference")
    ap.add_argument("--log_level", default="INFO")
    return ap


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    # 1) device initialization (will not trigger model inference, suitable for health check)
    init_ret = client.init_device(args.device)
    logging.info("Init device resp: %s", init_ret)

    # 2) optional: try one simple inference (if the server policy.infer needs specific fields, it may return an error, which also proves the link is ok)
    if args.test == "infer":
        try:
            obs = {"request_id": "smoke-test", "ping": True} # give your model inputs


            infer_ret = client.infer(obs)
            logging.info("Infer resp: %s", infer_ret)
        except Exception as e:
            logging.error("Infer error (this still proves transport OK): %s", e)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    _main()