import dataclasses
import logging
import socket

# from hume.models import HumePolicy
from tools.websocket_policy_server import WebsocketPolicyServer
from tools.model_interface import QwenpiPolicyInterfence


def main(args) -> None:

    inferencer = QwenpiPolicyInterfence(
        saved_model_path=args.ckpt_path,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
        cfg_scale=args.cfg_scale,
        use_bf16=args.use_bf16,
        action_ensemble=args.action_ensemble,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=inferencer,
        host="0.0.0.0",  # represent listen any external/internal access
        port=args.port,
        metadata={},
    )
    logging.info("server running")
    server.serve_forever()


import argparse


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="results/Checkpoints/1_need/QWenDiT-Vanilla/checkpoints/steps_20000_pytorch_model.pt",
    )
    parser.add_argument("--unnorm_key", type=str, default="bridge_dataset")
    parser.add_argument("--image_size", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", type=bool, default=False)  #
    parser.add_argument("--action_ensemble", type=bool, default=False)
    parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)

    args = parser.parse_args()
    return args


def start_debugpy_once():
    """only start once debugpy"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser()
    # start_debugpy_once()
    main(args)
