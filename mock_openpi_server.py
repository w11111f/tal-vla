#!/usr/bin/env python3
import argparse
import asyncio
import logging

import msgpack
import websockets


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal websocket policy server for smoke tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=1)
    return parser


async def _handle_connection(websocket: websockets.ServerConnection, *, action_dim: int, action_horizon: int) -> None:
    metadata = {"mock": True, "action_dim": action_dim, "action_horizon": action_horizon}
    await websocket.send(msgpack.packb(metadata, use_bin_type=True))

    async for message in websocket:
        if isinstance(message, str):
            logging.warning("Ignoring unexpected text frame from client: %s", message[:200])
            continue

        response = {
            "actions": [[0.0] * action_dim for _ in range(action_horizon)],
        }
        await websocket.send(msgpack.packb(response, use_bin_type=True))


async def _main() -> None:
    args = _build_argparser().parse_args()
    async with websockets.serve(
        lambda ws: _handle_connection(ws, action_dim=args.action_dim, action_horizon=args.action_horizon),
        args.host,
        args.port,
        max_size=None,
        compression=None,
    ):
        logging.info("Mock OpenPI server listening on ws://%s:%s", args.host, args.port)
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(_main())
