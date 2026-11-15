"""Run the NeuroServe API server.

    python -m neuroserve.server --config configs/dev.yaml [--host 0.0.0.0] [--port 8000]
"""

import argparse
import logging
import os

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroServe OpenAI-compatible API server")
    parser.add_argument("--config", default=os.environ.get("NS_CONFIG", "configs/dev.yaml"))
    parser.add_argument("--host", help="overrides server.host from the config")
    parser.add_argument("--port", type=int, help="overrides server.port from the config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    uvicorn.run(create_app(config), host=args.host or config.server.host,
                port=args.port or config.server.port, log_level="info")


if __name__ == "__main__":
    main()
