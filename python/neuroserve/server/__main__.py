"""Run the NeuroServe API server.

    python -m neuroserve.server --config configs/dev.yaml [--host 0.0.0.0] [--port 8000]
"""

import argparse
import os

import uvicorn

from .app import create_app
from .config import load_config
from .logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroServe OpenAI-compatible API server")
    parser.add_argument("--config", default=os.environ.get("NS_CONFIG", "configs/dev.yaml"))
    parser.add_argument("--host", help="overrides server.host from the config")
    parser.add_argument("--port", type=int, help="overrides server.port from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.format)
    # log_config=None: uvicorn uses NeuroServe's logging; access_log=False: NeuroServe writes its own.
    uvicorn.run(create_app(config), host=args.host or config.server.host,
                port=args.port or config.server.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
