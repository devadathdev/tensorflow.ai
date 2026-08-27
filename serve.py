#!/usr/bin/env python3
"""API server for TensorVision AI."""

import os
import sys
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.api.server import create_app, run_server
from src.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Run TensorVision AI API server")
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--host', type=str, help='Host to bind')
    parser.add_argument('--port', '-p', type=int, help='Port to bind')
    parser.add_argument('--workers', '-w', type=int, help='Number of workers')
    parser.add_argument('--reload', '-r', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    config = get_config(args.config)
    api_config = config.api
    
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or api_config.get('port', 8000)
    workers = args.workers or api_config.get('workers', 1)
    
    print(f"Starting TensorVision AI API server on {host}:{port}")
    print(f"Workers: {workers}")
    print(f"API docs: http://{host}:{port}/docs")
    print()
    
    run_server(host=host, port=port, workers=workers, reload=args.reload)


if __name__ == '__main__':
    main()