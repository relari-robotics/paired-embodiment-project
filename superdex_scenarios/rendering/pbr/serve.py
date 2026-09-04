"""Serve and open the standalone PBR replay viewer."""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as server:
        server.daemon_threads = True
        url = f"http://127.0.0.1:{args.port}/superdex_scenarios/rendering/pbr/"
        print(f"PBR replay viewer: {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
