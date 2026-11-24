"""Serve the local Prism workspace."""

import argparse
import os
from pathlib import Path
import warnings

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .paths import default_directory


def create_app(directory):
    # Colour announces optional plotting and scientific extras during import.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r'"(?:SciPy|Matplotlib)" related API features are not available:',
                                module=r"colour\.utilities\.verbose")
        from .workspace.browser_writes import BrowserWriteOrigin
        from .workspace.storage import LocalState
        from .workspace.routes import workspace_router
        from .render.routes import render_router

    state = LocalState(Path(directory).expanduser().resolve())
    app = FastAPI(title="Prism", docs_url=None, redoc_url=None)
    app.add_middleware(BrowserWriteOrigin)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "::1"])
    app.include_router(workspace_router(state))
    app.include_router(render_router(state))
    app.mount("/static", StaticFiles(directory=state.assets / "static"), name="static")

    @app.exception_handler(ValueError)
    async def invalid_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    return app


class WorkspaceServer(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started:
            port = self.servers[0].sockets[0].getsockname()[1]
            print(f"Open Prism at http://127.0.0.1:{port}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=default_directory())
    args = parser.parse_args()
    os.environ["PRISM_CALIBRATION_DIR"] = str(args.data_dir.resolve() / "calibration")
    config = uvicorn.Config(create_app(args.data_dir), host="127.0.0.1", port=args.port,
                            log_level="warning", access_log=False)
    WorkspaceServer(config).run()
