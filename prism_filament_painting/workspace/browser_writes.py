"""Reject browser writes from outside the request's own host and port."""

import re

from starlette.datastructures import Headers
from starlette.responses import JSONResponse


# A serialized HTTP origin contains only a scheme and an authority.
_ORIGIN = re.compile(r"(https?)://(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:[0-9]+)?", re.I)


def _matches_host(origin, hosts):
    match = _ORIGIN.fullmatch(origin)
    if match is None or len(hosts) != 1:
        return False
    scheme, hostname, port = match.groups()
    default = ":443" if scheme.lower() == "https" else ":80"
    authority = hostname + (port or "")
    return authority.lower().removesuffix(default) == hosts[0].lower().removesuffix(default)


class BrowserWriteOrigin:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            headers = Headers(scope=scope)
            sites = headers.getlist("sec-fetch-site")
            origins = headers.getlist("origin")
            if ((sites and sites != ["same-origin"]) or
                    (origins and (len(origins) != 1 or not _matches_host(origins[0], headers.getlist("host"))))):
                response = JSONResponse({"detail": "Submit this change from the Prism page."}, status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
