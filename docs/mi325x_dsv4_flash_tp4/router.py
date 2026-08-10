"""Transparent two-backend OpenAI/SSE router for this tuning run.

It forwards each request unchanged to exactly one backend.  The only extra
endpoint is /router/status, used outside the workload to audit distribution.
"""
import asyncio
import os
from collections import defaultdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BACKENDS = os.environ.get("ROUTER_BACKENDS", "http://127.0.0.1:18000,http://127.0.0.1:18001").split(",")
POLICY = os.environ.get("ROUTER_POLICY", "round_robin")
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
client = httpx.AsyncClient(timeout=None, trust_env=False, limits=httpx.Limits(max_connections=4096, max_keepalive_connections=512))
lock = asyncio.Lock()
cursor = 0
stats = {backend: defaultdict(int) for backend in BACKENDS}


async def pick_backend():
    global cursor
    async with lock:
        if POLICY == "least_connections":
            backend = min(BACKENDS, key=lambda item: (stats[item]["active"], stats[item]["requests"], item))
        else:
            backend = BACKENDS[cursor % len(BACKENDS)]
            cursor += 1
        stats[backend]["active"] += 1
        stats[backend]["requests"] += 1
        return backend


@app.get("/router/status")
async def router_status():
    async with lock:
        return {"policy": POLICY, "backends": {backend: dict(values) for backend, values in stats.items()}}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    backend = await pick_backend()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "connection", "transfer-encoding"}}
    target = f"{backend}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    try:
        upstream_request = client.build_request(request.method, target, headers=headers, content=await request.body())
        upstream = await client.send(upstream_request, stream=True)
    except Exception:
        async with lock:
            stats[backend]["active"] -= 1
            stats[backend]["errors"] += 1
        return JSONResponse({"error": {"message": "upstream connection failure", "type": "router_error"}}, status_code=502)

    if upstream.status_code >= 400:
        async with lock:
            stats[backend]["http_errors"] += 1

    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}}

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except Exception:
            async with lock:
                stats[backend]["stream_errors"] += 1
            raise
        finally:
            await upstream.aclose()
            async with lock:
                stats[backend]["active"] -= 1

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=response_headers)
