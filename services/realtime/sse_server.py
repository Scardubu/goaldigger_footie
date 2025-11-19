"""
Minimal Server-Sent Events (SSE) server for live odds and match updates.
Run with: uvicorn services.realtime.sse_server:app --port 8079
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="GoalDiggers Realtime SSE")


async def event_stream() -> AsyncGenerator[str, None]:
    while True:
        # Mock odds delta
        payload = {
            "type": "odds",
            "matchId": "mock-123",
            "market": "1X2",
            "home": round(random.uniform(1.8, 2.5), 2),
            "draw": round(random.uniform(3.2, 3.8), 2),
            "away": round(random.uniform(3.0, 4.2), 2),
            "ts": int(time.time()),
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(2)


@app.get("/events")
async def sse(request: Request):
    generator = event_stream()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/sse-health")
async def sse_health():
    """Health check endpoint for SSE server."""
    return {"status": "OK", "message": "SSE server is running"}


@app.get("/health")
async def health():
    """Alternative health check endpoint."""
    return {"status": "OK", "message": "SSE server is running"}
    
@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {"status": "OK", "message": "SSE server root endpoint"}
