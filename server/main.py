"""
GradScope FastAPI Application & WebSocket Server
Streams live training updates (loss, graph nodes, gradients, manifold coordinates)
to the React visualizer frontend.
"""

from __future__ import annotations
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from server.schemas import ClientMessage, TrainingConfig
from server.training import TrainingSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradScopeServer")

app = FastAPI(
    title="GradScope API",
    description="Live training stream backend for from-scratch autograd visualizer"
)

# Enable CORS for local Vite dev server and any host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "engine": "GradScope Pure Autograd",
        "version": "1.0.0"
    }


@app.websocket("/ws/train")
async def websocket_train_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/train")

    session = TrainingSession()
    
    # Send initial step 0 snapshot immediately on connection
    init_snap = session.build_snapshot()
    await websocket.send_text(init_snap.model_dump_json())

    # Task to handle incoming messages from client
    async def receive_loop():
        try:
            while True:
                data_text = await websocket.receive_text()
                try:
                    data = json.loads(data_text)
                    action = data.get("action")
                    cfg_data = data.get("config")

                    if action == "start":
                        session.is_running = True
                        logger.info("Training started")

                    elif action == "pause":
                        session.is_running = False
                        logger.info("Training paused")

                    elif action == "step":
                        session.is_running = False
                        snap = session.step()
                        await websocket.send_text(snap.model_dump_json())

                    elif action == "reset":
                        session.is_running = False
                        session.reset()
                        snap = session.build_snapshot()
                        await websocket.send_text(snap.model_dump_json())
                        logger.info("Training reset")

                    elif action == "config" and cfg_data:
                        new_cfg = TrainingConfig(**cfg_data)
                        was_reset = session.update_config(new_cfg)
                        snap = session.build_snapshot()
                        await websocket.send_text(snap.model_dump_json())
                        logger.info(f"Config updated (reset={was_reset}): {new_cfg.dataset}")

                except Exception as e:
                    logger.error(f"Error handling client message: {e}")

        except WebSocketDisconnect:
            logger.info("Client disconnected from receive loop")

    # Task to step and stream snapshots when running
    async def stream_loop():
        try:
            while True:
                if session.is_running:
                    snap = session.step()
                    await websocket.send_text(snap.model_dump_json())
                    delay = max(session.config.step_delay_ms / 1000.0, 0.01)
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            logger.info("Client disconnected from stream loop")

    rec_task = asyncio.create_task(receive_loop())
    stm_task = asyncio.create_task(stream_loop())

    done, pending = await asyncio.wait(
        [rec_task, stm_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    logger.info("WebSocket connection closed cleanly")
