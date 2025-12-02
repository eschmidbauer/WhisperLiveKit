import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whisperlivekit import AudioProcessor, TranscriptionEngine  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DEFAULT_CONFIG = {
    # Server
    "host": "localhost",
    "port": 8000,
    "log_level": "INFO",
    # Audio/VAD
    "min_chunk_size": 0.1,
    "vac": True,
    "vac_onnx": False,
    "vac_chunk_size": 0.04,
    # Model
    "model_size": "base",
    "model_path": None,
    "model_dir": None,
    "model_cache_dir": None,
    "lora_path": None,
    "lan": "auto",
    "warmup_file": None,
    # Simul-Whisper
    "backend": "auto",
    "disable_fast_encoder": False,
    "custom_alignment_heads": None,
    "frame_threshold": 25,
    "beams": 1,
    "decoder_type": None,
    "audio_max_len": 20.0,
    "audio_min_len": 0.0,
    "cif_ckpt_path": None,
    "never_fire": False,
    "init_prompt": None,
    "static_init_prompt": None,
    "max_context_tokens": None,
}

args = SimpleNamespace(**DEFAULT_CONFIG)
transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(**DEFAULT_CONFIG)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return {"status": "ok"}


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results.")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for launching the FastAPI server."""
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level=args.log_level.lower(),
        lifespan="on",
    )


if __name__ == "__main__":
    main()
