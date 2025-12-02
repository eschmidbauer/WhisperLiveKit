"""Example client that streams 16 kHz PCM audio to a running `wlk` server."""
import argparse
import asyncio
import contextlib
import json
import sys
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional

from loguru import logger

try:
    import websockets
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Install the `websockets` package to run example_client.py (`pip install websockets`)."
    ) from exc


SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream audio to /asr WebSocket endpoint.")
    parser.add_argument("--host", default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--wav", type=Path, help="Path to a 16 kHz mono WAV file to stream.")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Stream live audio from microphone (requires sounddevice).",
    )
    parser.add_argument(
        "--mic-device",
        help="Microphone device name or index (see --list-mics). Defaults to sounddevice's default input.",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available input devices and exit.",
    )
    parser.add_argument("--chunk-ms", type=int, default=20, help="Chunk size in milliseconds.")
    parser.add_argument(
        "--print-interim",
        action="store_true",
        default=True,
        help="Print interim buffer transcription messages (default: enabled).",
    )
    parser.add_argument(
        "--log-chunks",
        action="store_true",
        help="Log each chunk size while streaming (debug aid).",
    )
    return parser.parse_args()


def _validate_wav(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.getnframes()
    if sr != SAMPLE_RATE or ch != 1 or width != 2:
        raise SystemExit("WAV must be mono, 16-bit, 16 kHz.")
    return sr, frames, width


async def wav_chunks(path: Path, chunk_ms: int, log_chunks: bool = False) -> AsyncGenerator[bytes, None]:
    """Yield 16-bit PCM chunks from a WAV file, paced to real time."""
    with wave.open(str(path), "rb") as wf:
        frames_per_chunk = int(SAMPLE_RATE * chunk_ms / 1000)
        idx = 0
        start = asyncio.get_event_loop().time()
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            if log_chunks:
                logger.debug(f"[WAV] chunk#{idx} bytes={len(data)}")
            yield data
            idx += 1
            # Pace to real time to mirror microphone streaming
            target = start + idx * (chunk_ms / 1000)
            now = asyncio.get_event_loop().time()
            sleep_for = target - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


def _parse_device(device: Optional[str]):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


async def mic_chunks(chunk_ms: int, device: Optional[str] = None, log_chunks: bool = False) -> AsyncGenerator[bytes, None]:
    """Yield 16-bit PCM frames from a microphone device."""
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install `sounddevice` for microphone streaming (`pip install sounddevice`).")

    q: asyncio.Queue[bytes] = asyncio.Queue()
    frames_per_chunk = int(SAMPLE_RATE * chunk_ms / 1000)
    loop = asyncio.get_running_loop()
    device_spec = _parse_device(device)

    if device_spec is not None:
        logger.info(f"Using microphone device: {device_spec}")
    else:
        logger.info("Using default microphone device.")

    def callback(indata, frames, _time, status):
        if status:
            logger.warning(f"Input status: {status}")
        payload = bytes(indata)
        if log_chunks:
            logger.debug(f"[MIC] chunk frames={frames} bytes={len(payload)}")
        loop.call_soon_threadsafe(q.put_nowait, payload)

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=frames_per_chunk,
        device=device_spec,
        callback=callback,
    )
    stream.start()
    try:
        while True:
            data = await q.get()
            yield data
    finally:
        stream.stop()
        stream.close()


async def receiver(ws, print_interim: bool, stop_event: asyncio.Event) -> None:
    """Consume messages from the server until it signals completion."""
    last_final = ""
    try:
        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.info(f"Non-JSON message: {message}")
                continue

            msg_type = payload.get("type")
            if msg_type == "config":
                logger.info(f"Server config: {payload}")
                continue
            if msg_type == "ready_to_stop":
                logger.info("Server indicated end of stream.")
                stop_event.set()
                break
            if msg_type == "asr_result":
                text = payload.get("text", "").strip()
                if text:
                    if payload.get("final"):
                        last_final = text
                        logger.info(f"[FINAL] {text}")
                    elif print_interim:
                        logger.info(f"[BUFFER] {text}")
                continue

            if payload.get("error"):
                logger.error(f"Server error: {payload['error']}")

            status = payload.get("status")
            if status == "no_audio_detected":
                logger.info("No audio detected yet...")

            buffer_text = payload.get("buffer_transcription", "")
            lines = payload.get("lines", [])
            if lines:
                text = " ".join(line.get("text", "") for line in lines).strip()
                if text and text != last_final:
                    last_final = text
                    logger.info(f"[FINAL] {text}")
            elif print_interim and buffer_text:
                logger.info(f"[BUFFER] {buffer_text.strip()}")
    finally:
        stop_event.set()


async def stream_audio(ws, source: AsyncGenerator[bytes, None]) -> None:
    async for lin16 in source:
        # Server expects raw 16-bit linear PCM.
        await ws.send(lin16)


async def main() -> None:
    args = parse_args()

    if args.list_mics:
        try:
            import sounddevice as sd
        except ImportError:
            raise SystemExit("Install `sounddevice` to list microphone devices (`pip install sounddevice`).")
        devices_raw = sd.query_devices()
        # Normalize to plain dicts for type-checkers and easier access
        devices = [dict(d) for d in devices_raw] if devices_raw is not None else []
        logger.info("Available input devices:")
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                default = ""
                if idx == sd.default.device[0]:
                    default = " (default)"
                logger.info(f"[{idx}] {dev['name']} - max_input_channels={dev['max_input_channels']}{default}")
        return

    if args.wav:
        sr, frames, width = _validate_wav(args.wav)
        duration = frames / sr if sr else 0
        logger.info(f"WAV validated: sr={sr}Hz channels=1 width={width * 8}-bit duration≈{duration:.2f}s")

    if not args.wav and not args.mic:
        raise SystemExit("Provide --wav path or --mic to stream audio.")

    uri = f"ws://{args.host}:{args.port}/asr"
    logger.info(f"Connecting to {uri}")
    async with websockets.connect(uri, max_size=None) as ws:
        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receiver(ws, args.print_interim, stop_event))
        source: Optional[AsyncGenerator[bytes, None]] = None
        if args.wav:
            source = wav_chunks(args.wav, args.chunk_ms, log_chunks=args.log_chunks)
        else:
            source = mic_chunks(args.chunk_ms, args.mic_device, log_chunks=args.log_chunks)

        try:
            await stream_audio(ws, source)
            # Send an empty binary frame to signal the end of the stream (expected by wlk with --pcm-input)
            await ws.send(b"")
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Stopping on user interrupt.")
            with contextlib.suppress(Exception):
                await ws.send(b"")
            stop_event.set()
        finally:
            if not recv_task.done():
                recv_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await recv_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
