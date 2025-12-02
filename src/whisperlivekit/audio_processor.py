import asyncio
import logging
import traceback
from time import time
from typing import Any, AsyncGenerator, List, Optional, Union

import numpy as np

from whisperlivekit.core import TranscriptionEngine, online_factory
from whisperlivekit.silero_vad_iterator import (FixedVADIterator,
                                                load_silero_vad)
from whisperlivekit.timed_objects import (ASRToken, ChangeSpeaker, FrontData,
                                          Segment, Silence, State, Transcript)
from whisperlivekit.tokens_alignment import TokensAlignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object()  # unique sentinel object for end of stream marker
MIN_DURATION_REAL_SILENCE = 5


async def get_all_from_queue(queue: asyncio.Queue) -> Union[object, Silence, np.ndarray, List[Any]]:
    items: List[Any] = []

    first_item = await queue.get()
    queue.task_done()
    if first_item is SENTINEL:
        return first_item
    if isinstance(first_item, Silence):
        return first_item
    items.append(first_item)

    while True:
        if not queue._queue:
            break
        next_item = queue._queue[0]
        if next_item is SENTINEL:
            break
        if isinstance(next_item, Silence):
            break
        items.append(await queue.get())
        queue.task_done()
    if isinstance(items[0], np.ndarray):
        return np.concatenate(items)
    else:
        return items


class AudioProcessor:
    """
    Processes audio streams for transcription.
    Handles audio processing, state management, and result formatting.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the audio processor with configuration, models, and state."""

        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)

        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = True  # PCM-only support

        # State management
        self.is_stopping: bool = False
        self.current_silence: Optional[Silence] = None
        self.state: State = State()
        self.lock: asyncio.Lock = asyncio.Lock()
        self.sep: str = " "  # Default separator
        self.last_response_content: FrontData = FrontData()

        self.tokens_alignment: TokensAlignment = TokensAlignment(self.state, self.args, self.sep)
        self.beg_loop: Optional[float] = None

        # Models and processing
        self.asr: Any = models.asr
        self.vac: Optional[FixedVADIterator] = None
        if self.args.vac:
            # Give each processor its own VAD model to avoid shared state across clients.
            vac_model_instance = load_silero_vad(onnx=self.args.vac_onnx)
            self.vac = FixedVADIterator(vac_model_instance)

        self.transcription_queue: Optional[asyncio.Queue] = asyncio.Queue() if self.args.transcription else None
        self.pcm_buffer: bytearray = bytearray()
        self.total_pcm_samples: int = 0
        self.transcription_task: Optional[asyncio.Task] = None
        self.watchdog_task: Optional[asyncio.Task] = None
        self.all_tasks_for_cleanup: List[asyncio.Task] = []

        self.transcription: Optional[Any] = None

        if self.args.transcription:
            self.transcription = online_factory(self.args, models.asr)
            self.sep = self.transcription.asr.sep

    async def _push_silence_event(self) -> None:
        if self.transcription_queue:
            await self.transcription_queue.put(self.current_silence)

    async def _begin_silence(self) -> None:
        if self.current_silence:
            return
        now = time() - self.beg_loop
        self.current_silence = Silence(
            is_starting=True, start=now
        )
        await self._push_silence_event()

    async def _end_silence(self) -> None:
        if not self.current_silence:
            return
        now = time() - self.beg_loop
        self.current_silence.end = now
        self.current_silence.is_starting = False
        self.current_silence.has_ended = True
        self.current_silence.compute_duration()
        if self.current_silence.duration > MIN_DURATION_REAL_SILENCE:
            self.state.new_tokens.append(self.current_silence)
        await self._push_silence_event()
        self.current_silence = None

    async def _enqueue_active_audio(self, pcm_chunk: np.ndarray) -> None:
        if pcm_chunk is None or pcm_chunk.size == 0:
            return
        if self.transcription_queue:
            await self.transcription_queue.put(pcm_chunk.copy())

    def _slice_before_silence(self, pcm_array: np.ndarray, chunk_sample_start: int, silence_sample: Optional[int]) -> Optional[np.ndarray]:
        if silence_sample is None:
            return None
        relative_index = int(silence_sample - chunk_sample_start)
        if relative_index <= 0:
            return None
        split_index = min(relative_index, len(pcm_array))
        if split_index <= 0:
            return None
        return pcm_array[:split_index]

    def convert_pcm_to_float(self, pcm_buffer: Union[bytes, bytearray]) -> np.ndarray:
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def get_current_state(self) -> State:
        """Get current state."""
        async with self.lock:
            current_time = time()

            remaining_transcription = 0
            if self.state.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.state.end_buffer, 1))

            self.state.remaining_time_transcription = remaining_transcription
            return self.state

    async def transcription_processor(self) -> None:
        """Process audio chunks for transcription."""
        cumulative_pcm_duration_stream_time = 0.0

        while True:
            try:
                # item = await self.transcription_queue.get()
                item = await get_all_from_queue(self.transcription_queue)
                if item is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    break

                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.state.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                new_tokens = []
                current_audio_processed_upto = self.state.end_buffer

                if isinstance(item, Silence):
                    if item.is_starting:
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                            self.transcription.start_silence
                        )
                        asr_processing_logs += f" + Silence starting"
                    if item.has_ended:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                        cumulative_pcm_duration_stream_time += item.duration
                        current_audio_processed_upto = cumulative_pcm_duration_stream_time
                        self.transcription.end_silence(item.duration, self.state.tokens[-1].end if self.state.tokens else 0)
                    if self.state.tokens:
                        asr_processing_logs += f" | last_end = {self.state.tokens[-1].end} |"
                    logger.info(asr_processing_logs)
                    new_tokens = new_tokens or []
                    current_audio_processed_upto = max(current_audio_processed_upto, stream_time_end_of_current_pcm)
                elif isinstance(item, ChangeSpeaker):
                    self.transcription.new_speaker(item)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                    logger.info(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += len(pcm_array) / self.sample_rate
                    stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                    self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                    new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)
                    new_tokens = new_tokens or []

                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = _buffer_transcript.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.state.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)

                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)

                candidate_end_times.append(current_audio_processed_upto)

                async with self.lock:
                    self.state.tokens.extend(new_tokens)
                    self.state.buffer_transcription = _buffer_transcript
                    self.state.end_buffer = max(candidate_end_times)
                    self.state.new_tokens.extend(new_tokens)
                    self.state.new_tokens_buffer = _buffer_transcript

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL:  # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()

        if self.is_stopping:
            logger.info("Transcription processor finishing due to stopping flag.")

        logger.info("Transcription processor task finished.")

    async def results_formatter(self) -> AsyncGenerator[FrontData, None]:
        """Format processing results for output."""
        while True:
            try:
                self.tokens_alignment.update()
                lines = self.tokens_alignment.get_lines(
                    current_silence=self.current_silence
                )
                state = await self.get_current_state()

                buffer_transcription_text = state.buffer_transcription.text if state.buffer_transcription else ''

                response_status = "active_transcription"
                if not lines and not buffer_transcription_text:
                    response_status = "no_audio_detected"

                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription_text,
                    remaining_time_transcription=state.remaining_time_transcription,
                )

                should_push = (response != self.last_response_content)
                if should_push:
                    yield response
                    self.last_response_content = response

                if self.is_stopping and self._processing_tasks_done():
                    logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                    return

                await asyncio.sleep(0.05)

            except Exception as e:
                logger.warning(f"Exception in results_formatter. Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)

    async def create_tasks(self) -> AsyncGenerator[FrontData, None]:
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog: List[asyncio.Task] = []

        if self.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)

        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor: List[asyncio.Task]) -> None:
        """Monitors the health of critical processing tasks."""
        tasks_remaining: List[asyncio.Task] = [task for task in tasks_to_monitor if task]
        while True:
            try:
                if not tasks_remaining:
                    logger.info("Watchdog task finishing: all monitored tasks completed.")
                    return

                await asyncio.sleep(10)

                for i, task in enumerate(list(tasks_remaining)):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                        tasks_remaining.remove(task)

            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")
        self.is_stopping = True
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()

        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        logger.info("AudioProcessor cleanup complete.")

    def _processing_tasks_done(self) -> bool:
        """Return True when all active processing tasks have completed."""
        tasks_to_check = [
            self.transcription_task,
        ]
        return all(task.done() for task in tasks_to_check if task)

    async def process_audio(self, message: Optional[bytes]) -> None:
        """Process incoming audio data."""

        if not self.beg_loop:
            self.beg_loop = time()
            self.current_silence = Silence(start=0.0, is_starting=True)
            self.tokens_alignment.beg_loop = self.beg_loop

        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True

            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        self.pcm_buffer.extend(message)
        await self.handle_pcm_data()

    async def handle_pcm_data(self) -> None:
        # Process when enough data
        if len(self.pcm_buffer) < self.bytes_per_sec:
            return

        if len(self.pcm_buffer) > self.max_bytes_per_sec:
            logger.warning(
                f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                f"Consider using a smaller model."
            )

        chunk_size = min(len(self.pcm_buffer), self.max_bytes_per_sec)
        aligned_chunk_size = (chunk_size // self.bytes_per_sample) * self.bytes_per_sample

        if aligned_chunk_size == 0:
            return
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:aligned_chunk_size])
        self.pcm_buffer = self.pcm_buffer[aligned_chunk_size:]

        num_samples = len(pcm_array)
        chunk_sample_start = self.total_pcm_samples
        chunk_sample_end = chunk_sample_start + num_samples

        res = None
        if self.args.vac:
            res = self.vac(pcm_array)

        if res is not None:
            if "start" in res and self.current_silence:
                await self._end_silence()

            if "end" in res and not self.current_silence:
                pre_silence_chunk = self._slice_before_silence(
                    pcm_array, chunk_sample_start, res.get("end")
                )
                if pre_silence_chunk is not None and pre_silence_chunk.size > 0:
                    await self._enqueue_active_audio(pre_silence_chunk)
                await self._begin_silence()

        if not self.current_silence:
            await self._enqueue_active_audio(pcm_array)

        self.total_pcm_samples = chunk_sample_end

        if not self.args.transcription:
            await asyncio.sleep(0.1)
