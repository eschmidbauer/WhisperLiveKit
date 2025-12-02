from time import time
from typing import List, Optional

from whisperlivekit.timed_objects import (ASRToken, Segment, Silence,
                                          SilentSegment)


class TokensAlignment:
    """Minimal alignment helper to build front-end segments from ASR tokens."""

    def __init__(self, state: any, _args: any, sep: Optional[str]) -> None:
        self.state = state
        self.sep = sep if sep is not None else " "
        self.beg_loop: Optional[float] = None
        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.new_tokens: List[ASRToken] = []

    def update(self) -> None:
        """Drain state buffers into local alignment buffers."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []

    def _flush_current_line(self) -> None:
        if not self.current_line_tokens:
            return
        segment = Segment.from_tokens(self.current_line_tokens)
        if segment:
            self.validated_segments.append(segment)
        self.current_line_tokens = []

    def get_lines(self, current_silence: Optional[Silence] = None) -> List[Segment]:
        """Return validated segments, including active silence if present."""
        for token in self.new_tokens:
            if token.is_silence():
                self._flush_current_line()
                end_silence = token.end if token.has_ended else (time() - self.beg_loop if self.beg_loop else token.end)
                start_silence = token.start if token.start is not None else 0
                self.validated_segments.append(
                    SilentSegment(start=start_silence, end=end_silence)
                )
            else:
                self.current_line_tokens.append(token)

        segments = list(self.validated_segments)
        if self.current_line_tokens:
            segment = Segment.from_tokens(self.current_line_tokens)
            if segment:
                segments.append(segment)

        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else time() - self.beg_loop
            if segments and segments[-1].is_silence():
                segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
            else:
                segments.append(
                    SilentSegment(start=current_silence.start, end=end_silence)
                )

        return segments
