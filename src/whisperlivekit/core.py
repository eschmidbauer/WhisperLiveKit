import logging
from argparse import Namespace

from whisperlivekit.silero_vad_iterator import load_silero_vad
from whisperlivekit.simul_whisper import (SimulStreamingASR,
                                          SimulStreamingOnlineProcessor)


def update_with_kwargs(_dict, kwargs):
    _dict.update({
        k: v for k, v in kwargs.items() if k in _dict
    })
    return _dict


logger = logging.getLogger(__name__)

class TranscriptionEngine:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs):
        if TranscriptionEngine._initialized:
            return

        global_params = {
            "host": "localhost",
            "port": 8000,
            "vac": True,
            "vac_onnx": False,
            "vac_chunk_size": 0.04,
            "log_level": "DEBUG",
            "transcription": True,
            "pcm_input": True,
            "backend": "auto",
        }
        global_params = update_with_kwargs(global_params, kwargs)

        transcription_common_params = {
            "warmup_file": None,
            "min_chunk_size": 0.1,
            "model_size": "base",
            "model_cache_dir": None,
            "model_dir": None,
            "model_path": None,
            "lora_path": None,
            "lan": "auto",
            "direct_english_translation": False,
        }
        transcription_common_params = update_with_kwargs(transcription_common_params, kwargs)                                            

        if transcription_common_params['model_size'].endswith(".en"):
            transcription_common_params["lan"] = "en"

        self.args = Namespace(**{**global_params, **transcription_common_params})
        
        self.asr = None
        self.tokenizer = None
        self.vac_model = None
        
        if self.args.vac:
            use_onnx = kwargs.get("vac_onnx", False)
            self.vac_model = load_silero_vad(onnx=use_onnx)
        
        if self.args.transcription:
            simulstreaming_params = {
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
            simulstreaming_params = update_with_kwargs(simulstreaming_params, kwargs)
            
            self.tokenizer = None
            self.asr = SimulStreamingASR(
                **transcription_common_params,
                **simulstreaming_params,
                backend=self.args.backend,
            )
            logger.info(
                "Using SimulStreaming policy with %s backend",
                getattr(self.asr, "encoder_backend", "whisper"),
            )
        TranscriptionEngine._initialized = True


def online_factory(args, asr):
    return SimulStreamingOnlineProcessor(asr)
