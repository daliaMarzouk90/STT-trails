"""
STT (Speech-to-Text) Package

This package contains various STT model implementations that inherit from BaseSTT.

Available STT Models:
- DummySTT: Test implementation for interface validation
- WhisperSTT: OpenAI Whisper implementation (local + API)
"""

from .stt_base import BaseSTT, STTResult, DummySTT

# Import WhisperSTT with error handling
try:
    from .whisper_stt import WhisperSTT
    __all__ = ['BaseSTT', 'STTResult', 'DummySTT', 'WhisperSTT']
except ImportError as e:
    # WhisperSTT dependencies not available
    __all__ = ['BaseSTT', 'STTResult', 'DummySTT']