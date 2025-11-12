#!/usr/bin/env python3
"""
Whisper STT Implementation

OpenAI Whisper speech-to-text implementation using the static BaseSTT interface.
Supports both local Whisper models and OpenAI API calls.

Usage:
    from stt.whisper_stt import WhisperSTT
    
    # Load model (local)
    WhisperSTT.load_model()
    
    # Transcribe audio
    result = WhisperSTT.transcribe_file("audio.wav")
    print(result.text)
    
    # Or use OpenAI API
    WhisperSTT.load_model(use_api=True, api_key="your-key")
    result = WhisperSTT.transcribe_file("audio.wav")
"""

from typing import Union, Optional, Dict, Any
import numpy as np
from pathlib import Path
import time
import logging
import tempfile
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .stt_base import BaseSTT, STTResult

logger = logging.getLogger(__name__)


class WhisperSTT(BaseSTT):
    """
    OpenAI Whisper STT implementation with support for both local models and API.
    
    Supports:
    - Local Whisper models (tiny, base, small, medium, large)
    - OpenAI Whisper API calls
    - Multiple audio formats via soundfile
    - Confidence scoring and metadata
    """
    
    model_name = "WhisperSTT"
    model = None
    is_loaded = False
    config = {
        "model_size": "base",
        "use_api": False,
        "api_key": None,
        "language": None,  # Auto-detect if None
        "task": "transcribe",  # "transcribe" or "translate"
        "temperature": 0.0,
        "best_of": 5,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "suppress_tokens": "-1",
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "fp16": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6
    }
    
    @classmethod
    def load_model(cls, 
                   model_size: str = "base",
                   use_api: bool = False,
                   api_key: Optional[str] = None,
                   **kwargs) -> None:
        """
        Load the Whisper model (local or API setup).
        
        Args:
            model_size: Size of local model ("tiny", "base", "small", "medium", "large")
            use_api: Use OpenAI API instead of local model
            api_key: OpenAI API key (required if use_api=True)
            **kwargs: Additional Whisper parameters
        """
        cls.config.update({
            "model_size": model_size,
            "use_api": use_api,
            "api_key": api_key,
            **kwargs
        })
        
        if use_api:
            cls._load_api_model(api_key)
        else:
            cls._load_local_model(model_size)
    
    @classmethod
    def _load_local_model(cls, model_size: str) -> None:
        """Load local Whisper model."""
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "OpenAI Whisper not installed. Install with: pip install openai-whisper"
            )
        
        logger.info(f"Loading Whisper local model: {model_size}")
        start_time = time.time()
        
        try:
            cls.model = whisper.load_model(model_size)
            cls.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{model_size}' loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            cls.is_loaded = False
            raise RuntimeError(f"Failed to load Whisper model '{model_size}': {e}")
    
    @classmethod
    def _load_api_model(cls, api_key: Optional[str]) -> None:
        """Setup OpenAI API client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI Python client not installed. Install with: pip install openai"
            )
        
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
        
        logger.info("Setting up OpenAI Whisper API client")
        
        try:
            openai.api_key = api_key
            cls.model = "whisper-1"  # API model identifier
            cls.is_loaded = True
            logger.info("OpenAI Whisper API client configured successfully")
            
        except Exception as e:
            cls.is_loaded = False
            raise RuntimeError(f"Failed to setup OpenAI API: {e}")
    
    @classmethod
    def transcribe_audio(cls, 
                        audio_data: Union[np.ndarray, str, Path], 
                        sample_rate: Optional[int] = None) -> STTResult:
        """
        Transcribe audio using Whisper (local or API).
        
        Args:
            audio_data: Audio input (numpy array, file path, or audio file)
            sample_rate: Sample rate for numpy arrays
            
        Returns:
            STTResult: Transcription with confidence and metadata
        """
        if not cls.is_loaded:
            raise RuntimeError("Whisper model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            if cls.config["use_api"]:
                result = cls._transcribe_api(audio_data, sample_rate)
            else:
                result = cls._transcribe_local(audio_data, sample_rate)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Whisper transcription failed: {e}")
    
    @classmethod
    def _transcribe_local(cls, audio_data: Union[np.ndarray, str, Path], sample_rate: Optional[int]) -> STTResult:
        """Transcribe using local Whisper model."""
        
        # Prepare transcription options
        transcribe_options = {
            "language": cls.config.get("language"),
            "task": cls.config.get("task", "transcribe"),
            "temperature": cls.config.get("temperature", 0.0),
            "best_of": cls.config.get("best_of", 5),
            "beam_size": cls.config.get("beam_size", 5),
            "patience": cls.config.get("patience", 1.0),
            "length_penalty": cls.config.get("length_penalty", 1.0),
            "suppress_tokens": cls.config.get("suppress_tokens", "-1"),
            "initial_prompt": cls.config.get("initial_prompt"),
            "condition_on_previous_text": cls.config.get("condition_on_previous_text", True),
            "fp16": cls.config.get("fp16", True),
            "compression_ratio_threshold": cls.config.get("compression_ratio_threshold", 2.4),
            "logprob_threshold": cls.config.get("logprob_threshold", -1.0),
            "no_speech_threshold": cls.config.get("no_speech_threshold", 0.6)
        }
        
        # Remove None values
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        # Handle numpy arrays
        if isinstance(audio_data, np.ndarray):
            audio_input = audio_data.astype(np.float32)
            # Whisper expects mono audio
            if audio_input.ndim > 1:
                audio_input = np.mean(audio_input, axis=1)
        else:
            # File path
            audio_input = str(audio_data)
        
        # Transcribe
        result = cls.model.transcribe(audio_input, **transcribe_options)
        
        # Calculate confidence (average of segment confidences if available)
        confidence = None
        if "segments" in result and result["segments"]:
            segment_confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # Convert log prob to confidence estimate
                    conf = min(1.0, max(0.0, np.exp(segment["avg_logprob"])))
                    segment_confidences.append(conf)
            
            if segment_confidences:
                confidence = np.mean(segment_confidences)
        
        # Prepare metadata
        metadata = {
            "model": cls.config["model_size"],
            "language": result.get("language"),
            "task": cls.config["task"],
            "segments": len(result.get("segments", [])),
            "api_used": False
        }
        
        return STTResult(
            text=result["text"].strip(),
            confidence=confidence,
            metadata=metadata
        )
    
    @classmethod
    def _transcribe_api(cls, audio_data: Union[np.ndarray, str, Path], sample_rate: Optional[int]) -> STTResult:
        """Transcribe using OpenAI API."""
        
        # Handle numpy arrays - save to temp file for API
        if isinstance(audio_data, np.ndarray):
            if not SOUNDFILE_AVAILABLE:
                raise ImportError("soundfile required for numpy array support. Install with: pip install soundfile")
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                sf.write(temp_path, audio_data, sample_rate or 16000)
                audio_file_path = temp_path
                cleanup_temp = True
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise RuntimeError(f"Failed to save temporary audio file: {e}")
        else:
            audio_file_path = str(audio_data)
            cleanup_temp = False
        
        try:
            # Make API call
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language=cls.config.get("language"),
                    prompt=cls.config.get("initial_prompt"),
                    temperature=cls.config.get("temperature", 0.0)
                )
            
            # API doesn't provide confidence scores
            metadata = {
                "model": "whisper-1",
                "language": cls.config.get("language", "auto"),
                "task": "transcribe",
                "api_used": True
            }
            
            return STTResult(
                text=transcript["text"].strip(),
                confidence=None,  # API doesn't provide confidence
                metadata=metadata
            )
            
        finally:
            # Clean up temporary file if created
            if cleanup_temp and os.path.exists(audio_file_path):
                try:
                    os.unlink(audio_file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {audio_file_path}: {e}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Any]:
        """Get information about available Whisper models."""
        local_models = ["tiny", "base", "small", "medium", "large"] if WHISPER_AVAILABLE else []
        api_available = OPENAI_AVAILABLE
        
        return {
            "local_models": local_models,
            "api_available": api_available,
            "whisper_installed": WHISPER_AVAILABLE,
            "openai_installed": OPENAI_AVAILABLE,
            "soundfile_installed": SOUNDFILE_AVAILABLE
        }
    
    @classmethod
    def set_language(cls, language: Optional[str]) -> None:
        """Set the transcription language."""
        cls.config["language"] = language
        logger.info(f"Language set to: {language or 'auto-detect'}")
    
    @classmethod
    def set_task(cls, task: str) -> None:
        """Set the task (transcribe or translate)."""
        if task not in ["transcribe", "translate"]:
            raise ValueError("Task must be 'transcribe' or 'translate'")
        cls.config["task"] = task
        logger.info(f"Task set to: {task}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing WhisperSTT implementation...")
    
    # Check availability
    models_info = WhisperSTT.get_available_models()
    print(f"Available models: {models_info}")
    
    if models_info["whisper_installed"]:
        try:
            # Test with local model
            print("\\nTesting local Whisper model...")
            WhisperSTT.load_model("tiny")  # Use tiny model for faster testing
            
            # Test with dummy numpy audio
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
            result = WhisperSTT.transcribe_numpy(dummy_audio, 16000)
            print(f"Dummy audio result: {result}")
            print(f"Model info: {WhisperSTT.get_model_info()}")
            
        except Exception as e:
            print(f"Local model test failed: {e}")
    else:
        print("Whisper not installed - install with: pip install openai-whisper")
    
    print("\\nWhisperSTT implementation ready!")