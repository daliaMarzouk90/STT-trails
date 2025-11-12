#!/usr/bin/env python3
"""
Base Speech-to-Text (STT) Static Class

This module provides an abstract base class for implementing different STT models using static methods.
All STT implementations should inherit from this class and implement the required static methods.

Usage:
    from stt_base import BaseSTT
    
    class MySTTModel(BaseSTT):
        model = None  # Class variable to hold the model
        
        @classmethod
        def load_model(cls):
            # Load your specific model
            cls.model = your_model_loader()
            
        @classmethod
        def transcribe_audio(cls, audio_data, sample_rate):
            # Implement transcription logic
            return STTResult("transcribed text")
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, ClassVar
import numpy as np
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTResult:
    """Container for STT transcription results with metadata."""
    
    def __init__(self, 
                 text: str, 
                 confidence: Optional[float] = None,
                 processing_time: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return self.text
    
    def __repr__(self) -> str:
        return f"STTResult(text='{self.text}', confidence={self.confidence}, time={self.processing_time}s)"


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text models using static methods.
    
    All STT implementations must inherit from this class and implement:
    - load_model(): Load and initialize the STT model (classmethod)
    - transcribe_audio(): Convert audio to text (classmethod)
    """
    
    # Class variables that subclasses should define
    model_name: ClassVar[str] = "BaseSTT"
    model: ClassVar[Any] = None
    is_loaded: ClassVar[bool] = False
    config: ClassVar[Dict[str, Any]] = {}
    
    @classmethod
    @abstractmethod
    def load_model(cls) -> None:
        """
        Load and initialize the STT model.
        
        This method must be implemented by subclasses to load their specific model.
        After successful loading, set cls.is_loaded = True
        """
        pass
    
    @classmethod
    @abstractmethod
    def transcribe_audio(cls, 
                        audio_data: Union[np.ndarray, str, Path], 
                        sample_rate: Optional[int] = None) -> STTResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio input - can be:
                       - numpy array of audio samples
                       - path to audio file (str or Path)
            sample_rate: Sample rate of audio data (required for numpy arrays)
            
        Returns:
            STTResult: Object containing transcribed text and metadata
            
        This method must be implemented by subclasses.
        """
        pass
    
    @classmethod
    def transcribe_file(cls, file_path: Union[str, Path]) -> STTResult:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            STTResult: Transcription result
        """
        if not cls.is_loaded:
            raise RuntimeError(f"{cls.model_name} model not loaded. Call load_model() first.")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Transcribing file: {file_path}")
        start_time = time.time()
        
        result = cls.transcribe_audio(file_path)
        
        if result.processing_time is None:
            result.processing_time = time.time() - start_time
        
        logger.info(f"Transcription completed in {result.processing_time:.2f}s")
        return result
    
    @classmethod
    def transcribe_numpy(cls, 
                        audio_array: np.ndarray, 
                        sample_rate: int) -> STTResult:
        """
        Transcribe a numpy array of audio samples to text.
        
        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            STTResult: Transcription result
        """
        if not cls.is_loaded:
            raise RuntimeError(f"{cls.model_name} model not loaded. Call load_model() first.")
        
        if not isinstance(audio_array, np.ndarray):
            raise TypeError("audio_array must be a numpy array")
        
        logger.info(f"Transcribing numpy array: shape={audio_array.shape}, sr={sample_rate}")
        start_time = time.time()
        
        result = cls.transcribe_audio(audio_array, sample_rate)
        
        if result.processing_time is None:
            result.processing_time = time.time() - start_time
        
        logger.info(f"Transcription completed in {result.processing_time:.2f}s")
        return result
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        return {
            "model_name": cls.model_name,
            "is_loaded": cls.is_loaded,
            "config": cls.config
        }
    
    @classmethod
    def ensure_loaded(cls) -> None:
        """Ensure the model is loaded, load it if not."""
        if not cls.is_loaded:
            cls.load_model()
    
    @classmethod
    def get_status(cls) -> str:
        """Get a string representation of the model status."""
        status = "loaded" if cls.is_loaded else "not loaded"
        return f"{cls.model_name} STT Model ({status})"


class DummySTT(BaseSTT):
    """
    Dummy STT implementation for testing the static class interface.
    Returns placeholder text instead of actual transcription.
    """
    
    model_name = "DummySTT"
    model = None
    is_loaded = False
    config = {}
    
    @classmethod
    def load_model(cls) -> None:
        """Load the dummy model (just a placeholder)."""
        logger.info("Loading dummy STT model...")
        time.sleep(0.5)  # Simulate loading time
        cls.model = "dummy_model_loaded"
        cls.is_loaded = True
        logger.info("Dummy STT model loaded successfully")
    
    @classmethod
    def transcribe_audio(cls, 
                        audio_data: Union[np.ndarray, str, Path], 
                        sample_rate: Optional[int] = None) -> STTResult:
        """
        Dummy transcription - returns placeholder text.
        """
        if isinstance(audio_data, np.ndarray):
            duration = len(audio_data) / (sample_rate or 16000)
            text = f"[Dummy transcription of {duration:.1f}s audio]"
        else:
            text = f"[Dummy transcription of file: {Path(audio_data).name}]"
        
        # Simulate processing time
        processing_time = 0.1 + np.random.random() * 0.2
        time.sleep(processing_time)
        
        return STTResult(
            text=text,
            confidence=0.95,
            processing_time=processing_time,
            metadata={"model": "dummy", "simulated": True}
        )


# Example usage and testing
if __name__ == "__main__":
    # Test the dummy implementation
    print("Testing BaseSTT with static DummySTT implementation...")
    
    # Load the model
    DummySTT.load_model()
    
    # Test with dummy numpy array
    dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
    result = DummySTT.transcribe_numpy(dummy_audio, 16000)
    print(f"Numpy result: {result}")
    print(f"Model info: {DummySTT.get_model_info()}")
    print(f"Status: {DummySTT.get_status()}")
    
    print("\\nStatic BaseSTT interface ready for real STT implementations!")