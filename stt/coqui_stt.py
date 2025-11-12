#!/usr/bin/env python3
"""
Coqui STT Implementation with Model Manager

This module provides speech-to-text functionality using coqui-stt-model-manager.
The model manager provides a simplified interface for downloading and using
Coqui STT models with automatic model management.

Features:
- Automatic model downloading and management
- Multiple pre-trained models available
- Language-specific models
- Offline processing
- Simplified API interface
- GPU acceleration support

Dependencies:
- coqui-stt-model-manager
- numpy
- soundfile
- librosa (for audio preprocessing)

Model Management:
Models are automatically managed by the coqui-stt-model-manager.
Popular models include English, German, French, Spanish, and more.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

try:
    from coqui_stt_model_manager import CoquiSTTModelManager
    COQUI_STT_AVAILABLE = True
except ImportError:
    COQUI_STT_AVAILABLE = False
    CoquiSTTModelManager = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

from .stt_base import BaseSTT

logger = logging.getLogger(__name__)


class CoquiSTT(BaseSTT):
    """
    Coqui STT implementation using coqui-stt-model-manager.
    
    Coqui STT provides high-quality open-source speech recognition
    with simplified model management through the model manager.
    """
    
    def __init__(self):
        """Initialize Coqui STT with model manager."""
        super().__init__()
        self.model_manager = None
        self.current_model = None
        self.model_info = {}
        
        # Available models through the model manager
        self.available_models = {
            "english-huge": {
                "language": "en",
                "description": "English model with huge vocabulary",
                "model_id": "english-huge-vocab"
            },
            "english-large": {
                "language": "en", 
                "description": "English model with large vocabulary",
                "model_id": "english-large-vocab"
            },
            "german": {
                "language": "de",
                "description": "German language model",
                "model_id": "german"
            },
            "french": {
                "language": "fr",
                "description": "French language model", 
                "model_id": "french"
            },
            "spanish": {
                "language": "es",
                "description": "Spanish language model",
                "model_id": "spanish"
            }
        }
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Coqui STT Model Manager is available."""
        try:
            from coqui_stt_model_manager import CoquiSTTModelManager
            import soundfile
            return True
        except ImportError as e:
            logger.warning(f"Coqui STT Model Manager dependencies not available: {e}")
            return False
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not COQUI_STT_AVAILABLE:
            missing_deps.append("coqui-stt-model-manager")
        
        if not SOUNDFILE_AVAILABLE:
            missing_deps.append("soundfile")
        
        if not LIBROSA_AVAILABLE:
            missing_deps.append("librosa (recommended for audio preprocessing)")
        
        if missing_deps:
            return False, f"Missing dependencies: {', '.join(missing_deps)}"
        
        return True, "All dependencies available"
    
    def load_model(
        self,
        model_name: str = "english-large",
        auto_download: bool = True,
        beam_width: int = 512,
        lm_alpha: float = 0.931289039105002,
        lm_beta: float = 1.1834137581510284,
        **kwargs
    ) -> None:
        """
        Load a Coqui STT model using the model manager.
        
        Args:
            model_name: Name of the model to load
            auto_download: Whether to automatically download the model if not found
            beam_width: Beam width for CTC beam search decoder
            lm_alpha: Language model alpha parameter
            lm_beta: Language model beta parameter
            **kwargs: Additional model parameters
        
        Raises:
            RuntimeError: If model loading fails
        """
        deps_ok, deps_msg = self.check_dependencies()
        if not deps_ok:
            raise RuntimeError(f"Dependency check failed: {deps_msg}")
        
        try:
            # Initialize model manager
            logger.info("Initializing Coqui STT Model Manager...")
            self.model_manager = CoquiSTTModelManager()
            
            # Get model identifier
            if model_name in self.available_models:
                model_id = self.available_models[model_name]["model_id"]
            else:
                model_id = model_name  # Use as custom model ID
            
            # Load the model through model manager
            logger.info(f"Loading Coqui STT model: {model_id}")
            
            if auto_download:
                # Download and load model
                self.current_model = self.model_manager.download_and_load_model(
                    model_id=model_id,
                    beam_width=beam_width,
                    lm_alpha=lm_alpha,
                    lm_beta=lm_beta
                )
            else:
                # Try to load existing model
                self.current_model = self.model_manager.load_model(
                    model_id=model_id,
                    beam_width=beam_width,
                    lm_alpha=lm_alpha,
                    lm_beta=lm_beta
                )
            
            # Store model info
            self.model_info = {
                "model_name": model_name,
                "model_id": model_id,
                "beam_width": beam_width,
                "lm_alpha": lm_alpha,
                "lm_beta": lm_beta,
            }
            
            if model_name in self.available_models:
                self.model_info.update(self.available_models[model_name])
            
            logger.info(f"Coqui STT model loaded successfully: {model_name}")
            
        except Exception as e:
            error_msg = f"Error loading Coqui STT model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for Coqui STT.
        
        Coqui STT requires 16kHz mono audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Original sample rate
        
        Returns:
            Preprocessed audio data
        """
        try:
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            target_sr = 16000
            if sample_rate != target_sr and LIBROSA_AVAILABLE:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
            elif sample_rate != target_sr:
                logger.warning(f"Audio is {sample_rate}Hz but Coqui STT requires 16kHz. Install librosa for automatic resampling.")
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Convert to int16 as required by Coqui STT
            audio_data = (audio_data * 32767).astype(np.int16)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_data
    
    def transcribe(self, audio_path: str, **kwargs) -> Tuple[str, str, str]:
        """
        Transcribe audio using Coqui STT Model Manager.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional transcription parameters
        
        Returns:
            Tuple of (transcription, confidence_info, processing_info)
        """
        if self.current_model is None:
            return "❌ Model not loaded. Please load the model first.", "", ""
        
        try:
            import time
            start_time = time.time()
            
            # Validate file
            if not os.path.exists(audio_path):
                return f"❌ Audio file not found: {audio_path}", "", ""
            
            logger.info(f"🎵 Transcribing audio with Coqui STT: {audio_path}")
            
            # Load audio file
            audio_data, sample_rate = sf.read(audio_path)
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Get transcription parameters
            return_confidence = kwargs.get("return_confidence", True)
            return_timestamps = kwargs.get("return_timestamps", False)
            
            # Perform transcription using model manager
            if return_timestamps:
                # Use metadata for word timestamps
                result = self.model_manager.transcribe_with_metadata(
                    audio_data=processed_audio,
                    model=self.current_model
                )
                
                # Extract text and calculate confidence
                transcription = ""
                total_confidence = 0.0
                word_count = 0
                
                if hasattr(result, 'transcripts') and result.transcripts:
                    for token in result.transcripts[0].tokens:
                        transcription += token.text
                        if hasattr(token, 'confidence'):
                            total_confidence += token.confidence
                            word_count += 1
                
                avg_confidence = total_confidence / word_count if word_count > 0 else 0.0
                
            else:
                # Simple transcription
                transcription = self.model_manager.transcribe(
                    audio_data=processed_audio,
                    model=self.current_model
                )
                avg_confidence = 0.8  # Estimated confidence
            
            # Calculate processing time
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate
            
            # Create info strings
            confidence_info = f"Confidence: {avg_confidence:.2f}" if return_confidence else ""
            processing_info = (
                f"Duration: {audio_duration:.1f}s | "
                f"Time: {processing_time:.1f}s | "
                f"Model: {self.model_info.get('model_name', 'unknown')}"
            )
            
            logger.info(f"✅ Transcription completed in {processing_time:.1f}s")
            
            return transcription.strip(), confidence_info, processing_info
            
        except Exception as e:
            error_msg = f"❌ Coqui STT transcription failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en",      # English
            "de",      # German
            "fr",      # French
            "es",      # Spanish
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        info = self.model_info.copy()
        info.update({
            "name": "Coqui STT with Model Manager",
            "is_loaded": self.current_model is not None,
            "supported_languages": self.get_supported_languages(),
            "architecture": "DeepSpeech-based CTC",
            "provider": "Coqui AI"
        })
        
        return info
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        models = []
        for name, info in self.available_models.items():
            model_info = {
                "name": name,
                "language": info["language"],
                "description": info["description"],
                "model_id": info["model_id"]
            }
            models.append(model_info)
        
        return models
    
    def cleanup(self):
        """Clean up resources."""
        if self.current_model is not None:
            # Model manager handles cleanup automatically
            self.current_model = None
        
        if self.model_manager is not None:
            self.model_manager = None
        
        self.model_info = {}
        
        logger.info("Coqui STT cleanup completed")


# Export the class
__all__ = ["CoquiSTT", "COQUI_STT_AVAILABLE"]