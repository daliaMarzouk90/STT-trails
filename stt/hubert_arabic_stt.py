#!/usr/bin/env python3
"""
HuBERT Arabic Egyptian STT Implementation

Hugging Face HuBERT speech-to-text implementation for Arabic Egyptian dialect
using the omarxadel/hubert-large-arabic-egyptian model.

Usage:
    from stt.hubert_arabic_stt import HuBERTArabicSTT
    
    # Load model
    HuBERTArabicSTT.load_model()
    
    # Transcribe audio
    result = HuBERTArabicSTT.transcribe_audio(audio_array, 16000)
    print(result.text)
"""

from typing import Union, Optional, Dict, Any
import numpy as np
from pathlib import Path
import time
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import torch
    import torchaudio
    from transformers import (
        HubertForCTC, 
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
        AutoProcessor,
        AutoModelForCTC
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .stt_base import BaseSTT, STTResult

logger = logging.getLogger(__name__)


class HuBERTArabicSTT(BaseSTT):
    """
    HuBERT Arabic Egyptian STT implementation using Hugging Face transformers.
    
    Supports:
    - Arabic Egyptian dialect transcription
    - Local model execution (no API required)
    - Automatic audio preprocessing
    - Confidence estimation
    - Chunked processing for long audio
    """
    
    model_name = "HuBERTArabicSTT"
    model = None
    processor = None
    tokenizer = None
    is_loaded = False
    config = {
        "model_id": "omarxadel/hubert-large-arabic-egyptian",
        "fallback_models": [
            "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian",
            "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
            "facebook/wav2vec2-large-xlsr-53",
        ],
        "device": "auto",  # auto, cpu, cuda
        "chunk_length": 15,  # seconds, for long audio processing
        "sample_rate": 16000,
        "return_confidence": True,
        "language": "ar-EG",  # Arabic Egyptian
        "hf_token": None,  # Hugging Face token for private models
        "use_auth_token": True  # Try to use cached token
    }
    
    @classmethod
    def load_model(cls, 
                   model_id: str = None,
                   device: str = "auto",
                   hf_token: str = None,
                   **kwargs) -> None:
        """
        Load the HuBERT Arabic model.
        
        Args:
            model_id: Hugging Face model ID (default: omarxadel/hubert-large-arabic-egyptian)
            device: Device to use (auto, cpu, cuda)
            hf_token: Hugging Face token for private models (optional)
            **kwargs: Additional configuration parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library required. Install with: "
                "pip install transformers torch torchaudio"
            )
        
        # Update configuration
        cls.config.update({
            "model_id": model_id or cls.config["model_id"],
            "device": device,
            "hf_token": hf_token,
            **kwargs
        })
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
        
        cls.config["device"] = device
        
        # Try to load the model, with fallbacks
        models_to_try = [cls.config["model_id"]] + cls.config["fallback_models"]
        
        for model_id_to_try in models_to_try:
            logger.info(f"Attempting to load HuBERT model: {model_id_to_try}")
            
            try:
                success = cls._load_model_with_id(model_id_to_try, device, hf_token)
                if success:
                    cls.config["model_id"] = model_id_to_try  # Update to successful model
                    return
            except Exception as e:
                logger.warning(f"Failed to load {model_id_to_try}: {e}")
                continue
        
        # If all models failed
        raise RuntimeError(f"Failed to load any HuBERT model. Tried: {models_to_try}")
    
    @classmethod
    def _load_model_with_id(cls, model_id: str, device: str, hf_token: str = None) -> bool:
        """
        Load a specific model ID with authentication handling.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Loading HuBERT model: {model_id}")
        logger.info(f"Using device: {device}")
        
        start_time = time.time()
        
        # Prepare authentication
        auth_kwargs = {}
        if hf_token:
            auth_kwargs["token"] = hf_token
        elif cls.config.get("use_auth_token", True):
            auth_kwargs["use_auth_token"] = True
        
        try:
            # Try to load as HuBERT model first
            if "hubert" in model_id.lower():
                logger.info("Loading as HuBERT model...")
                cls.processor = AutoProcessor.from_pretrained(model_id, **auth_kwargs)
                cls.model = AutoModelForCTC.from_pretrained(model_id, **auth_kwargs)
            else:
                # Fallback to Wav2Vec2 for other models
                logger.info("Loading as Wav2Vec2 model...")
                cls.processor = Wav2Vec2Processor.from_pretrained(model_id, **auth_kwargs)
                cls.model = AutoModelForCTC.from_pretrained(model_id, **auth_kwargs)
            
            # Move model to device
            cls.model = cls.model.to(device)
            cls.model.eval()  # Set to evaluation mode
            
            # Load tokenizer for confidence calculation
            try:
                cls.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id, **auth_kwargs)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
                cls.tokenizer = None
            
            cls.is_loaded = True
            load_time = time.time() - start_time
            
            logger.info(f"✅ HuBERT model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model vocab size: {cls.model.config.vocab_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    @classmethod
    def transcribe_audio(cls, 
                        audio_data: Union[np.ndarray, str, Path], 
                        sample_rate: Optional[int] = None) -> STTResult:
        """
        Transcribe audio using HuBERT Arabic model.
        
        Args:
            audio_data: Audio input (numpy array or file path)
            sample_rate: Sample rate for numpy arrays
            
        Returns:
            STTResult: Transcription with confidence and metadata
        """
        if not cls.is_loaded:
            raise RuntimeError(f"{cls.model_name} not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Process input audio
            processed_audio, actual_sr = cls._process_audio_input(audio_data, sample_rate)
            
            # Check audio length
            duration = len(processed_audio) / actual_sr
            if duration < 0.1:
                return STTResult(
                    text="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"error": "Audio too short", "duration": duration}
                )
            
            # Process with model
            if duration > cls.config.get("chunk_length", 15):
                # Handle long audio by chunking
                text, confidence = cls._transcribe_long_audio(processed_audio, actual_sr)
            else:
                # Process short audio directly
                text, confidence = cls._transcribe_chunk(processed_audio, actual_sr)
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "model": cls.config["model_id"],
                "model_type": "HuBERT" if "hubert" in cls.config["model_id"].lower() else "Wav2Vec2",
                "device": cls.config["device"],
                "language": "ar-EG",
                "duration": duration,
                "sample_rate": actual_sr,
                "chunks_processed": 1 if duration <= cls.config.get("chunk_length", 15) else int(duration / cls.config["chunk_length"]) + 1
            }
            
            return STTResult(
                text=text.strip(),
                confidence=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            return STTResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": error_msg}
            )
    
    @classmethod
    def _process_audio_input(cls, audio_data: Union[np.ndarray, str, Path], sample_rate: Optional[int]) -> tuple:
        """Process and validate audio input."""
        if isinstance(audio_data, (str, Path)):
            # Load audio file
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if LIBROSA_AVAILABLE:
                audio_array, sr = librosa.load(str(audio_path), sr=cls.config["sample_rate"])
            else:
                # Fallback to torchaudio
                audio_tensor, sr = torchaudio.load(str(audio_path))
                audio_array = audio_tensor.numpy().flatten()
                
                # Resample if needed
                if sr != cls.config["sample_rate"]:
                    resampler = torchaudio.transforms.Resample(sr, cls.config["sample_rate"])
                    audio_tensor = resampler(audio_tensor)
                    audio_array = audio_tensor.numpy().flatten()
                    sr = cls.config["sample_rate"]
        
        else:
            # Handle numpy array
            audio_array = audio_data.astype(np.float32)
            sr = sample_rate or cls.config["sample_rate"]
            
            # Resample if needed
            if sr != cls.config["sample_rate"]:
                if LIBROSA_AVAILABLE:
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=sr, 
                        target_sr=cls.config["sample_rate"]
                    )
                else:
                    # Simple resampling fallback
                    if sr > cls.config["sample_rate"]:
                        step = sr // cls.config["sample_rate"]
                        audio_array = audio_array[::step]
                    else:
                        repeat = cls.config["sample_rate"] // sr
                        audio_array = np.repeat(audio_array, repeat)
                
                sr = cls.config["sample_rate"]
        
        # Normalize audio
        if len(audio_array) > 0:
            # Convert to mono if stereo
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
        
        return audio_array, sr
    
    @classmethod
    def _transcribe_chunk(cls, audio_array: np.ndarray, sample_rate: int) -> tuple:
        """Transcribe a single audio chunk."""
        # Preprocess audio
        input_values = cls.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        input_values = {k: v.to(cls.config["device"]) for k, v in input_values.items()}
        
        # Inference
        with torch.no_grad():
            logits = cls.model(**input_values).logits
        
        # Get predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode transcription
        transcription = cls.processor.batch_decode(predicted_ids)[0]
        
        # Calculate confidence (average of max probabilities)
        confidence = cls._calculate_confidence(logits)
        
        return transcription, confidence
    
    @classmethod
    def _transcribe_long_audio(cls, audio_array: np.ndarray, sample_rate: int) -> tuple:
        """Transcribe long audio by chunking."""
        chunk_length = cls.config.get("chunk_length", 15)
        chunk_samples = int(chunk_length * sample_rate)
        overlap_samples = int(1.0 * sample_rate)  # 1 second overlap
        
        transcriptions = []
        confidences = []
        
        for start in range(0, len(audio_array), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_array))
            chunk = audio_array[start:end]
            
            if len(chunk) < 0.5 * sample_rate:  # Skip very short chunks
                continue
            
            try:
                chunk_text, chunk_confidence = cls._transcribe_chunk(chunk, sample_rate)
                if chunk_text.strip():
                    transcriptions.append(chunk_text.strip())
                    confidences.append(chunk_confidence)
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk: {e}")
                continue
        
        # Combine results
        full_text = " ".join(transcriptions)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    @classmethod
    def _calculate_confidence(cls, logits: torch.Tensor) -> float:
        """Calculate confidence score from model logits."""
        try:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get maximum probability for each time step
            max_probs = torch.max(probabilities, dim=-1)[0]
            
            # Average over time steps (excluding padding if any)
            confidence = torch.mean(max_probs).item()
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0.5  # Default confidence
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Any]:
        """Get information about available HuBERT models."""
        models_info = {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "librosa_available": LIBROSA_AVAILABLE,
            "torch_available": True if TRANSFORMERS_AVAILABLE else False,
        }
        
        if TRANSFORMERS_AVAILABLE:
            models_info.update({
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                "hubert_models": [
                    {
                        "id": "omarxadel/hubert-large-arabic-egyptian",
                        "name": "HuBERT Arabic Egyptian (Large)",
                        "language": "Arabic Egyptian Dialect",
                        "size": "1.3GB",
                        "type": "HuBERT"
                    }
                ],
                "fallback_models": [
                    {
                        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian",
                        "name": "Wav2Vec2 Arabic Egyptian",
                        "language": "Arabic Egyptian",
                        "size": "1.2GB",
                        "type": "Wav2Vec2"
                    },
                    {
                        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
                        "name": "Wav2Vec2 Arabic Standard",
                        "language": "Arabic Standard",
                        "size": "1.2GB",
                        "type": "Wav2Vec2"
                    },
                    {
                        "id": "facebook/wav2vec2-large-xlsr-53",
                        "name": "Wav2Vec2 Multilingual",
                        "language": "Multilingual",
                        "size": "1.2GB",
                        "type": "Wav2Vec2"
                    }
                ]
            })
        
        return models_info
    
    @classmethod
    def set_language(cls, language: Optional[str]) -> None:
        """Set language (for compatibility - this model is Arabic-specific)."""
        if language and not language.startswith("ar"):
            logger.warning(f"This model is optimized for Arabic. Language '{language}' may not work well.")
        
        cls.config["language"] = language or "ar-EG"
        logger.info(f"Language set to: {cls.config['language']}")
    
    @classmethod
    def set_device(cls, device: str) -> None:
        """Change device for model inference."""
        if cls.model is not None:
            cls.model = cls.model.to(device)
            cls.config["device"] = device
            logger.info(f"Model moved to device: {device}")
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        if cls.is_loaded:
            base_info.update({
                "model_id": cls.config["model_id"],
                "model_type": "HuBERT" if "hubert" in cls.config["model_id"].lower() else "Wav2Vec2",
                "device": cls.config["device"],
                "language": cls.config["language"],
                "sample_rate": cls.config["sample_rate"],
                "vocab_size": cls.model.config.vocab_size if cls.model else None,
                "chunk_length": cls.config["chunk_length"],
            })
        
        return base_info


# Example usage and testing
if __name__ == "__main__":
    print("Testing HuBERT Arabic STT implementation...")
    
    # Check availability
    models_info = HuBERTArabicSTT.get_available_models()
    print(f"Available models info: {models_info}")
    
    if models_info["transformers_available"]:
        try:
            print("Loading HuBERT Arabic model...")
            HuBERTArabicSTT.load_model(device="cpu")  # Use CPU for testing
            
            print("Creating test audio...")
            # Generate test audio (2 seconds of random noise)
            test_audio = np.random.randn(32000).astype(np.float32) * 0.1
            
            print("Testing transcription...")
            result = HuBERTArabicSTT.transcribe_audio(test_audio, 16000)
            print(f"Result: {result}")
            print(f"Metadata: {result.metadata}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Note: This is expected with random audio - the model expects Arabic speech")
    
    else:
        print("Transformers not installed - install with:")
        print("pip install transformers torch torchaudio")
        print("Optional: pip install librosa (for better audio processing)")
    
    print("\nHuBERT Arabic STT implementation ready!")