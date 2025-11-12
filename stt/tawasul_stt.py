#!/usr/bin/env python3
"""
Tawasul STT V0 Implementation

This module provides Arabic speech-to-text transcription using the Tawasul STT V0 model,
which is specifically designed for Arabic language recognition.

Tawasul STT V0 is built on Wav2Vec2 architecture and fine-tuned for Arabic speech.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import torch for type hints
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy torch class for type hints when torch is not available
    class torch:
        class Tensor:
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TawasulSTT:
    """
    Tawasul STT V0 static implementation for Arabic speech recognition.
    
    This class provides Arabic speech-to-text transcription using the Tawasul STT V0 model,
    which is specifically optimized for Arabic language variants.
    All methods are static for direct class-level access.
    """
    
    # Class variables for model state
    model = None
    processor = None
    tokenizer = None
    device = "cpu"
    model_id = "Kareem35/Tawasul-STT-V0"
    is_loaded = False
    hf_token = None
    chunk_length = 20  # seconds
    max_audio_length = 300  # 5 minutes max
    
    # Model fallback chain for better reliability
    fallback_models = [
        "Kareem35/Tawasul-STT-V0",
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "facebook/wav2vec2-large-xlsr-53",
        "facebook/wav2vec2-base-960h"
    ]
    
    @staticmethod
    def is_available() -> bool:
        """Check if Tawasul STT dependencies are available."""
        if not TORCH_AVAILABLE:
            logger.warning("Tawasul STT dependencies not available: torch not installed")
            return False
            
        try:
            import transformers
            import torchaudio
            import librosa
            import soundfile
            return True
        except ImportError as e:
            logger.warning(f"Tawasul STT dependencies not available: {e}")
            return False
    
    @staticmethod
    def load_model(
        model_id: Optional[str] = None,
        device: str = "auto",
        chunk_length: int = 20,
        hf_token: Optional[str] = None,
        max_audio_length: int = 300,
        **kwargs
    ) -> None:
        """
        Load the Tawasul STT V0 model.
        
        Args:
            model_id: Model identifier (defaults to Tawasul STT V0)
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            chunk_length: Audio chunk length in seconds for processing
            hf_token: Hugging Face authentication token
            max_audio_length: Maximum audio length in seconds
            **kwargs: Additional model parameters
        """
        try:
            import torch
            import transformers
            from transformers import (
                Wav2Vec2ForCTC, 
                Wav2Vec2Processor, 
                Wav2Vec2Tokenizer
            )
            import torchaudio
            import librosa
            
            # Set authentication token
            if hf_token:
                TawasulSTT.hf_token = hf_token
                # Set token for transformers
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=True)
                    logger.info("✅ Authenticated with Hugging Face")
                except Exception as e:
                    logger.warning(f"HF authentication warning: {e}")
            
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    TawasulSTT.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    TawasulSTT.device = "mps"
                else:
                    TawasulSTT.device = "cpu"
            else:
                TawasulSTT.device = device
            
            # Set model parameters
            TawasulSTT.model_id = model_id or "Kareem35/Tawasul-STT-V0"
            TawasulSTT.chunk_length = chunk_length
            TawasulSTT.max_audio_length = max_audio_length
            
            # Try loading the model with fallback chain
            model_loaded = False
            last_error = None
            
            models_to_try = [TawasulSTT.model_id] + [m for m in TawasulSTT.fallback_models if m != TawasulSTT.model_id]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"🔄 Loading Tawasul STT model: {model_name}")
                    
                    # Load model components
                    TawasulSTT.processor = Wav2Vec2Processor.from_pretrained(
                        model_name,
                        token=TawasulSTT.hf_token,
                        trust_remote_code=True
                    )
                    
                    TawasulSTT.model = Wav2Vec2ForCTC.from_pretrained(
                        model_name,
                        token=TawasulSTT.hf_token,
                        trust_remote_code=True
                    )
                    
                    # Try to load tokenizer if available
                    try:
                        TawasulSTT.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
                            model_name,
                            token=TawasulSTT.hf_token
                        )
                    except Exception:
                        logger.info("Using processor instead of separate tokenizer")
                        TawasulSTT.tokenizer = TawasulSTT.processor.tokenizer
                    
                    # Move model to device
                    TawasulSTT.model = TawasulSTT.model.to(TawasulSTT.device)
                    TawasulSTT.model.eval()
                    
                    # Test model with dummy input
                    test_input = torch.randn(1, 16000).to(TawasulSTT.device)
                    with torch.no_grad():
                        _ = TawasulSTT.model(test_input)
                    
                    TawasulSTT.model_id = model_name  # Update to actually loaded model
                    model_loaded = True
                    logger.info(f"✅ Successfully loaded Tawasul STT model: {model_name} on {TawasulSTT.device}")
                    break
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
                    continue
            
            if not model_loaded:
                raise RuntimeError(f"Failed to load any Tawasul STT model. Last error: {last_error}")
            
            TawasulSTT.is_loaded = True
            
            # Log model info
            total_params = sum(p.numel() for p in TawasulSTT.model.parameters())
            logger.info(f"📊 Model loaded: {total_params:,} parameters on {TawasulSTT.device}")
            
        except Exception as e:
            error_msg = f"Failed to load Tawasul STT model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for Tawasul STT model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        try:
            import librosa
            import torch
            import numpy as np
            
            # Load audio file with proper error handling
            try:
                # Load audio at 16kHz as required by Tawasul STT
                audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            except Exception as load_error:
                raise RuntimeError(f"Failed to load audio file {audio_path}: {load_error}")
            
            # Validate audio data
            if len(audio) == 0:
                raise RuntimeError("Audio file is empty or corrupted")
            
            # Convert to float32 for processing
            audio = audio.astype(np.float32)
            
            # Remove DC offset (center around zero)
            audio = audio - np.mean(audio)
            
            # Normalize audio with proper scaling
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                # Normalize to [-0.95, 0.95] to prevent clipping
                audio = audio / max_val * 0.95
            else:
                logger.warning("Audio appears to be silent")
            
            # Apply simple noise gate to reduce background noise
            noise_threshold = np.max(np.abs(audio)) * 0.01  # 1% of max amplitude
            audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
            
            # Check and limit audio duration
            audio_duration = len(audio) / sample_rate
            if audio_duration > TawasulSTT.max_audio_length:
                logger.warning(f"Audio duration ({audio_duration:.1f}s) exceeds maximum ({TawasulSTT.max_audio_length}s)")
                # Truncate to maximum length
                max_samples = int(TawasulSTT.max_audio_length * sample_rate)
                audio = audio[:max_samples]
                logger.info(f"Audio truncated to {TawasulSTT.max_audio_length}s")
            
            # Validate minimum duration
            min_duration = 0.1  # 100ms minimum
            if audio_duration < min_duration:
                logger.warning(f"Audio duration ({audio_duration:.3f}s) is very short")
            
            # Convert to PyTorch tensor
            audio_tensor = torch.FloatTensor(audio)
            
            # Log preprocessing info
            final_duration = len(audio_tensor) / sample_rate
            logger.debug(f"Audio preprocessed: {final_duration:.2f}s, max_amp: {torch.max(torch.abs(audio_tensor)):.3f}")
            
            return audio_tensor, sample_rate
            
        except Exception as e:
            error_msg = f"Audio preprocessing failed for {audio_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _chunk_audio(audio_tensor: torch.Tensor, sample_rate: int) -> List[torch.Tensor]:
        """
        Split audio into chunks for processing.
        
        Args:
            audio_tensor: Audio tensor
            sample_rate: Sample rate
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(TawasulSTT.chunk_length * sample_rate)
        chunks = []
        
        for i in range(0, len(audio_tensor), chunk_samples):
            chunk = audio_tensor[i:i + chunk_samples]
            if len(chunk) > sample_rate * 0.5:  # Only process chunks > 0.5 seconds
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _transcribe_chunk(audio_chunk: torch.Tensor) -> Tuple[str, float]:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio_chunk: Audio chunk tensor
            
        Returns:
            Tuple of (transcription, confidence_score)
        """
        try:
            import torch
            
            # Prepare input
            input_values = TawasulSTT.processor(
                audio_chunk, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values
            
            input_values = input_values.to(TawasulSTT.device)
            
            # Get model predictions
            with torch.no_grad():
                logits = TawasulSTT.model(input_values).logits
            
            # Get predicted tokens
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode transcription
            transcription = TawasulSTT.processor.decode(predicted_ids[0])
            
            # Calculate confidence (approximation)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            
            return transcription.strip(), confidence
            
        except Exception as e:
            logger.error(f"Chunk transcription error: {str(e)}")
            return "", 0.0
    
    @staticmethod
    def transcribe(audio_path: str, **kwargs) -> Tuple[str, str, str]:
        """
        Transcribe audio file using Tawasul STT V0.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional transcription parameters
            
        Returns:
            Tuple of (transcription, confidence_info, processing_info)
        """
        if not TawasulSTT.is_loaded:
            return "❌ Model not loaded. Please load the model first.", "", ""
        
        try:
            start_time = time.time()
            
            # Validate file
            if not os.path.exists(audio_path):
                return f"❌ Audio file not found: {audio_path}", "", ""
            
            logger.info(f"🎵 Transcribing audio with Tawasul STT: {audio_path}")
            
            # Preprocess audio
            audio_tensor, sample_rate = TawasulSTT._preprocess_audio(audio_path)
            audio_duration = len(audio_tensor) / sample_rate
            
            # Process audio in chunks
            chunks = TawasulSTT._chunk_audio(audio_tensor, sample_rate)
            
            if not chunks:
                return "❌ No valid audio chunks found", "", ""
            
            # Transcribe each chunk
            transcriptions = []
            confidences = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                transcription, confidence = TawasulSTT._transcribe_chunk(chunk)
                
                if transcription:  # Only add non-empty transcriptions
                    transcriptions.append(transcription)
                    confidences.append(confidence)
            
            # Combine results
            if not transcriptions:
                return "❌ No transcription generated", "", ""
            
            final_transcription = " ".join(transcriptions).strip()
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create info strings
            confidence_info = f"Confidence: {avg_confidence:.2f}"
            processing_info = (
                f"Duration: {audio_duration:.1f}s | "
                f"Chunks: {len(chunks)} | "
                f"Time: {processing_time:.1f}s | "
                f"Model: {TawasulSTT.model_id.split('/')[-1]}"
            )
            
            logger.info(f"✅ Transcription completed in {processing_time:.1f}s")
            
            return final_transcription, confidence_info, processing_info
            
        except Exception as e:
            error_msg = f"❌ Tawasul STT transcription failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of supported languages."""
        return [
            "ar",      # Arabic
            "ar-SA",   # Saudi Arabic
            "ar-EG",   # Egyptian Arabic
            "ar-JO",   # Jordanian Arabic
            "ar-LB",   # Lebanese Arabic
            "ar-SY",   # Syrian Arabic
            "ar-IQ",   # Iraqi Arabic
            "ar-MA",   # Moroccan Arabic
            "ar-DZ",   # Algerian Arabic
            "ar-TN",   # Tunisian Arabic
        ]
    
    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Tawasul STT V0",
            "model_id": TawasulSTT.model_id,
            "device": TawasulSTT.device,
            "is_loaded": TawasulSTT.is_loaded,
            "supported_languages": TawasulSTT.get_supported_languages(),
            "chunk_length": TawasulSTT.chunk_length,
            "max_audio_length": TawasulSTT.max_audio_length,
            "architecture": "Wav2Vec2",
            "specialization": "Arabic Speech Recognition"
        }