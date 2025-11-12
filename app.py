#!/usr/bin/env python3
"""
Modular Gradio Voice Transcriber

A flexible web interface for voice transcription supporting multiple STT models.
Easily extensible to support any STT implementation that follows the BaseSTT interface.

Usage:
    python gradio_voice_transcriber_clean.py
"""

import gradio as gr
import numpy as np
import logging
import time
from typing import Tuple, Optional, Dict, Any, Type, List, Union
from pathlib import Path

# Import base STT class and available implementations
from stt.stt_base import BaseSTT, STTResult
from stt.whisper_stt import WhisperSTT

# Try to import Wav2Vec2 Arabic STT (optional)
try:
    from stt.wav2vec2_arabic_stt import Wav2Vec2ArabicSTT
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False

# Try to import HuBERT Arabic STT (optional)
try:
    from stt.hubert_arabic_stt import HuBERTArabicSTT
    HUBERT_AVAILABLE = True
except ImportError:
    HUBERT_AVAILABLE = False

# Try to import Vosk STT (optional)
try:
    from stt.vosk_stt import VoskSTT
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# Try to import Coqui STT (optional)
try:
    from stt.coqui_stt import CoquiSTT
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# Try to import Tawasul STT (optional)
try:
    from stt.tawasul_stt import TawasulSTT
    TAWASUL_AVAILABLE = True
except ImportError:
    TAWASUL_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# STT Model Registry - Add new models here
STT_MODELS: Dict[str, Type[BaseSTT]] = {
    "WhisperSTT": WhisperSTT,
}

# Add Wav2Vec2 Arabic if available
if WAV2VEC2_AVAILABLE:
    STT_MODELS["Wav2Vec2ArabicSTT"] = Wav2Vec2ArabicSTT

# Add HuBERT Arabic if available
if HUBERT_AVAILABLE:
    STT_MODELS["HuBERTArabicSTT"] = HuBERTArabicSTT

# Add Vosk if available
if VOSK_AVAILABLE:
    STT_MODELS["VoskSTT"] = VoskSTT

# Add Coqui STT if available
if COQUI_AVAILABLE:
    STT_MODELS["CoquiSTT"] = CoquiSTT

# Add Tawasul STT if available
if TAWASUL_AVAILABLE:
    STT_MODELS["TawasulSTT"] = TawasulSTT

# Global state
current_stt_model: Optional[Type[BaseSTT]] = None
current_model_config: Dict[str, Any] = {}


class AudioProcessor:
    """Handle audio preprocessing for better transcription quality."""
    
    @staticmethod
    def preprocess(audio_data: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio for better transcription quality.
        
        Args:
            audio_data: Raw audio data
            sample_rate: Original sample rate
            target_sr: Target sample rate (default: 16000 for Whisper)
            
        Returns:
            Preprocessed audio data
        """
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to float32 [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)
        
        # Clip to prevent overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Simple noise gate (remove very quiet sections)
        if len(audio_data) > 0:
            threshold = np.max(np.abs(audio_data)) * 0.01
            audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
        
        # Resample if needed
        if sample_rate != target_sr:
            audio_data = AudioProcessor._resample(audio_data, sample_rate, target_sr)
        
        return audio_data
    
    @staticmethod
    def _resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (prefer librosa if available)."""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling fallback
            if orig_sr > target_sr:
                step = orig_sr // target_sr
                return audio_data[::step]
            else:
                repeat_factor = target_sr // orig_sr
                return np.repeat(audio_data, repeat_factor)
    
    @staticmethod
    def _preprocess_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file for STT models that need torch.Tensor input.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor_as_numpy, sample_rate) that can be converted to torch.Tensor
        """
        try:
            import librosa
            import soundfile as sf
            
            # Try to load with librosa first (more robust)
            try:
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            except Exception:
                # Fallback to soundfile
                audio_data, sample_rate = sf.read(audio_path)
                if sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
            
            # Convert to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio to [-1, 1]
            if audio_data.max() > 1.0:
                audio_data = audio_data / audio_data.max()
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Apply noise gate for very quiet audio
            threshold = np.max(np.abs(audio_data)) * 0.01
            audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
            
            # Convert to float32 for compatibility
            audio_data = audio_data.astype(np.float32)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {str(e)}")
    
    @staticmethod
    def _preprocess_audio_torch(audio_path: str):
        """
        Preprocess audio file and return torch.Tensor for PyTorch-based STT models.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate) where audio_tensor is torch.Tensor
        """
        try:
            import torch
            
            # Get numpy array first
            audio_data, sample_rate = AudioProcessor._preprocess_audio(audio_path)
            
            # Convert to torch tensor
            audio_tensor = torch.FloatTensor(audio_data)
            
            return audio_tensor, sample_rate
            
        except ImportError:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        except Exception as e:
            raise RuntimeError(f"Torch audio preprocessing failed: {str(e)}")
    
    @staticmethod
    def analyze_quality(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio quality and provide feedback."""
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        duration = len(audio_data) / sample_rate
        max_amp = np.max(np.abs(audio_data))
        mean_amp = np.mean(np.abs(audio_data))
        
        # Check for clipping and silence
        clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
        silence_threshold = max_amp * 0.01
        silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
        
        return {
            "duration": duration,
            "max_amplitude": max_amp,
            "mean_amplitude": mean_amp,
            "clipping_ratio": clipping_ratio,
            "silence_ratio": silence_ratio,
            "sample_rate": sample_rate,
            "is_good_quality": (
                duration > 1.0 and
                0.1 < max_amp < 0.9 and
                clipping_ratio < 0.01 and
                silence_ratio < 0.5
            )
        }


class ModelManager:
    """Handle STT model registration and loading."""
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available STT model names."""
        return list(STT_MODELS.keys())
    
    @staticmethod
    def get_model_options(model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration options."""
        if model_name == "WhisperSTT":
            return {
                "model_sizes": ["tiny", "base", "small", "medium", "large"],
                "supports_api": True,
                "languages": [
                    ("Auto-detect", "auto"),
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Italian", "it"),
                    ("Portuguese", "pt"),
                    ("Russian", "ru"),
                    ("Japanese", "ja"),
                    ("Korean", "ko"),
                    ("Chinese", "zh"),
                    ("Dutch", "nl"),
                    ("Arabic", "ar"),
                    ("Hindi", "hi")
                ],
                "default_params": {
                    "temperature": 0.0,
                    "beam_size": 5,
                    "best_of": 5,
                    "patience": 2.0,
                    "condition_on_previous_text": True,
                }
            }
        
        elif model_name == "Wav2Vec2ArabicSTT":
            return {
                "model_sizes": [
                    ("Arabic Standard", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"),
                    ("Multilingual", "facebook/wav2vec2-large-xlsr-53"),
                    ("English Fallback", "facebook/wav2vec2-base-960h"),
                    ("Arabic Egyptian (Experimental)", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian")
                ],
                "supports_api": False,
                "supports_hf_token": True,
                "languages": [
                    ("Arabic Egyptian", "ar-EG"),
                    ("Arabic Standard", "ar"),
                    ("Auto-detect", "auto"),
                ],
                "device_options": ["auto", "cpu", "cuda"],
                "default_params": {
                    "device": "auto",
                    "chunk_length": 20,
                    "return_confidence": True,
                }
            }
        
        elif model_name == "VoskSTT":
            return {
                "model_sizes": [
                    ("English US Small (40MB)", "vosk-model-small-en-us-0.15"),
                    ("English US Large (1.8GB)", "vosk-model-en-us-0.22"),
                    ("Arabic (318MB)", "vosk-model-ar-mgb2-0.4"),
                    ("French (1.4GB)", "vosk-model-fr-0.22"),
                    ("German (1.2GB)", "vosk-model-de-0.21"),
                    ("Spanish (1.4GB)", "vosk-model-es-0.42"),
                    ("Russian Large (1.5GB)", "vosk-model-ru-0.42"),
                    ("Russian Small (45MB)", "vosk-model-small-ru-0.22"),
                    ("Chinese Small (42MB)", "vosk-model-small-cn-0.22"),
                ],
                "supports_api": False,
                "supports_auto_download": True,
                "languages": [
                    ("Auto (based on model)", "auto"),
                    ("English", "en"),
                    ("Arabic", "ar"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Spanish", "es"),
                    ("Russian", "ru"),
                    ("Chinese", "zh"),
                ],
                "default_params": {
                    "auto_download": True,
                    "return_confidence": True,
                    "return_words": True,
                }
            }
        
        elif model_name == "HuBERTArabicSTT":
            return {
                "model_sizes": [
                    ("Arabic Egyptian (HuBERT)", "omarxadel/hubert-large-arabic-egyptian"),
                    ("Arabic Egyptian (Wav2Vec2)", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian"),
                    ("Arabic Standard (Wav2Vec2)", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"),
                    ("Arabic MSA", "facebook/wav2vec2-large-xlsr-53")
                ],
                "supports_api": False,
                "supports_hf_token": True,
                "languages": [
                    ("Arabic Egyptian", "ar-EG"),
                    ("Arabic Standard", "ar"),
                    ("Auto-detect", "auto"),
                ],
                "device_options": ["auto", "cpu", "cuda"],
                "default_params": {
                    "device": "auto",
                    "chunk_length": 20,
                    "return_confidence": True,
                    "max_audio_length": 120
                }
            }
        
        elif model_name == "CoquiSTT":
            return {
                "model_sizes": [
                    ("English Large Vocab", "english-large"),
                    ("English Huge Vocab", "english-huge"),
                    ("German", "german"),
                    ("French", "french"),
                    ("Spanish", "spanish")
                ],
                "supports_api": False,
                "supports_auto_download": True,
                "languages": [
                    ("English", "en"),
                    ("German", "de"),
                    ("French", "fr"),
                    ("Spanish", "es"),
                    ("Auto (based on model)", "auto"),
                ],
                "default_params": {
                    "auto_download": True,
                    "beam_width": 512,
                    "lm_alpha": 0.931289039105002,
                    "lm_beta": 1.1834137581510284,
                    "return_confidence": True,
                    "return_timestamps": False,
                }
            }
        
        elif model_name == "TawasulSTT":
            return {
                "model_sizes": [
                    ("Tawasul STT V0 (Arabic)", "Kareem35/Tawasul-STT-V0"),
                    ("Arabic Standard (Wav2Vec2)", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"),
                    ("Arabic Egyptian (Wav2Vec2)", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian"),
                    ("Multilingual Fallback", "facebook/wav2vec2-large-xlsr-53")
                ],
                "supports_api": False,
                "supports_hf_token": True,
                "languages": [
                    ("Arabic Standard", "ar"),
                    ("Arabic Egyptian", "ar-EG"),
                    ("Arabic Saudi", "ar-SA"),
                    ("Arabic Jordanian", "ar-JO"),
                    ("Arabic Lebanese", "ar-LB"),
                    ("Arabic Syrian", "ar-SY"),
                    ("Arabic Iraqi", "ar-IQ"),
                    ("Auto-detect", "auto"),
                ],
                "device_options": ["auto", "cpu", "cuda"],
                "default_params": {
                    "device": "auto",
                    "chunk_length": 20,
                    "return_confidence": True,
                    "max_audio_length": 300
                }
            }
        
        # Default options for other models
        return {
            "model_sizes": ["default"],
            "supports_api": False,
            "languages": [("Auto-detect", "auto")],
            "default_params": {}
        }
    
    @staticmethod
    def load_model(model_name: str, **kwargs) -> str:
        """Load specified STT model with configuration."""
        global current_stt_model, current_model_config
        
        if model_name not in STT_MODELS:
            return f"❌ Unknown model: {model_name}. Available: {list(STT_MODELS.keys())}"
        
        try:
            model_class = STT_MODELS[model_name]
            
            # Handle TawasulSTT as static class (don't instantiate)
            if model_name == "TawasulSTT":
                model_instance = model_class  # Use class directly for static methods
            else:
                # Instantiate the model for instance-based classes
                model_instance = model_class()
            
            if model_name == "WhisperSTT":
                # Handle WhisperSTT specific loading
                model_size = kwargs.get("model_size", "base")
                use_api = kwargs.get("use_api", False)
                api_key = kwargs.get("api_key", "")
                
                if use_api and not api_key.strip():
                    return "❌ Error: API key required for API mode"
                
                # Load with optimized parameters
                load_params = {
                    "model_size": model_size,
                    "use_api": use_api,
                }
                
                if api_key:
                    load_params["api_key"] = api_key.strip()
                
                # Add quality optimization parameters for local models
                if not use_api:
                    load_params.update({
                        "temperature": 0.0,
                        "beam_size": 5,
                        "best_of": 5,
                        "patience": 2.0,
                        "condition_on_previous_text": True,
                    })
                
                model_instance.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_size": model_size,
                    "use_api": use_api
                }
                
                status = f"✅ {model_name} ({'API' if use_api else model_size}) loaded successfully"
            
            elif model_name == "Wav2Vec2ArabicSTT":
                # Handle Wav2Vec2 Arabic specific loading
                device = kwargs.get("device", "auto")
                chunk_length = kwargs.get("chunk_length", 20)
                hf_token = kwargs.get("hf_token", "")
                model_id = kwargs.get("model_size", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
                
                load_params = {
                    "device": device,
                    "chunk_length": chunk_length,
                    "model_id": model_id,
                }
                
                if hf_token:
                    load_params["hf_token"] = hf_token.strip()
                
                model_instance.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "device": device,
                    "chunk_length": chunk_length
                }
                
                # Extract model name for display
                model_display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                status = f"✅ {model_name} ({model_display_name}) loaded on {device}"
            
            elif model_name == "VoskSTT":
                # Handle VoskSTT specific loading
                model_name_param = kwargs.get("model_size", "vosk-model-small-en-us-0.15")
                auto_download = kwargs.get("auto_download", True)
                
                load_params = {
                    "model_name": model_name_param,
                    "auto_download": auto_download,
                }
                
                model_instance.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_name_param": model_name_param,
                    "auto_download": auto_download
                }
                
                status = f"✅ {model_name} ({model_name_param}) loaded successfully"
            
            elif model_name == "HuBERTArabicSTT":
                # Handle HuBERT Arabic specific loading
                device = kwargs.get("device", "auto")
                chunk_length = kwargs.get("chunk_length", 20)
                hf_token = kwargs.get("hf_token", "")
                model_id = kwargs.get("model_size", "omarxadel/hubert-large-arabic-egyptian")
                max_audio_length = kwargs.get("max_audio_length", 120)
                
                load_params = {
                    "device": device,
                    "chunk_length": chunk_length,
                    "model_id": model_id,
                    "max_audio_length": max_audio_length,
                }
                
                if hf_token:
                    load_params["hf_token"] = hf_token.strip()
                
                model_instance.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "device": device,
                    "chunk_length": chunk_length,
                    "max_audio_length": max_audio_length
                }
                
                # Extract model name for display
                model_display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                status = f"✅ {model_name} ({model_display_name}) loaded on {device}"
            
            elif model_name == "CoquiSTT":
                # Handle Coqui STT specific loading
                model_name_param = kwargs.get("model_size", "english-large")
                auto_download = kwargs.get("auto_download", True)
                beam_width = kwargs.get("beam_width", 512)
                lm_alpha = kwargs.get("lm_alpha", 0.931289039105002)
                lm_beta = kwargs.get("lm_beta", 1.1834137581510284)
                
                load_params = {
                    "model_name": model_name_param,
                    "auto_download": auto_download,
                    "beam_width": beam_width,
                    "lm_alpha": lm_alpha,
                    "lm_beta": lm_beta,
                }
                
                model_instance.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_name_param": model_name_param,
                    "auto_download": auto_download,
                    "beam_width": beam_width,
                    "lm_alpha": lm_alpha,
                    "lm_beta": lm_beta
                }
                
                status = f"✅ {model_name} ({model_name_param}) loaded successfully"
            
            elif model_name == "TawasulSTT":
                # Handle Tawasul STT specific loading (static class)
                device = kwargs.get("device", "auto")
                chunk_length = kwargs.get("chunk_length", 20)
                hf_token = kwargs.get("hf_token", "")
                model_id = kwargs.get("model_size", "Kareem35/Tawasul-STT-V0")
                max_audio_length = kwargs.get("max_audio_length", 300)
                
                load_params = {
                    "device": device,
                    "chunk_length": chunk_length,
                    "model_id": model_id,
                    "max_audio_length": max_audio_length,
                }
                
                if hf_token:
                    load_params["hf_token"] = hf_token.strip()
                
                # Call static method directly
                model_class.load_model(**load_params)
                
                current_model_config = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "device": device,
                    "chunk_length": chunk_length,
                    "max_audio_length": max_audio_length
                }
                
                # Extract model name for display
                model_display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                status = f"✅ {model_name} ({model_display_name}) loaded on {device}"
            
            else:
                # Generic model loading for future STT models
                model_instance.load_model(**kwargs)
                current_model_config = {"model_name": model_name, **kwargs}
                status = f"✅ {model_name} loaded successfully"
            
            current_stt_model = model_instance
            logger.info(status)
            return status
            
        except Exception as e:
            error_msg = f"❌ Error loading {model_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    @staticmethod
    def get_model_info() -> str:
        """Get information about available and loaded models."""
        info = f"**Available Models:** {', '.join(STT_MODELS.keys())}\n\n"
        
        if current_stt_model:
            model_info = current_stt_model.get_model_info()
            # Handle different key names for model name
            model_name = model_info.get('model_name') or model_info.get('name', 'Unknown')
            info += f"**Currently Loaded:** {model_name}\n"
            info += f"**Status:** {'✅ Ready' if model_info['is_loaded'] else '❌ Not loaded'}\n"
            info += f"**Config:** {current_model_config}"
        else:
            info += "**Currently Loaded:** None"
        
        return info


class TranscriptionEngine:
    """Handle audio transcription using the loaded STT model."""
    
    @staticmethod
    def transcribe(audio_input: Tuple[int, np.ndarray], 
                  language: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Transcribe audio input using the currently loaded STT model.
        
        Args:
            audio_input: Tuple of (sample_rate, audio_data) from Gradio
            language: Language code for transcription
            
        Returns:
            Tuple of (transcription, confidence_info, processing_info)
        """
        if audio_input is None:
            return "❌ No audio provided", "", ""
        
        if not current_stt_model or not current_stt_model.is_loaded:
            return "❌ No STT model loaded. Please load a model first.", "", ""
        
        try:
            sample_rate, audio_data = audio_input
            
            # Preprocess audio
            processed_audio = AudioProcessor.preprocess(audio_data, sample_rate)
            
            # Quality checks
            quality = AudioProcessor.analyze_quality(processed_audio, 16000)
            
            if quality["duration"] < 0.5:
                return "❌ Audio too short (minimum 0.5 seconds)", "", ""
            
            if quality["max_amplitude"] < 0.001:
                return "❌ Audio too quiet or silent", "", f"Max amplitude: {quality['max_amplitude']:.6f}"
            
            # Set language for models that support it
            if hasattr(current_stt_model, 'set_language') and language and language != "auto":
                current_stt_model.set_language(language)
            
            # Transcribe using different approaches for different models
            start_time = time.time()
            
            # Check if this is TawasulSTT (static class) which needs file path
            if current_model_config.get('model_name') == 'TawasulSTT':
                # TawasulSTT needs a file path, so save audio to temporary file
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, processed_audio, 16000)
                
                try:
                    # Call TawasulSTT.transcribe() with file path
                    transcription, confidence_info_raw, processing_info_raw = current_stt_model.transcribe(temp_path)
                    
                    # Create a result-like object for consistency
                    class TempResult:
                        def __init__(self, text, confidence=None, processing_time=None):
                            self.text = text
                            self.confidence = confidence
                            self.processing_time = processing_time
                    
                    # Extract confidence from confidence_info_raw if available
                    confidence_value = None
                    if confidence_info_raw and "Confidence:" in confidence_info_raw:
                        try:
                            conf_str = confidence_info_raw.split("Confidence:")[1].strip()
                            confidence_value = float(conf_str)
                        except:
                            confidence_value = None
                    
                    processing_time = time.time() - start_time
                    result = TempResult(transcription, confidence_value, processing_time)
                    
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            else:
                # For other STT models that use transcribe_audio
                result = current_stt_model.transcribe_audio(processed_audio, 16000)
            
            # Prepare output
            transcription = result.text.strip() if result.text else "No speech detected"
            
            # Filter out common false positives
            if transcription.lower() in ["you", "thank you.", "thanks for watching!", ""]:
                transcription = "🔇 No clear speech detected"
            
            # Confidence info
            confidence_info = ""
            if result.confidence is not None:
                confidence_info = f"Confidence: {result.confidence:.2%}"
                if result.confidence < 0.3:
                    confidence_info += " (Low - consider re-recording)"
            else:
                confidence_info = "Confidence: N/A"
            
            # Processing info
            processing_info = f"Processing: {result.processing_time or 0:.2f}s\n"
            processing_info += f"Model: {current_model_config.get('model_name', 'Unknown')}\n"
            processing_info += f"Audio: {quality['duration']:.2f}s, {quality['max_amplitude']:.3f} amplitude\n"
            processing_info += f"Quality: {'✅ Good' if quality['is_good_quality'] else '⚠️ Poor'}"
            
            return transcription, confidence_info, processing_info
            
        except Exception as e:
            error_msg = f"❌ Transcription error: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""


class GradioInterface:
    """Create and manage the Gradio web interface."""
    
    @staticmethod
    def create_interface():
        """Create the main Gradio interface."""
        with gr.Blocks(
            title="🎙️ Modular Voice Transcriber",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown(
                """
                # 🎙️ Modular Voice Transcriber
                
                A flexible interface supporting multiple STT models. 
                Easily extensible for new transcription engines.
                """
            )
            
            with gr.Row():
                # Model Configuration Panel
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 Model Configuration")
                    
                    # Model selection
                    model_selector = gr.Dropdown(
                        choices=ModelManager.get_available_models(),
                        value="WhisperSTT",
                        label="STT Model",
                        info="Choose your speech-to-text engine"
                    )
                    
                    # Dynamic model options (will update based on selected model)
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="base",
                        label="Model Size",
                        visible=True
                    )
                    
                    use_api = gr.Checkbox(
                        label="Use API",
                        info="Use cloud API instead of local model",
                        visible=True
                    )
                    
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter API key...",
                        visible=False
                    )
                    
                    # Device selection for models that support it
                    device_selector = gr.Dropdown(
                        choices=["auto", "cpu", "cuda"],
                        value="auto",
                        label="Device",
                        info="Processing device (auto recommended)",
                        visible=False
                    )
                    
                    # HuggingFace token for private models
                    hf_token = gr.Textbox(
                        label="HuggingFace Token",
                        type="password",
                        placeholder="hf_...",
                        info="Optional: For private or experimental models",
                        visible=False
                    )
                    
                    # Load button and status
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                    load_status = gr.Textbox(
                        label="Status",
                        value="No model loaded",
                        interactive=False
                    )
                    
                    # Model info
                    model_info = gr.Markdown(ModelManager.get_model_info())
                
                # Transcription Panel
                with gr.Column(scale=2):
                    gr.Markdown("### 🎤 Voice Transcription")
                    
                    # Language selection
                    language = gr.Dropdown(
                        choices=[("Auto-detect", "auto"), ("English", "en")],
                        value="auto",
                        label="Language"
                    )
                    
                    # Audio input
                    audio_input = gr.Audio(
                        label="Record or Upload Audio",
                        type="numpy",
                        format="wav"
                    )
                    
                    # Action buttons
                    with gr.Row():
                        transcribe_btn = gr.Button("🎯 Transcribe", variant="primary")
                        quality_btn = gr.Button("📊 Check Quality")
                        clear_btn = gr.Button("🗑️ Clear")
                    
                    # Outputs
                    transcription_output = gr.Textbox(
                        label="📝 Transcription",
                        lines=4,
                        placeholder="Transcribed text will appear here..."
                    )
                    
                    with gr.Row():
                        confidence_output = gr.Textbox(
                            label="🎯 Confidence",
                            interactive=False
                        )
                        processing_output = gr.Textbox(
                            label="⏱️ Processing Info",
                            interactive=False
                        )
                    
                    quality_output = gr.Markdown(
                        value="",
                        visible=False,
                        label="📊 Audio Quality Analysis"
                    )
            
            # Usage tips
            gr.Markdown(
                """
                ### 💡 Tips for Best Results
                - **Record clearly** in a quiet environment
                - **Speak at normal pace** - not too fast or slow
                - **Use good audio quality** - avoid background noise
                - **Try different models** - larger models are more accurate but slower
                - **Check quality analysis** to identify audio issues
                """
            )
            
            # Event handlers
            def update_model_options(model_name: str):
                """Update interface based on selected model."""
                options = ModelManager.get_model_options(model_name)
                
                # Determine visibility of components
                show_model_size = len(options["model_sizes"]) > 1
                show_api = options["supports_api"]
                show_device = "device_options" in options
                show_hf_token = options.get("supports_hf_token", False)
                
                # Extract model size options (handle both simple lists and tuples)
                if show_model_size and isinstance(options["model_sizes"][0], tuple):
                    # Model sizes are tuples of (display_name, value)
                    size_choices = options["model_sizes"]
                    size_value = size_choices[0][1]  # Use the value from first tuple
                else:
                    # Model sizes are simple strings
                    size_choices = options["model_sizes"]
                    size_value = size_choices[0]
                
                return (
                    gr.update(choices=size_choices, value=size_value, visible=show_model_size),
                    gr.update(visible=show_api),
                    gr.update(visible=False),  # Hide API key initially
                    gr.update(choices=options["languages"], value="auto"),
                    gr.update(
                        choices=options.get("device_options", ["auto"]),
                        value="auto",
                        visible=show_device
                    ),
                    gr.update(visible=show_hf_token)
                )
            
            def toggle_api_key(use_api: bool):
                """Show/hide API key field."""
                return gr.update(visible=use_api)
            
            def load_selected_model(model_name: str, model_size: str, use_api: bool, api_key: str, device: str, hf_token: str):
                """Load the selected model with configuration."""
                kwargs = {"model_size": model_size, "use_api": use_api}
                if api_key:
                    kwargs["api_key"] = api_key
                if device and device != "auto":
                    kwargs["device"] = device
                if hf_token:
                    kwargs["hf_token"] = hf_token
                return ModelManager.load_model(model_name, **kwargs)
            
            def analyze_audio_quality(audio_input):
                """Analyze and display audio quality."""
                if audio_input is None:
                    return "", gr.update(visible=False)
                
                sample_rate, audio_data = audio_input
                quality = AudioProcessor.analyze_quality(audio_data, sample_rate)
                
                report = f"""
                **📊 Audio Quality Analysis:**
                - Duration: {quality['duration']:.2f}s
                - Max amplitude: {quality['max_amplitude']:.3f}
                - Clipping: {quality['clipping_ratio']:.2%}
                - Silence ratio: {quality['silence_ratio']:.2%}
                - Overall quality: {'✅ Good' if quality['is_good_quality'] else '⚠️ Needs improvement'}
                
                **🔧 Recommendations:**
                {_get_quality_recommendations(quality)}
                """
                
                return report, gr.update(visible=True)
            
            # Connect events
            model_selector.change(
                fn=update_model_options,
                inputs=model_selector,
                outputs=[model_size, use_api, api_key, language, device_selector, hf_token]
            )
            
            use_api.change(
                fn=toggle_api_key,
                inputs=use_api,
                outputs=api_key
            )
            
            load_btn.click(
                fn=load_selected_model,
                inputs=[model_selector, model_size, use_api, api_key, device_selector, hf_token],
                outputs=load_status
            ).then(
                fn=lambda: ModelManager.get_model_info(),
                outputs=model_info
            )
            
            transcribe_btn.click(
                fn=TranscriptionEngine.transcribe,
                inputs=[audio_input, language],
                outputs=[transcription_output, confidence_output, processing_output]
            )
            
            quality_btn.click(
                fn=analyze_audio_quality,
                inputs=audio_input,
                outputs=[quality_output, quality_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "", "", gr.update(visible=False)),
                outputs=[transcription_output, confidence_output, processing_output, quality_output, quality_output]
            )
            
            # Auto-transcribe on audio change (optional)
            audio_input.change(
                fn=TranscriptionEngine.transcribe,
                inputs=[audio_input, language],
                outputs=[transcription_output, confidence_output, processing_output]
            )
        
        return demo


def _get_quality_recommendations(quality: Dict[str, Any]) -> str:
    """Generate quality recommendations based on analysis."""
    recommendations = []
    
    if quality["duration"] < 1.0:
        recommendations.append("• Try recording for longer (1+ seconds)")
    
    if quality["max_amplitude"] < 0.1:
        recommendations.append("• Increase volume or move closer to microphone")
    elif quality["max_amplitude"] > 0.9:
        recommendations.append("• Reduce volume to avoid clipping")
    
    if quality["clipping_ratio"] > 0.01:
        recommendations.append("• Audio is clipping - reduce input gain")
    
    if quality["silence_ratio"] > 0.5:
        recommendations.append("• Too much silence - record in quieter environment")
    
    if not recommendations:
        recommendations.append("• Audio quality looks good!")
    
    return "\n".join(recommendations)


def main():
    """Main application entry point."""
    # Check dependencies
    print("🔍 Checking dependencies...")
    
    try:
        import gradio
        print("✅ Gradio available")
    except ImportError:
        print("❌ Gradio not installed. Run: pip install gradio")
        return
    
    # Check available STT models
    print(f"🤖 Available STT models: {ModelManager.get_available_models()}")
    
    # Create and launch interface
    print("🚀 Launching Gradio interface...")
    demo = GradioInterface.create_interface()
    
    demo.launch(
        share=True,  # Set to True for public sharing
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()