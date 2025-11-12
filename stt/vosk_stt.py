#!/usr/bin/env python3
"""
Vosk STT Implementation

Vosk speech-to-text implementation using the static BaseSTT interface.
Supports multiple languages with offline models and real-time recognition.

Usage:
    from stt.vosk_stt import VoskSTT
    
    # Load model
    VoskSTT.load_model(model_name="vosk-model-en-us-0.22")
    
    # Transcribe audio
    result = VoskSTT.transcribe_audio(audio_array, 16000)
    print(result.text)
"""

from typing import Union, Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import time
import json
import logging
import os
import urllib.request
import zipfile
import tempfile

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .stt_base import BaseSTT, STTResult

logger = logging.getLogger(__name__)


class VoskSTT(BaseSTT):
    """
    Vosk STT implementation supporting multiple languages and offline recognition.
    
    Features:
    - Multiple language support
    - Offline processing (no internet required after model download)
    - Real-time recognition capability
    - Small to large model options
    - Word-level timestamps and confidence scores
    - Lightweight and fast
    """
    
    model_name = "VoskSTT"
    model = None
    recognizer = None
    is_loaded = False
    config = {
        "model_name": "vosk-model-en-us-0.22",  # Default English model
        "model_path": None,  # Auto-determined
        "sample_rate": 16000,
        "language": "en",
        "download_url_base": "https://alphacephei.com/vosk/models/",
        "models_dir": str(Path.home() / ".vosk" / "models"),
        "return_confidence": True,
        "return_words": True,
        "chunk_size": 4096,
    }
    
    # Available Vosk models with their properties
    AVAILABLE_MODELS = {
        # English models
        "vosk-model-en-us-0.22": {
            "language": "en-US",
            "size": "1.8GB",
            "description": "English US Large",
            "url": "vosk-model-en-us-0.22.zip"
        },
        "vosk-model-small-en-us-0.15": {
            "language": "en-US", 
            "size": "40MB",
            "description": "English US Small",
            "url": "vosk-model-small-en-us-0.15.zip"
        },
        
        # Arabic models
        "vosk-model-ar-mgb2-0.4": {
            "language": "ar",
            "size": "318MB", 
            "description": "Arabic",
            "url": "vosk-model-ar-mgb2-0.4.zip"
        },
        
        # Multilingual and other languages
        "vosk-model-small-cn-0.22": {
            "language": "zh-CN",
            "size": "42MB",
            "description": "Chinese Small",
            "url": "vosk-model-small-cn-0.22.zip"
        },
        "vosk-model-fr-0.22": {
            "language": "fr-FR",
            "size": "1.4GB",
            "description": "French",
            "url": "vosk-model-fr-0.22.zip"
        },
        "vosk-model-de-0.21": {
            "language": "de-DE", 
            "size": "1.2GB",
            "description": "German",
            "url": "vosk-model-de-0.21.zip"
        },
        "vosk-model-es-0.42": {
            "language": "es-ES",
            "size": "1.4GB", 
            "description": "Spanish",
            "url": "vosk-model-es-0.42.zip"
        },
        "vosk-model-ru-0.42": {
            "language": "ru-RU",
            "size": "1.5GB",
            "description": "Russian", 
            "url": "vosk-model-ru-0.42.zip"
        },
        "vosk-model-small-ru-0.22": {
            "language": "ru-RU",
            "size": "45MB",
            "description": "Russian Small",
            "url": "vosk-model-small-ru-0.22.zip"
        }
    }
    
    @classmethod
    def load_model(cls,
                   model_name: str = None,
                   model_path: str = None,
                   auto_download: bool = True,
                   **kwargs) -> None:
        """
        Load the Vosk model.
        
        Args:
            model_name: Name of the Vosk model (e.g., "vosk-model-en-us-0.22")
            model_path: Direct path to model directory (overrides model_name)
            auto_download: Automatically download model if not found
            **kwargs: Additional configuration parameters
        """
        if not VOSK_AVAILABLE:
            raise ImportError(
                "Vosk library required. Install with: pip install vosk"
            )
        
        # Update configuration
        cls.config.update({
            "model_name": model_name or cls.config["model_name"],
            "model_path": model_path,
            "auto_download": auto_download,
            **kwargs
        })
        
        # Determine model path
        if model_path:
            final_model_path = Path(model_path)
        else:
            final_model_path = cls._get_model_path(cls.config["model_name"])
        
        # Check if model exists, download if needed
        if not final_model_path.exists():
            if auto_download:
                logger.info(f"Model not found at {final_model_path}")
                cls._download_model(cls.config["model_name"])
            else:
                raise FileNotFoundError(f"Model not found: {final_model_path}")
        
        logger.info(f"Loading Vosk model from: {final_model_path}")
        start_time = time.time()
        
        try:
            # Load the Vosk model
            cls.model = vosk.Model(str(final_model_path))
            
            # Create recognizer
            cls.recognizer = vosk.KaldiRecognizer(cls.model, cls.config["sample_rate"])
            
            # Configure recognizer options (with compatibility checks)
            try:
                if hasattr(cls.recognizer, 'SetMaxAlternatives'):
                    cls.recognizer.SetMaxAlternatives(cls.config.get("max_alternatives", 3))
                    logger.info("✅ Max alternatives enabled")
            except (AttributeError, Exception) as e:
                logger.warning(f"⚠️  Max alternatives not supported: {e}")
            
            try:
                if hasattr(cls.recognizer, 'SetReturnWordTimes'):
                    cls.recognizer.SetReturnWordTimes(cls.config.get("return_words", True))
                    logger.info("✅ Word timing enabled")
                else:
                    logger.info("ℹ️  Word timing not available in this Vosk version")
            except (AttributeError, Exception) as e:
                logger.warning(f"⚠️  Word timing not supported: {e}")
            
            try:
                if hasattr(cls.recognizer, 'SetWords'):
                    cls.recognizer.SetWords(cls.config.get("return_words", True))
                    logger.info("✅ Word-level output enabled")
            except (AttributeError, Exception) as e:
                logger.info(f"ℹ️  Word-level output using basic mode: {e}")
            
            # Test recognizer with a small sample
            test_result = cls.recognizer.AcceptWaveform(b'\x00' * 1600)  # 0.1s of silence
            logger.info("✅ Recognizer test successful")
            
            cls.is_loaded = True
            load_time = time.time() - start_time
            
            model_info = cls.AVAILABLE_MODELS.get(cls.config["model_name"], {})
            language = model_info.get("language", "unknown")
            
            logger.info(f"✅ Vosk model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model: {cls.config['model_name']}")
            logger.info(f"Language: {language}")
            logger.info(f"Sample rate: {cls.config['sample_rate']}Hz")
            
        except Exception as e:
            cls.is_loaded = False
            error_msg = f"Failed to load Vosk model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @classmethod
    def _get_model_path(cls, model_name: str) -> Path:
        """Get the local path where a model should be stored."""
        models_dir = Path(cls.config["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / model_name
    
    @classmethod
    def _download_model(cls, model_name: str) -> None:
        """Download a Vosk model if it's not already available."""
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        model_info = cls.AVAILABLE_MODELS[model_name]
        download_url = cls.config["download_url_base"] + model_info["url"]
        model_path = cls._get_model_path(model_name)
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return
        
        logger.info(f"Downloading Vosk model: {model_name}")
        logger.info(f"Size: {model_info['size']} - This may take a while...")
        logger.info(f"URL: {download_url}")
        
        try:
            # Create temporary file for download
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 100 == 0:  # Show progress every 100 blocks
                        print(f"\rDownloading... {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(download_url, tmp_path, show_progress)
            print()  # New line after progress
            
            logger.info(f"Download complete. Extracting to: {model_path}")
            
            # Extract the zip file
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Extract to temporary directory first
                extract_dir = model_path.parent / f"{model_name}_temp"
                extract_dir.mkdir(exist_ok=True)
                zip_ref.extractall(extract_dir)
                
                # Find the actual model directory (should contain conf/ and graph/ subdirs)
                extracted_items = list(extract_dir.iterdir())
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    # Move the inner directory to the final location
                    extracted_items[0].rename(model_path)
                    extract_dir.rmdir()
                else:
                    # Multiple items or files - rename the temp directory
                    extract_dir.rename(model_path)
            
            # Cleanup
            os.unlink(tmp_path)
            
            logger.info(f"✅ Model downloaded and extracted successfully: {model_path}")
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)
            
            error_msg = f"Failed to download model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @classmethod
    def transcribe_audio(cls,
                        audio_data: Union[np.ndarray, str, Path],
                        sample_rate: Optional[int] = None) -> STTResult:
        """
        Transcribe audio using Vosk.
        
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
            
            # Transcribe using Vosk
            result_text, confidence, words = cls._transcribe_with_vosk(processed_audio)
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "model": cls.config["model_name"],
                "language": cls.AVAILABLE_MODELS.get(cls.config["model_name"], {}).get("language", "unknown"),
                "duration": duration,
                "sample_rate": actual_sr,
                "words": words if cls.config.get("return_words", True) else None,
                "vosk_version": vosk.__version__ if hasattr(vosk, '__version__') else "unknown"
            }
            
            return STTResult(
                text=result_text.strip(),
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
            
            if SOUNDFILE_AVAILABLE:
                audio_array, sr = sf.read(str(audio_path))
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)  # Convert to mono
            else:
                raise ImportError("soundfile required for file input. Install with: pip install soundfile")
        else:
            # Handle numpy array
            audio_array = audio_data.astype(np.float32)
            sr = sample_rate or cls.config["sample_rate"]
            
            # Convert to mono if stereo
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
        
        # Resample to target sample rate if needed
        target_sr = cls.config["sample_rate"]
        if sr != target_sr:
            # Simple resampling
            if sr > target_sr:
                step = sr // target_sr
                audio_array = audio_array[::step]
            else:
                repeat = target_sr // sr
                audio_array = np.repeat(audio_array, repeat)
            sr = target_sr
        
        # Normalize and convert to 16-bit PCM format expected by Vosk
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        return audio_int16, sr
    
    @classmethod
    def _transcribe_with_vosk(cls, audio_int16: np.ndarray) -> tuple:
        """Transcribe audio using Vosk recognizer."""
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        # Reset recognizer for new transcription
        cls.recognizer = vosk.KaldiRecognizer(cls.model, cls.config["sample_rate"])
        
        # Configure recognizer with compatibility checks
        try:
            if hasattr(cls.recognizer, 'SetReturnWordTimes'):
                cls.recognizer.SetReturnWordTimes(cls.config.get("return_words", True))
        except (AttributeError, Exception):
            pass  # Use basic recognition without word timing
        
        # Process audio in chunks
        chunk_size = cls.config.get("chunk_size", 4096)
        partial_results = []
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if cls.recognizer.AcceptWaveform(chunk):
                result = json.loads(cls.recognizer.Result())
                if result.get("text"):
                    partial_results.append(result)
        
        # Get final result
        final_result = json.loads(cls.recognizer.FinalResult())
        if final_result.get("text"):
            partial_results.append(final_result)
        
        # Combine all results
        if not partial_results:
            return "", 0.0, []
        
        # Extract text and confidence
        full_text = " ".join([r.get("text", "") for r in partial_results]).strip()
        
        # Calculate average confidence from words
        all_words = []
        total_confidence = 0.0
        word_count = 0
        
        for result in partial_results:
            if "result" in result:
                words = result["result"]
                all_words.extend(words)
                for word in words:
                    if "conf" in word:
                        total_confidence += word["conf"]
                        word_count += 1
        
        average_confidence = total_confidence / word_count if word_count > 0 else 0.0
        
        return full_text, average_confidence, all_words
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Any]:
        """Get information about available Vosk models."""
        return {
            "vosk_available": VOSK_AVAILABLE,
            "soundfile_available": SOUNDFILE_AVAILABLE,
            "models": cls.AVAILABLE_MODELS,
            "models_dir": cls.config["models_dir"],
            "downloaded_models": cls._get_downloaded_models()
        }
    
    @classmethod
    def _get_downloaded_models(cls) -> List[str]:
        """Get list of already downloaded models."""
        models_dir = Path(cls.config["models_dir"])
        if not models_dir.exists():
            return []
        
        downloaded = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name in cls.AVAILABLE_MODELS:
                # Check if it looks like a valid Vosk model
                if (model_dir / "conf").exists() or (model_dir / "graph").exists():
                    downloaded.append(model_dir.name)
        
        return downloaded
    
    @classmethod
    def set_language(cls, language: Optional[str]) -> None:
        """Set language preference (informational - model determines actual language)."""
        cls.config["language"] = language or "auto"
        logger.info(f"Language preference set to: {cls.config['language']}")
        logger.info("Note: Vosk model determines actual recognition language")
    
    @classmethod
    def list_models(cls) -> None:
        """Print available models in a formatted way."""
        print("\n🎤 Available Vosk Models:")
        print("=" * 60)
        
        downloaded = cls._get_downloaded_models()
        
        for model_name, info in cls.AVAILABLE_MODELS.items():
            status = "✅ Downloaded" if model_name in downloaded else "📥 Available"
            print(f"{status} {model_name}")
            print(f"   Language: {info['language']}")
            print(f"   Size: {info['size']}")
            print(f"   Description: {info['description']}")
            print()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Vosk STT implementation...")
    
    # Check availability
    models_info = VoskSTT.get_available_models()
    print(f"Vosk available: {models_info['vosk_available']}")
    print(f"Downloaded models: {models_info['downloaded_models']}")
    
    if models_info["vosk_available"]:
        try:
            # List available models
            VoskSTT.list_models()
            
            # Try to load a small English model for testing
            print("\\nTesting with small English model...")
            VoskSTT.load_model(model_name="vosk-model-small-en-us-0.15")
            
            # Test with dummy audio
            print("Testing transcription...")
            test_audio = np.random.randn(16000).astype(np.float32) * 0.1
            
            result = VoskSTT.transcribe_audio(test_audio, 16000)
            print(f"Result: {result}")
            print(f"Metadata: {result.metadata}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Note: This is expected with random audio")
    
    else:
        print("Vosk not installed - install with: pip install vosk")
        print("Also recommended: pip install soundfile")
    
    print("\\nVosk STT implementation ready!")