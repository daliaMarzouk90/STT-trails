# Modular STT Integration Guide

This guide explains how to integrate new Speech-to-Text models into the modular Gradio voice transcriber.

## 🏗️ Architecture Overview

The system is built with a modular architecture that makes it easy to add new STT engines:

```
gradio_voice_transcriber_clean.py
├── ModelManager        # Handles model registration and loading
├── AudioProcessor      # Preprocesses audio for better quality
├── TranscriptionEngine # Manages transcription workflow
└── GradioInterface     # Creates the web UI
```

## 🔧 Adding a New STT Model

### Step 1: Create Your STT Class

Create a new file in the `stt/` directory (e.g., `your_stt.py`) that inherits from `BaseSTT`:

```python
from stt.stt_base import BaseSTT, STTResult
import numpy as np

class YourSTT(BaseSTT):
    model_name = "YourSTT"
    model = None
    is_loaded = False
    config = {}
    
    @classmethod
    def load_model(cls, **kwargs):
        # Initialize your STT service
        cls.model = your_stt_client()
        cls.is_loaded = True
    
    @classmethod
    def transcribe_audio(cls, audio_data, sample_rate=None):
        # Implement transcription logic
        result = cls.model.transcribe(audio_data)
        return STTResult(text=result.text, confidence=result.confidence)
```

### Step 2: Register Your Model

Add your model to the registry in `gradio_voice_transcriber_clean.py`:

```python
# Import your model
from stt.your_stt import YourSTT

# Add to registry
STT_MODELS = {
    "WhisperSTT": WhisperSTT,
    "YourSTT": YourSTT,  # Add this line
}
```

### Step 3: Configure Model Options

Update the `ModelManager.get_model_options()` method:

```python
@staticmethod
def get_model_options(model_name: str) -> Dict[str, Any]:
    if model_name == "YourSTT":
        return {
            "model_sizes": ["small", "large"],
            "supports_api": True,
            "languages": [("English", "en"), ("Spanish", "es")],
            "default_params": {"temperature": 0.0}
        }
    # ... existing code
```

### Step 4: Handle Model Loading

Update the loading logic in `ModelManager.load_model()`:

```python
if model_name == "YourSTT":
    api_key = kwargs.get("api_key", "")
    model_size = kwargs.get("model_size", "small")
    
    YourSTT.load_model(api_key=api_key, model_size=model_size)
    status = f"✅ {model_name} loaded successfully"
```

## 📝 Real Examples

### Azure Speech Service

```python
import azure.cognitiveservices.speech as speechsdk

class AzureSTT(BaseSTT):
    model_name = "AzureSTT"
    
    @classmethod
    def load_model(cls, subscription_key, region):
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region
        )
        cls.model = speech_config
        cls.is_loaded = True
    
    @classmethod
    def transcribe_audio(cls, audio_data, sample_rate=None):
        # Convert audio and send to Azure
        # Return STTResult with transcription
        pass
```

### Google Cloud Speech

```python
from google.cloud import speech

class GoogleSTT(BaseSTT):
    model_name = "GoogleSTT"
    
    @classmethod
    def load_model(cls, credentials_path):
        cls.model = speech.SpeechClient()
        cls.is_loaded = True
    
    @classmethod
    def transcribe_audio(cls, audio_data, sample_rate=None):
        # Process with Google Cloud Speech
        pass
```

### AssemblyAI

```python
import assemblyai as aai

class AssemblyAISTT(BaseSTT):
    model_name = "AssemblyAISTT"
    
    @classmethod
    def load_model(cls, api_key):
        aai.settings.api_key = api_key
        cls.model = aai.Transcriber()
        cls.is_loaded = True
    
    @classmethod
    def transcribe_audio(cls, audio_data, sample_rate=None):
        # Save audio temporarily and transcribe
        pass
```

## 🎯 Best Practices

### 1. Error Handling
```python
@classmethod
def load_model(cls, **kwargs):
    try:
        # Model loading logic
        cls.is_loaded = True
    except Exception as e:
        cls.is_loaded = False
        raise RuntimeError(f"Failed to load {cls.model_name}: {e}")
```

### 2. Configuration Management
```python
class YourSTT(BaseSTT):
    config = {
        "default_language": "en",
        "timeout": 30,
        "retry_count": 3
    }
    
    @classmethod
    def set_language(cls, language):
        cls.config["default_language"] = language
```

### 3. Audio Format Handling
```python
@classmethod
def transcribe_audio(cls, audio_data, sample_rate=None):
    # Handle numpy arrays
    if isinstance(audio_data, np.ndarray):
        # Convert to required format
        audio_bytes = audio_to_bytes(audio_data, sample_rate)
    else:
        # Handle file paths
        with open(audio_data, 'rb') as f:
            audio_bytes = f.read()
    
    # Transcribe and return result
```

### 4. Metadata and Confidence
```python
return STTResult(
    text=transcription,
    confidence=confidence_score,
    processing_time=processing_time,
    metadata={
        "model": cls.model_name,
        "language_detected": detected_language,
        "audio_duration": duration,
        "service_info": additional_info
    }
)
```

## 🚀 Testing Your Integration

1. **Unit Test Your STT Class**:
```python
def test_your_stt():
    YourSTT.load_model(api_key="test")
    dummy_audio = np.random.randn(16000).astype(np.float32)
    result = YourSTT.transcribe_audio(dummy_audio, 16000)
    assert result.text is not None
```

2. **Test in Gradio Interface**:
   - Run `python gradio_voice_transcriber_clean.py`
   - Select your model from the dropdown
   - Load it and test with audio

## 🛠️ Advanced Features

### Custom UI Components

You can add model-specific UI components by extending the interface:

```python
# Add custom fields for your model
if model_name == "YourSTT":
    custom_setting = gr.Slider(
        minimum=0, maximum=1, value=0.5,
        label="Custom Setting"
    )
```

### Background Processing

For long-running transcriptions:

```python
@classmethod
def transcribe_audio_async(cls, audio_data, callback):
    # Start background transcription
    # Call callback when done
    pass
```

## 📊 Current Available Models

- **WhisperSTT**: OpenAI Whisper (local + API)
- **ExampleCustomSTT**: Template for new integrations

## 🎯 Next Steps

1. Choose your STT service
2. Follow the integration pattern
3. Test thoroughly
4. Contribute back to the project!

The modular design makes it easy to support any STT service while maintaining a consistent user experience.