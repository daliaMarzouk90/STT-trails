---
title: Modular Voice Transcriber
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: 3.10
pinned: false
---

# Modular Voice Transcriber

A comprehensive, modular Gradio-based web interface for speech-to-text transcription supporting multiple STT engines including OpenAI Whisper, Wav2Vec2, HuBERT Arabic, Tawasul, Vosk, and Coqui STT models.

## 🌟 Features

- **Comprehensive STT Support**: 7+ different speech-to-text engines
- **Multiple Models**: OpenAI Whisper, Wav2Vec2 Arabic, HuBERT, Tawasul, Vosk, Coqui STT
- **Arabic Language Focus**: Specialized models for Arabic dialect recognition
- **Web Interface**: User-friendly Gradio interface with image gallery
- **Real-time Processing**: Live audio recording and transcription
- **Quality Analysis**: Audio quality feedback and recommendations
- **Device Support**: CPU/GPU automatic detection and selection
- **Authentication**: Support for private HuggingFace models
- **Static Class Support**: Optimized memory usage for certain models
- **Visual Interface**: Interactive image gallery with thumbnail navigation

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Essential models (Whisper + Wav2Vec2)
python setup.py --profile essential --test

# Or specific models
python setup.py --profile whisper-only --test
python setup.py --profile wav2vec2-only --test
```

### Option 2: Manual Installation

```bash
# Base installation (Whisper + Wav2Vec2)
pip install -r requirements.txt

# Specific model installations
pip install -r requirements_whisper.txt     # OpenAI Whisper only
pip install -r requirements_wav2vec2.txt    # Wav2Vec2 only
pip install -r requirements_hubert.txt      # HuBERT Arabic only
pip install -r requirements_tawasul.txt     # Tawasul Arabic only
pip install -r requirements_vosk.txt        # Vosk offline only
pip install -r requirements_coqui.txt       # Coqui STT only

# Or install with specific extras
pip install -e .[essential]    # Whisper + Wav2Vec2
pip install -e .[all-stt]     # All models
```

## 🎯 Supported STT Models

| Model | Language | Size | Type | Quality | Features |
|-------|----------|------|------|---------|----------|
| **Whisper Tiny** | Multilingual | 39MB | Local/API | Fast | General purpose |
| **Whisper Base** | Multilingual | 142MB | Local/API | Good | General purpose |
| **Whisper Medium** | Multilingual | 1.5GB | Local/API | Better | General purpose |
| **Whisper Large** | Multilingual | 2.9GB | Local/API | Best | General purpose |
| **Wav2Vec2 Arabic** | Arabic | 1.2GB | Local | Excellent | Arabic dialects |
| **HuBERT Arabic** | Arabic Egyptian | 1.2GB | Local | Excellent | Egyptian dialect |
| **Tawasul V0** | Arabic | 800MB | Local | Very Good | Arabic speech, Static class |
| **Vosk** | Multilingual | 50MB-1.8GB | Local/Offline | Good | Offline capable |
| **Coqui STT** | Multilingual | 180MB-2GB | Local | Good | Open source |

## 🔧 Usage

### Start the Interface
```bash
python gradio_voice_transcriber_clean.py
```

### Using Different Models

1. **Select Model**: Choose from dropdown (WhisperSTT, Wav2Vec2ArabicSTT, HubertArabicSTT, TawasulSTT, VoskSTT, CoquiSTT)
2. **Configure**: Set model size, device, language, and authentication if needed
3. **Load**: Click "Load Model" (first time downloads model automatically)
4. **Transcribe**: Record audio or upload audio files
5. **Gallery**: Browse sample images using the interactive thumbnail gallery

### Authentication for Private Models

Some experimental models require HuggingFace authentication:

```bash
# Option 1: Use helper script
python setup_hf_auth.py

# Option 2: Manual login
pip install huggingface-hub
huggingface-cli login

# Option 3: Use token in interface
# Get token from: https://huggingface.co/settings/tokens
# Enter in "HuggingFace Token" field
```

## 🏗️ Adding New STT Models

The system is designed to be easily extensible:

1. **Create STT Class**: Inherit from `BaseSTT` in `stt/your_model.py`
2. **Register Model**: Add to `STT_MODELS` in `gradio_voice_transcriber_clean.py`
3. **Configure Options**: Update `get_model_options()` method
4. **Test**: Run your model through the interface

See `STT_INTEGRATION_GUIDE.md` for detailed instructions.

## 🎨 Interface Features

### Image Gallery
The enhanced interface (`gradio_voice_transcript_temp.py`) includes:
- **Interactive Gallery**: Browse sample images with thumbnail navigation
- **Horizontal Scrolling**: Smooth image browsing experience
- **Thumbnail Selection**: Click thumbnails to view full images
- **Gallery Controls**: Navigation and zoom functionality

### Static Class Models
Some models (like TawasulSTT) use static class implementation for:
- **Memory Efficiency**: Reduced memory footprint
- **Faster Loading**: Optimized model initialization
- **Shared Resources**: Better resource management

## 📁 Model Storage Locations

Different STT models store their files in various locations:

| Model Type | Storage Location | Description |
|------------|------------------|-------------|
| **Hugging Face Models** | `~/.cache/huggingface/` | Wav2Vec2, HuBERT, Tawasul models |
| **Whisper Models** | `~/.cache/whisper/` | OpenAI Whisper model files |
| **Vosk Models** | `~/.vosk/models/` | Offline Vosk language models |
| **Coqui Models** | Managed by model manager | Coqui STT model files |

## 📁 Project Structure

```
STT-trails/
├── gradio_voice_transcriber.py          # Main comprehensive interface  
├── gradio_voice_transcript_temp.py      # Enhanced interface with image gallery
├── stt/                                  # STT implementations
│   ├── stt_base.py                      # Base class for all STT models
│   ├── whisper_stt.py                   # OpenAI Whisper implementation
│   ├── wav2vec2_arabic_stt.py           # Wav2Vec2 Arabic model
│   ├── hubert_arabic_stt.py             # HuBERT Arabic dialect model
│   ├── tawasul_stt.py                   # Tawasul Arabic model (static class)
│   ├── vosk_stt.py                      # Vosk offline STT
│   ├── coqui_stt.py                     # Coqui open-source STT
│   └── example_custom_stt.py            # Template for new models
├── setup.py                             # Installation helper
├── setup_hf_auth.py                     # HuggingFace authentication helper
├── test_*.py                            # Model testing scripts
├── requirements*.txt                     # Dependencies for each model
├── recordings/                          # Audio recordings directory
├── INSTALL.md                           # Detailed installation guide
├── STT_INTEGRATION_GUIDE.md             # Developer integration guide
└── pyproject.toml                       # Project configuration
```

## 🧪 Testing

```bash
# Test specific models
python test_whisper_local.py         # Test Whisper models
python test_wav2vec2_arabic.py       # Test Wav2Vec2 Arabic
python test_hubert_arabic.py         # Test HuBERT Arabic
python test_tawasul.py               # Test Tawasul Arabic
python test_vosk.py                  # Test Vosk offline STT
python test_coqui.py                 # Test Coqui STT

# Test installation
python setup.py --profile essential --test

# Run the interface
python gradio_voice_transcriber.py           # Main interface
python gradio_voice_transcript_temp.py       # Interface with image gallery
```

## 🔍 Troubleshooting

### Model Loading Issues
- **HuggingFace Authentication**: Use `setup_hf_auth.py` or manual token
- **Memory Issues**: Use smaller models or CPU-only mode
- **Internet Required**: First model download needs internet connection

### Audio Issues
- **No Audio Detected**: Check microphone permissions and volume
- **Poor Quality**: Use audio quality analysis feature
- **Wrong Language**: Select appropriate model for your language

### Performance Tips
- **Use GPU**: Automatic if CUDA PyTorch is installed
- **Chunk Long Audio**: Handled automatically for 20+ second clips
- **Choose Right Model**: Balance size vs. accuracy for your use case

## 📄 License

This project is open source. See individual model licenses:
- OpenAI Whisper: MIT License
- Wav2Vec2: MIT License
- HuggingFace Transformers: Apache 2.0

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add your STT model following the integration guide
4. Submit a pull request

## 📚 Resources

- [OpenAI Whisper](https://openai.com/research/whisper)
- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuggingFace Models](https://huggingface.co/models)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Made with ❤️ for the speech recognition community**
