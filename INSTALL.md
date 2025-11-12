# Installation Guide

This guide explains how to install and set up the Modular Voice Transcriber with different STT models.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Essential models (Whisper + Wav2Vec2)
python setup.py --profile essential --test

# Or for specific models only
python setup.py --profile whisper-only --test
python setup.py --profile wav2vec2-only --test
```

### Option 2: Manual Installation

#### Base Installation
```bash
# Core requirements
pip install gradio>=4.0.0 numpy>=1.21.0 soundfile>=0.12.1
```

#### Choose Your STT Models

**OpenAI Whisper (Local + API)**
```bash
pip install -r requirements_whisper.txt
# Or: pip install -e .[whisper,whisper-api]
```

**Wav2Vec2 Arabic**
```bash
pip install -r requirements_wav2vec2.txt
# Or: pip install -e .[wav2vec2]
```

**All Models**
```bash
pip install -r requirements.txt
# Or: pip install -e .[all-stt]
```

## 📦 Installation Profiles

| Profile | Models Included | Use Case |
|---------|----------------|----------|
| `minimal` | None | Interface only (for development) |
| `essential` | Whisper + Wav2Vec2 | Best balance of features |
| `whisper-only` | OpenAI Whisper | English + Multilingual |
| `wav2vec2-only` | Wav2Vec2 Arabic | Arabic Egyptian dialect |
| `all` | All supported models | Complete functionality |

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.9+
- 8GB RAM
- 5GB free disk space
- GPU with CUDA support (for faster transcription)

## 📋 Model Download Sizes

| Model | First Download | Disk Space |
|-------|---------------|------------|
| Whisper Tiny | 39MB | 39MB |
| Whisper Base | 142MB | 142MB |
| Whisper Medium | 1.5GB | 1.5GB |
| Wav2Vec2 Arabic | 1.2GB | 1.2GB |

## 🧪 Testing Your Installation

### Test Individual Models
```bash
# Test Wav2Vec2 Arabic
python test_wav2vec2_arabic.py

# Test Whisper (coming soon)
python test_whisper_local.py
```

### Test Full Interface
```bash
python gradio_voice_transcriber_clean.py
```

## 🔍 Troubleshooting

### Common Issues

**Import Error: transformers**
```bash
pip install transformers torch torchaudio
```

**Import Error: whisper**
```bash
pip install openai-whisper
```

**CUDA Issues**
- Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org)
- Or use CPU-only: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

**Model Download Issues**
- Check internet connection
- Hugging Face models download automatically on first use
- Downloads go to `~/.cache/huggingface/` and `~/.cache/whisper/`

### Performance Tips

**For Better Speed:**
- Use GPU if available
- Choose smaller models for real-time use
- Use larger models for better accuracy

**For Better Quality:**
- Record in quiet environment
- Use good microphone
- Speak clearly and at normal pace
- Choose appropriate language/dialect model

## 🔄 Updating

```bash
# Update to latest versions
pip install --upgrade -r requirements.txt

# Update specific models
pip install --upgrade transformers openai-whisper
```

## 🎯 Next Steps

1. **Run the interface:** `python gradio_voice_transcriber_clean.py`
2. **Choose your model** from the dropdown
3. **Load the model** (first time will download)
4. **Test with audio** recording or upload
5. **Check quality analysis** for audio tips

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Whisper by OpenAI](https://openai.com/research/whisper)
- [Wav2Vec2 Models](https://huggingface.co/models?search=wav2vec2)
- [Transformers Library](https://huggingface.co/docs/transformers/)