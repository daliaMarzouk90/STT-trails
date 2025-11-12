#!/usr/bin/env python3
"""
Test script for Wav2Vec2 Arabic STT

This script tests the Wav2Vec2 Arabic STT implementation without requiring 
the full Gradio interface.
"""

import sys
import numpy as np
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_wav2vec2_arabic():
    """Test the Wav2Vec2 Arabic STT implementation."""
    print("🔍 Testing Wav2Vec2 Arabic STT...")
    
    try:
        from stt.wav2vec2_arabic_stt import Wav2Vec2ArabicSTT
        print("✅ Successfully imported Wav2Vec2ArabicSTT")
    except ImportError as e:
        print(f"❌ Failed to import Wav2Vec2ArabicSTT: {e}")
        print("\n📦 Required dependencies:")
        print("pip install transformers torch torchaudio")
        print("Optional: pip install librosa")
        return False
    
    # Check model availability
    print("\n📊 Checking model availability...")
    models_info = Wav2Vec2ArabicSTT.get_available_models()
    
    for key, value in models_info.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    if not models_info.get("transformers_available", False):
        print("\n❌ Transformers not available. Cannot proceed with test.")
        return False
    
    # Test model loading (this will download the model if not cached)
    print(f"\n🔄 Loading model...")
    print("⚠️  Note: First run will download ~1.2GB model from Hugging Face")
    
    try:
        Wav2Vec2ArabicSTT.load_model(device="cpu")  # Use CPU for testing
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = Wav2Vec2ArabicSTT.get_model_info()
        print(f"📋 Model info:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test with dummy audio (this won't produce meaningful Arabic text)
    print(f"\n🎤 Testing transcription with dummy audio...")
    print("⚠️  Note: Random audio won't produce meaningful Arabic text")
    
    try:
        # Create 2 seconds of random audio
        dummy_audio = np.random.randn(32000).astype(np.float32) * 0.1
        
        result = Wav2Vec2ArabicSTT.transcribe_audio(dummy_audio, 16000)
        
        print(f"📝 Transcription result:")
        print(f"   Text: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2%}" if result.confidence else "   Confidence: N/A")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Metadata: {result.metadata}")
        
        print("✅ Transcription test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False

def test_integration():
    """Test integration with the modular transcriber."""
    print(f"\n🔗 Testing integration with modular transcriber...")
    
    try:
        from gradio_voice_transcriber_clean import ModelManager, STT_MODELS
        
        available_models = ModelManager.get_available_models()
        print(f"📋 Available models: {available_models}")
        
        if "Wav2Vec2ArabicSTT" in available_models:
            print("✅ Wav2Vec2ArabicSTT is registered in the modular transcriber")
            
            # Test model options
            options = ModelManager.get_model_options("Wav2Vec2ArabicSTT")
            print(f"📊 Model options: {options}")
            
            return True
        else:
            print("❌ Wav2Vec2ArabicSTT not found in available models")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import modular transcriber components: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Wav2Vec2 Arabic STT Test Suite")
    print("=" * 50)
    
    # Test individual STT implementation
    stt_test = test_wav2vec2_arabic()
    
    # Test integration
    integration_test = test_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   STT Implementation: {'✅ PASS' if stt_test else '❌ FAIL'}")
    print(f"   Integration: {'✅ PASS' if integration_test else '❌ FAIL'}")
    
    if stt_test and integration_test:
        print("\n🎉 All tests passed! The Wav2Vec2 Arabic STT is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run: python gradio_voice_transcriber_clean.py")
        print("   2. Select 'Wav2Vec2ArabicSTT' from the dropdown")
        print("   3. Choose your device (CPU/CUDA)")
        print("   4. Load the model and test with Arabic audio!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        
        if not stt_test:
            print("\n📦 To fix STT implementation issues:")
            print("   pip install transformers torch torchaudio")
            print("   pip install librosa  # optional, for better audio processing")

if __name__ == "__main__":
    main()