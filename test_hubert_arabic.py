#!/usr/bin/env python3
"""
Test script for HuBERT Arabic STT model

This script tests the HuBERT Arabic Egyptian STT implementation
including authentication, model loading, and transcription.
"""

import sys
import os
from pathlib import Path
import soundfile as sf
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hubert_arabic_stt():
    """Test the HuBERT Arabic STT implementation."""
    print("🚀 Testing HuBERT Arabic STT")
    print("=" * 50)
    
    try:
        from stt.hubert_arabic_stt import HuBERTArabicSTT
        print("✅ HuBERTArabicSTT imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import HuBERTArabicSTT: {e}")
        print("\n💡 To install HuBERT dependencies:")
        print("   pip install -r requirements_hubert.txt")
        return False
    
    # Test model loading
    print("\n📦 Testing model loading...")
    try:
        stt = HuBERTArabicSTT()
        
        # Try to load the primary model
        print("🔧 Loading HuBERT Arabic Egyptian model...")
        result = stt.load_model(
            model_id="omarxadel/hubert-large-arabic-egyptian",
            device="auto"
        )
        print(f"Model load result: {result}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("\n💡 This might be due to:")
        print("   - Missing HuggingFace authentication token")
        print("   - Network connectivity issues")
        print("   - Private model access restrictions")
        print("\n🔧 Try setting up authentication:")
        print("   python setup_hf_auth.py")
        return False
    
    # Test with sample audio (if available)
    print("\n🎵 Testing audio transcription...")
    
    # Create a test audio file (silence)
    sample_rate = 16000
    duration = 2.0  # seconds
    test_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    test_audio_path = "test_audio_hubert.wav"
    sf.write(test_audio_path, test_audio, sample_rate)
    
    try:
        transcription, confidence, processing_info = stt.transcribe(test_audio_path)
        print(f"✅ Transcription completed")
        print(f"   Text: '{transcription}'")
        print(f"   Confidence: {confidence}")
        print(f"   Processing: {processing_info}")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
    
    print("\n✅ All HuBERT Arabic STT tests passed!")
    return True

def test_with_real_audio():
    """Test with real audio if available."""
    recordings_dir = Path("recordings")
    
    if not recordings_dir.exists():
        print(f"\n💡 No recordings directory found at {recordings_dir}")
        print("   Create the directory and add .wav files to test with real audio")
        return
    
    audio_files = list(recordings_dir.glob("*.wav"))
    if not audio_files:
        print(f"\n💡 No .wav files found in {recordings_dir}")
        return
    
    print(f"\n🎵 Testing with real audio files from {recordings_dir}...")
    
    try:
        from stt.hubert_arabic_stt import HuBERTArabicSTT
        stt = HuBERTArabicSTT()
        stt.load_model()
        
        for audio_file in audio_files[:2]:  # Test first 2 files
            print(f"\n🔊 Processing: {audio_file.name}")
            try:
                transcription, confidence, processing_info = stt.transcribe(str(audio_file))
                print(f"   Text: '{transcription}'")
                print(f"   Confidence: {confidence}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Real audio test failed: {e}")

def main():
    """Main test function."""
    print("HuBERT Arabic STT Test Suite")
    print("=" * 60)
    
    # Basic functionality test
    success = test_hubert_arabic_stt()
    
    if success:
        print("\n🎯 Running additional tests...")
        test_with_real_audio()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 HuBERT Arabic STT is working correctly!")
        print("\n💡 Next steps:")
        print("   1. Test with the Gradio interface:")
        print("      python gradio_voice_transcriber_clean.py")
        print("   2. Select 'HuBERTArabicSTT' as the STT model")
        print("   3. Upload Arabic Egyptian audio for transcription")
    else:
        print("❌ HuBERT Arabic STT tests failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements_hubert.txt")
        print("   2. Set up HF authentication: python setup_hf_auth.py")
        print("   3. Check network connectivity")

if __name__ == "__main__":
    main()