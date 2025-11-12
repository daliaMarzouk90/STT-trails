#!/usr/bin/env python3
"""
Test script for Coqui STT

This script tests the Coqui STT implementation with a sample audio file.
Coqui STT provides open-source speech recognition with multiple language support.

Usage:
    python test_coqui.py [audio_file]

If no audio file is provided, it will use the default recording if available.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from stt.coqui_stt import CoquiSTT, COQUI_STT_AVAILABLE

def test_coqui_stt(audio_file: str = None):
    """Test Coqui STT functionality."""
    print("🚀 Testing Coqui STT")
    print("=" * 50)
    
    # Check if Coqui STT is available
    if not COQUI_STT_AVAILABLE:
        print("❌ Coqui STT not available. Install with:")
        print("pip install coqui-stt soundfile librosa")
        return False
    
    # Create CoquiSTT instance
    coqui = CoquiSTT()
    
    # Check dependencies
    deps_ok, deps_msg = coqui.check_dependencies()
    print(f"Dependencies: {deps_msg}")
    if not deps_ok:
        return False
    
    # Get available models
    print("\n📦 Available Models:")
    available_models = coqui.get_available_models()
    for model in available_models:
        status = "✅ Downloaded" if model["downloaded"] else "⬇️ Available for download"
        scorer_status = " (with scorer)" if model["has_scorer"] else " (no scorer)"
        print(f"  - {model['name']}: {model['description']} ({model['size']}) {status}{scorer_status}")
    
    # Test model loading
    print("\n🔄 Loading English Large model...")
    model_name = "english-large"
    success = coqui.load_model(
        model_name=model_name,
        auto_download=True,
        beam_width=512
    )
    
    if not success:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Get model info
    model_info = coqui.get_model_info()
    print(f"\n📋 Model Info:")
    for key, value in model_info.items():
        print(f"  - {key}: {value}")
    
    # Test transcription
    if audio_file and Path(audio_file).exists():
        print(f"\n🎤 Transcribing: {audio_file}")
    else:
        # Look for default recording
        default_files = [
            "recordings/recorded_audio.wav",
            "recorded_audio.wav",
            "test_audio.wav"
        ]
        
        audio_file = None
        for file_path in default_files:
            if Path(file_path).exists():
                audio_file = file_path
                break
        
        if not audio_file:
            print("❌ No audio file found for testing")
            print("Record audio using the Gradio interface first, or provide a file path")
            return False
        
        print(f"\n🎤 Using default recording: {audio_file}")
    
    # Perform transcription
    try:
        print("Transcribing...")
        result = coqui.transcribe_audio(
            audio_file_path=audio_file,
            return_confidence=True,
            return_timestamps=False
        )
        
        if "error" in result:
            print(f"❌ Transcription error: {result['error']}")
            return False
        
        print("\n📝 Transcription Results:")
        print(f"  Text: {result['text']}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Language: {result.get('language', 'Unknown')}")
        
        # Test with timestamps if successful
        print("\n🕐 Testing with timestamps...")
        result_with_timestamps = coqui.transcribe_audio(
            audio_file_path=audio_file,
            return_confidence=True,
            return_timestamps=True
        )
        
        if "words" in result_with_timestamps:
            print(f"  Word count: {len(result_with_timestamps['words'])}")
            if result_with_timestamps['words']:
                print("  First few words with timestamps:")
                for word in result_with_timestamps['words'][:3]:
                    print(f"    - '{word['word']}' at {word['start_time']:.2f}s (confidence: {word.get('confidence', 'N/A')})")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False
    
    # Cleanup
    coqui.cleanup()
    print("\n✅ Test completed successfully!")
    return True

def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Get audio file from command line if provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run test
    success = test_coqui_stt(audio_file)
    
    if success:
        print("\n🎉 Coqui STT is working correctly!")
        print("\n💡 Next steps:")
        print("  1. Run the main transcriber: python gradio_voice_transcriber_clean.py")
        print("  2. Select 'CoquiSTT' as your model")
        print("  3. Choose your preferred language model")
        print("  4. Start transcribing!")
    else:
        print("\n❌ Coqui STT test failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())