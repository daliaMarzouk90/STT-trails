#!/usr/bin/env python3
"""
Test script for Tawasul STT V0 model

This script tests the Tawasul STT V0 Arabic speech recognition model
with sample audio files.

Usage:
    python test_tawasul.py [audio_file]
    
If no audio file is provided, it will test with any files in the recordings/ directory.
"""

import sys
import os
from pathlib import Path
import time
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from stt.tawasul_stt import TawasulSTT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tawasul_stt(audio_file: str = None):
    """Test Tawasul STT with an audio file."""
    
    print("🧪 Tawasul STT V0 Test")
    print("=" * 50)
    
    # Check if Tawasul STT is available
    if not TawasulSTT.is_available():
        print("❌ Tawasul STT dependencies not available!")
        print("Install with: pip install -r requirements_tawasul.txt")
        return False
    
    # Find audio file if not provided
    if not audio_file:
        recordings_dir = Path("recordings")
        if recordings_dir.exists():
            audio_files = list(recordings_dir.glob("*.wav")) + list(recordings_dir.glob("*.mp3"))
            if audio_files:
                audio_file = str(audio_files[0])
                print(f"🎵 Using sample audio: {audio_file}")
            else:
                print("❌ No audio files found in recordings/ directory")
                print("Please provide an audio file: python test_tawasul.py your_audio.wav")
                return False
        else:
            print("❌ No audio file provided and no recordings/ directory found")
            print("Usage: python test_tawasul.py your_audio.wav")
            return False
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        # Load the model (static method)
        print("📥 Loading Tawasul STT V0 model...")
        start_time = time.time()
        
        TawasulSTT.load_model(
            device="auto",  # Automatically choose best device
            chunk_length=20,  # 20-second chunks
            max_audio_length=300  # 5 minutes max
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Get model info (static method)
        model_info = TawasulSTT.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Name: {model_info['name']}")
        print(f"   Model ID: {model_info['model_id']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Specialization: {model_info['specialization']}")
        print(f"   Supported Languages: {', '.join(model_info['supported_languages'][:5])}...")
        
        # Transcribe audio (static method)
        print(f"\n🎙️ Transcribing audio: {audio_file}")
        transcription, confidence_info, processing_info = TawasulSTT.transcribe(audio_file)
        
        # Display results
        print("\n" + "=" * 50)
        print("📝 TRANSCRIPTION RESULTS")
        print("=" * 50)
        print(f"Text: {transcription}")
        print(f"Confidence: {confidence_info}")
        print(f"Processing: {processing_info}")
        print("=" * 50)
        
        if transcription and not transcription.startswith("❌"):
            print("✅ Transcription successful!")
            return True
        else:
            print("❌ Transcription failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main function."""
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Test with different configurations
    success = test_tawasul_stt(audio_file)
    
    if success:
        print("\n🎉 Tawasul STT test completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Try the main transcriber: python gradio_voice_transcriber_clean.py")
        print("   2. Test with different Arabic audio files")
        print("   3. Experiment with different model variants")
    else:
        print("\n❌ Tawasul STT test failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements_tawasul.txt")
        print("   2. Check audio file format (WAV/MP3)")
        print("   3. Ensure stable internet for model download")
        print("   4. Try with a different audio file")

if __name__ == "__main__":
    main()