#!/usr/bin/env python3
"""
Test script for Vosk STT implementation

This script tests the Vosk STT implementation to ensure compatibility
and proper error handling.
"""

import sys
import numpy as np
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_vosk_basic():
    """Test basic Vosk functionality."""
    print("🔍 Testing Vosk STT...")
    
    try:
        from stt.vosk_stt import VoskSTT
        print("✅ Successfully imported VoskSTT")
    except ImportError as e:
        print(f"❌ Failed to import VoskSTT: {e}")
        print("\n📦 Required dependencies:")
        print("pip install vosk")
        return False
    
    # Check Vosk availability
    print("\n📊 Checking Vosk availability...")
    models_info = VoskSTT.get_available_models()
    
    for key, value in models_info.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    if not models_info.get("vosk_available", False):
        print("\n❌ Vosk not available. Cannot proceed with test.")
        return False
    
    # Test model loading with a small model
    print(f"\n🔄 Testing model loading...")
    print("⚠️  Note: This will download a small model (~40MB) if not cached")
    
    try:
        # Try to load a small English model first
        VoskSTT.load_model(model_name="vosk-model-small-en-us-0.15")
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = VoskSTT.get_model_info()
        print(f"📋 Model info:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 This might be due to:")
        print("   - Network issues (model download)")
        print("   - Vosk version compatibility")
        print("   - Model availability")
        return False
    
    # Test with dummy audio
    print(f"\n🎤 Testing transcription with dummy audio...")
    print("⚠️  Note: Random audio won't produce meaningful text")
    
    try:
        # Create 2 seconds of random audio
        dummy_audio = np.random.randn(32000).astype(np.float32) * 0.1
        
        result = VoskSTT.transcribe_audio(dummy_audio, 16000)
        
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

def test_vosk_models():
    """Test different Vosk models."""
    print(f"\n📋 Testing different Vosk models...")
    
    try:
        from stt.vosk_stt import VoskSTT
        
        # Get available models
        available = VoskSTT.AVAILABLE_MODELS
        print(f"📊 Available models: {len(available)}")
        
        # Show a few interesting models
        interesting_models = [
            "vosk-model-small-en-us-0.15",
            "vosk-model-small-ru-0.22", 
            "vosk-model-small-fr-0.22",
            "vosk-model-small-de-0.15"
        ]
        
        print("\n🌍 Some available models:")
        for model_name in interesting_models:
            if model_name in available:
                model_info = available[model_name]
                print(f"   {model_name}:")
                print(f"     Language: {model_info['language']}")
                print(f"     Size: {model_info['size']}")
                print(f"     Description: {model_info['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
        return False

def test_integration():
    """Test integration with the modular transcriber."""
    print(f"\n🔗 Testing integration with modular transcriber...")
    
    try:
        from gradio_voice_transcriber_clean import ModelManager
        
        available_models = ModelManager.get_available_models()
        print(f"📋 Available models: {available_models}")
        
        if "VoskSTT" in available_models:
            print("✅ VoskSTT is registered in the modular transcriber")
            
            # Test model options
            options = ModelManager.get_model_options("VoskSTT")
            print(f"📊 Model options: {options}")
            
            return True
        else:
            print("❌ VoskSTT not found in available models")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import modular transcriber components: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Vosk STT Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    basic_test = test_vosk_basic()
    
    # Test model listing
    models_test = test_vosk_models()
    
    # Test integration
    integration_test = test_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Basic Functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"   Model Listing: {'✅ PASS' if models_test else '❌ FAIL'}")
    print(f"   Integration: {'✅ PASS' if integration_test else '❌ FAIL'}")
    
    if basic_test and models_test and integration_test:
        print("\n🎉 All tests passed! Vosk STT is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run: python gradio_voice_transcriber_clean.py")
        print("   2. Select 'VoskSTT' from the dropdown")
        print("   3. Choose your model (small models are faster)")
        print("   4. Load the model and test with audio!")
        print("\n🌍 Vosk supports many languages offline!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        
        if not basic_test:
            print("\n📦 To fix Vosk issues:")
            print("   pip install vosk")
            print("   Check internet connection for model download")

if __name__ == "__main__":
    main()