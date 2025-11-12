#!/usr/bin/env python3
"""
Simple Whisper Test

Load Whisper model and test transcription.
"""

import numpy as np
from stt import WhisperSTT

def main():
    print("Loading Whisper model...")
    WhisperSTT.load_model("tiny")  # Load tiny model (fastest)
    
    if not WhisperSTT.is_loaded:
        print("Failed to load model")
        return
    
    print("Model loaded successfully!")
    
    # Create some test audio (1 second of random noise)
    test_audio = np.random.randn(16000).astype(np.float32)
    
    print("Transcribing audio...")
    result = WhisperSTT.transcribe_numpy(test_audio, 16000)
    
    print(f"Result: {result.text}")
    print(f"Confidence: {result.confidence}")
    print(f"Time: {result.processing_time:.2f}s")

if __name__ == "__main__":
    main()