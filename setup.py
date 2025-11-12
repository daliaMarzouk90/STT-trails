#!/usr/bin/env python3
"""
Setup script for Modular Voice Transcriber

This script helps set up the environment and install dependencies
based on which STT models you want to use.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"📦 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def install_requirements(requirements_file):
    """Install requirements from a specific file."""
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        f"Installing requirements from {requirements_file}"
    )

def install_optional_dependencies(groups):
    """Install optional dependencies using pip install -e ."""
    group_str = ",".join(groups)
    return run_command(
        f"pip install -e .[{group_str}]",
        f"Installing optional dependencies: {group_str}"
    )

def test_imports(modules):
    """Test if modules can be imported."""
    print("\n🔍 Testing module imports...")
    all_good = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description="Setup Modular Voice Transcriber")
    parser.add_argument(
        "--profile",
        choices=["minimal", "essential", "whisper-only", "wav2vec2-only", "vosk-only", "hubert-only", "coqui-only", "tawasul-only", "all"],
        default="essential",
        help="Installation profile (default: essential)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the installation after setup"
    )
    
    args = parser.parse_args()
    
    print("🚀 Modular Voice Transcriber Setup")
    print("=" * 50)
    print(f"Profile: {args.profile}")
    print()
    
    # Install base requirements first
    print("📦 Installing base requirements...")
    base_success = run_command(
        "pip install gradio>=4.0.0 numpy>=1.21.0 soundfile>=0.12.1",
        "Installing base dependencies"
    )
    
    if not base_success:
        print("❌ Failed to install base requirements. Exiting.")
        return 1
    
    # Install profile-specific requirements
    success = True
    
    if args.profile == "minimal":
        print("\n📦 Minimal installation - Gradio interface only")
        # Base requirements already installed
        
    elif args.profile == "essential":
        print("\n📦 Essential installation - Whisper + Wav2Vec2")
        success = install_optional_dependencies(["essential"])
        
    elif args.profile == "whisper-only":
        print("\n📦 Whisper-only installation")
        success = install_requirements("requirements_whisper.txt")
        
    elif args.profile == "wav2vec2-only":
        print("\n📦 Wav2Vec2-only installation")
        success = install_requirements("requirements_wav2vec2.txt")
        
    elif args.profile == "vosk-only":
        print("\n📦 Vosk-only installation")
        success = install_requirements("requirements_vosk.txt")
        
    elif args.profile == "hubert-only":
        print("\n📦 HuBERT Arabic-only installation")
        success = install_requirements("requirements_hubert.txt")
        
    elif args.profile == "coqui-only":
        print("\n📦 Coqui STT-only installation")
        success = install_requirements("requirements_coqui.txt")
        
    elif args.profile == "tawasul-only":
        print("\n📦 Tawasul STT-only installation")
        success = install_requirements("requirements_tawasul.txt")
        
    elif args.profile == "all":
        print("\n📦 Full installation - All STT models")
        success = install_optional_dependencies(["all-stt"])
    
    if not success:
        print(f"❌ Failed to install {args.profile} profile requirements.")
        return 1
    
    # Test installation if requested
    if args.test:
        print("\n🧪 Testing installation...")
        
        # Basic imports
        basic_modules = ["gradio", "numpy", "soundfile"]
        test_imports(basic_modules)
        
        # Profile-specific tests
        if args.profile in ["essential", "whisper-only", "all"]:
            whisper_modules = ["whisper", "openai"]
            test_imports(whisper_modules)
        
        if args.profile in ["essential", "wav2vec2-only", "hubert-only", "tawasul-only", "all"]:
            wav2vec2_modules = ["transformers", "torch", "torchaudio"]
            test_imports(wav2vec2_modules)
        
        # Test our modules
        try:
            from stt.stt_base import BaseSTT
            from stt.whisper_stt import WhisperSTT
            print("✅ STT base classes")
        except ImportError as e:
            print(f"❌ STT base classes: {e}")
        
        if args.profile in ["essential", "wav2vec2-only", "all"]:
            try:
                from stt.wav2vec2_arabic_stt import Wav2Vec2ArabicSTT
                print("✅ Wav2Vec2 Arabic STT")
            except ImportError as e:
                print(f"❌ Wav2Vec2 Arabic STT: {e}")
        
        if args.profile in ["hubert-only", "all"]:
            try:
                from stt.hubert_arabic_stt import HuBERTArabicSTT
                print("✅ HuBERT Arabic STT")
            except ImportError as e:
                print(f"❌ HuBERT Arabic STT: {e}")
        
        if args.profile in ["coqui-only", "all"]:
            try:
                from stt.coqui_stt import CoquiSTT
                print("✅ Coqui STT")
            except ImportError as e:
                print(f"❌ Coqui STT: {e}")
        
        if args.profile in ["tawasul-only", "all"]:
            try:
                from stt.tawasul_stt import TawasulSTT
                print("✅ Tawasul STT")
            except ImportError as e:
                print(f"❌ Tawasul STT: {e}")
        
        if args.profile in ["vosk-only", "all"]:
            try:
                from stt.vosk_stt import VoskSTT
                print("✅ Vosk STT")
            except ImportError as e:
                print(f"❌ Vosk STT: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n💡 Next steps:")
    print("   1. Run the transcriber:")
    print("      python gradio_voice_transcriber_clean.py")
    print("\n   2. Or test specific models:")
    print("      python test_wav2vec2_arabic.py")
    print("\n   3. Check available models in the web interface")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())