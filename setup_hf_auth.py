#!/usr/bin/env python3
"""
HuggingFace Authentication Helper

This script helps set up HuggingFace authentication for accessing private models.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_hf_cli():
    """Check if huggingface-hub CLI is available."""
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ HuggingFace CLI available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ HuggingFace CLI not found")
        return False

def install_hf_hub():
    """Install huggingface-hub package."""
    print("📦 Installing huggingface-hub...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], 
                      check=True)
        print("✅ huggingface-hub installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install huggingface-hub: {e}")
        return False

def login_to_hf():
    """Login to HuggingFace using CLI."""
    print("\n🔐 Logging in to HuggingFace...")
    print("This will open a browser to get your token.")
    print("If you don't have a token, create one at: https://huggingface.co/settings/tokens")
    
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
        print("✅ Successfully logged in to HuggingFace")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to login: {e}")
        return False

def check_auth_status():
    """Check current authentication status."""
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], 
                              capture_output=True, text=True, check=True)
        username = result.stdout.strip()
        print(f"✅ Logged in as: {username}")
        return True, username
    except subprocess.CalledProcessError:
        print("❌ Not logged in to HuggingFace")
        return False, None

def test_model_access():
    """Test access to the Arabic Egyptian model."""
    print("\n🧪 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test the main model
        models_to_test = [
            "jonatasgrosman/wav2vec2-large-xlsr-53-arabic-egyptian",
            "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
            "facebook/wav2vec2-large-xlsr-53"
        ]
        
        for model_id in models_to_test:
            try:
                print(f"Testing: {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"✅ {model_id} - Accessible")
                return True
            except Exception as e:
                print(f"❌ {model_id} - {str(e)}")
                continue
        
        print("❌ None of the models are accessible")
        return False
        
    except ImportError:
        print("❌ Transformers library not installed")
        return False

def manual_token_setup():
    """Guide user through manual token setup."""
    print("\n📝 Manual Token Setup")
    print("=" * 40)
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token (starts with 'hf_')")
    print("4. Use it in the Gradio interface:")
    print("   - Select 'Wav2Vec2ArabicSTT'")
    print("   - Choose 'Arabic Egyptian (Experimental)' model")
    print("   - Enter your token in 'HuggingFace Token' field")
    print("   - Click 'Load Model'")
    print("\n💡 Alternatively, set environment variable:")
    print("   export HF_TOKEN=your_token_here")

def main():
    """Main authentication helper."""
    print("🤗 HuggingFace Authentication Helper")
    print("=" * 50)
    
    # Check if already logged in
    is_logged_in, username = check_auth_status()
    
    if is_logged_in:
        print(f"\n✅ Already authenticated as: {username}")
        
        # Test model access
        if test_model_access():
            print("\n🎉 Authentication is working! You can use the experimental models.")
        else:
            print("\n⚠️  Authentication works but model access failed.")
            print("The experimental model might not be available.")
            print("Try using the standard Arabic model instead.")
        
        return 0
    
    # Not logged in, try to set up
    print("\n❌ Not authenticated with HuggingFace")
    
    # Check if CLI is available
    if not check_hf_cli():
        print("\n📦 Installing HuggingFace CLI...")
        if not install_hf_hub():
            print("\n❌ Failed to install HuggingFace Hub")
            manual_token_setup()
            return 1
    
    # Try to login
    print("\n🔐 Setting up authentication...")
    if login_to_hf():
        # Test access after login
        if test_model_access():
            print("\n🎉 Setup complete! You can now use all models.")
        else:
            print("\n⚠️  Login successful but some models may not be accessible.")
    else:
        print("\n❌ Automatic login failed")
        manual_token_setup()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())