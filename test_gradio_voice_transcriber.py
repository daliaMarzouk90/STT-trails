import types
import numpy as np
import builtins
import time
import importlib
import pytest

# Import the module under test
import gradio_voice_transcriber as gvt


class DummySTT:
    is_loaded = True

    def __init__(self):
        self._language = None

    def load_model(self, **kwargs):
        self.is_loaded = True

    def set_language(self, lang):
        self._language = lang

    def transcribe_audio(self, audio, sample_rate):
        # Return an object mimicking STTResult
        class R:
            def __init__(self):
                self.text = "hello world"
                self.confidence = 0.75
                self.processing_time = 0.05
        return R()

    @staticmethod
    def get_model_info():
        return {"is_loaded": True, "model_name": "DummySTT"}


class DummyTawasul:
    # Static style class (no instantiation) used by code path
    is_loaded = True

    @staticmethod
    def load_model(**kwargs):
        DummyTawasul.is_loaded = True

    @staticmethod
    def get_model_info():
        return {"is_loaded": True, "model_name": "TawasulSTT"}

    @staticmethod
    def transcribe(path):
        # Return tuple like (text, confidence_info, processing_info)
        return ("transcribed from file", "Confidence: 0.42", "ok")


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    # Ensure clean state between tests
    gvt.current_stt_model = None
    gvt.current_model_config = {}
    yield
    gvt.current_stt_model = None
    gvt.current_model_config = {}


def test_audio_processor_preprocess_basic():
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = gvt.AudioProcessor.preprocess(audio, sr, target_sr=16000)
    # Should be float32, mono, and clipped range within [-1,1]
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert np.max(np.abs(out)) <= 1.0


def test_model_manager_load_whisper_missing_api_key_returns_error(monkeypatch):
    # Register DummySTT under WhisperSTT name to avoid heavy import
    monkeypatch.setitem(gvt.STT_MODELS, "WhisperSTT", DummySTT)
    # Request API mode without key
    msg = gvt.ModelManager.load_model("WhisperSTT", model_size="base", use_api=True, api_key="")
    assert "API key required" in msg


def test_model_manager_load_generic_success(monkeypatch):
    # Register a generic model name and load
    monkeypatch.setitem(gvt.STT_MODELS, "DummySTT", DummySTT)
    msg = gvt.ModelManager.load_model("DummySTT")
    assert msg.startswith("✅")
    assert gvt.current_stt_model is not None


def test_transcription_engine_no_audio():
    text, conf, proc = gvt.TranscriptionEngine.transcribe(None, language="en")
    assert text.startswith("❌ No audio provided")


def test_transcription_engine_requires_loaded_model():
    # Provide dummy audio but no model
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    text, conf, proc = gvt.TranscriptionEngine.transcribe((sr, audio), language="en")
    assert "No STT model loaded" in text


def test_transcription_engine_happy_path(monkeypatch):
    # Use DummySTT and set as the loaded model
    gvt.current_stt_model = DummySTT()
    gvt.current_model_config = {"model_name": "DummySTT"}
    # Provide a 1 second tone with enough amplitude
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    text, conf, proc = gvt.TranscriptionEngine.transcribe((sr, audio), language="en")
    assert text == "hello world"
    assert conf.startswith("Confidence: ")
    assert "Processing:" in proc


def test_transcription_engine_filters_false_positives(monkeypatch):
    class LowTextDummy(DummySTT):
        def transcribe_audio(self, audio, sample_rate):
            class R:
                def __init__(self):
                    self.text = "you"  # a known false positive which should be filtered
                    self.confidence = None
                    self.processing_time = 0.01
            return R()

    gvt.current_stt_model = LowTextDummy()
    gvt.current_model_config = {"model_name": "LowTextDummy"}
    sr = 16000
    audio = np.ones(sr, dtype=np.float32) * 0.2
    text, conf, proc = gvt.TranscriptionEngine.transcribe((sr, audio), language="en")
    assert text == "🔇 No clear speech detected"


def test_transcription_engine_tawasul_static_path_flow(monkeypatch, tmp_path):
    # Force the Tawasul path by setting current_model_config model_name
    gvt.current_stt_model = DummyTawasul
    gvt.current_model_config = {"model_name": "TawasulSTT"}

    # Create a simple audio array meeting quality gates
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Monkeypatch soundfile.write to write to the provided path without needing soundfile dependency
    written = {}

    def fake_write(path, data, samplerate):
        written["path"] = path
        written["samplerate"] = samplerate
        written["len"] = len(data)

    monkeypatch.setitem(builtins.__dict__, "__SOUNDFILE_WRITE__", fake_write)

    # Patch import inside function to use our fake write via simple shim
    import types as _types

    class SFShim:
        @staticmethod
        def write(path, data, samplerate):
            fake_write(path, data, samplerate)

    monkeypatch.setitem(importlib.import_module("soundfile").__dict__ if False else globals(), "sf", SFShim)

    # Run transcription
    text, conf, proc = gvt.TranscriptionEngine.transcribe((sr, audio), language="ar")
    assert text == "transcribed from file"
    assert conf.startswith("Confidence: ")
    assert "Model: TawasulSTT" in proc


def test_get_quality_recommendations_messages():
    q = {
        "duration": 0.5,
        "max_amplitude": 0.95,
        "clipping_ratio": 0.02,
        "silence_ratio": 0.6,
    }
    msg = gvt._get_quality_recommendations(q)
    # Expect multiple recommendations due to thresholds
    assert "recording for longer" in msg
    assert "clipping" in msg
    assert "Too much silence" in msg
