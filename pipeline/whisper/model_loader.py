import whisper

_model = None

def load_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model