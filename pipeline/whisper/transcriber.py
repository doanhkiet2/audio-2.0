from pipeline.whisper.model_loader import load_model

def transcribe(audio_path):
    model = load_model()
    result = model.transcribe(str(audio_path))
    return result["segments"]