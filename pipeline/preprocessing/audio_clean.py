from pipeline.preprocessing.ffmpeg import to_wav_16k_mono

def clean_audio(input_path, output_path):
    to_wav_16k_mono(input_path, output_path)