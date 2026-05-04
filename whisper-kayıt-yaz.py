import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

model_name = "openai/whisper-large" 
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def resample_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None) 
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

def transcribe_audio(audio_path):
    audio, rate = resample_audio(audio_path, target_sr=16000)
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        logits = model.generate(inputs.input_features)
    transcription = processor.batch_decode(logits, skip_special_tokens=True)
    return transcription[0]

audio_file = "ses12.wav"
text = transcribe_audio(audio_file)
print("Söyledikleriniz = ", text)
