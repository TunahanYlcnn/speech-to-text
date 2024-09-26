import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Whisper modelini ve processor'ı yükleyin
model_name = "openai/whisper-large"  # Model boyutunu seçin
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def resample_audio(audio_path, target_sr=16000):
    # Ses dosyasını yükleyin ve hedef örnekleme oranına dönüştürün
    audio, sr = librosa.load(audio_path, sr=None)  # Orijinal örnekleme oranını alır
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

def transcribe_audio(audio_path):
    # Ses dosyasını uygun örnekleme oranına dönüştürün
    audio, rate = resample_audio(audio_path, target_sr=16000)
    
    # Ses verisini model için uygun biçime dönüştürün
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    
    # Modeli çalıştırın ve logits'i alın
    with torch.no_grad():
        logits = model.generate(inputs.input_features)
    
    # Metne dönüştürün
    transcription = processor.batch_decode(logits, skip_special_tokens=True)
    return transcription[0]

# Ses dosyasını işleyin
audio_file = "ses12.wav"
text = transcribe_audio(audio_file)
print("Söyledikleriniz = ", text)
