import whisper
import torch
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import os

# Cihazı belirle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper modelini yükle
model = whisper.load_model("large").to(device)

def rms_normalize(audio_path, output_path, target_rms=0.5):
    # Ses dosyasını yükle
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    # RMS değerini hesapla
    rms = np.sqrt(np.mean(np.square(audio_data)))
    
    # Hedef RMS değerine göre ölçekleme faktörünü hesapla
    scaling_factor = target_rms / rms
    
    # Ses verisini ölçeklendir
    normalized_audio = audio_data * scaling_factor
    
    # Normalizasyon uygulanmış ses dosyasını kaydet
    sf.write(output_path, normalized_audio, sample_rate)

def frame_and_window(audio_data, frame_size=2048, hop_size=512):
    frames = []
    num_frames = int(np.ceil(len(audio_data) / float(hop_size)))
    
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + frame_size, len(audio_data))
        frame = audio_data[start:end]
        
        # Pencereleme (Hamming penceresi)
        window = np.hamming(len(frame))
        frame *= window
        
        frames.append(frame)
    
    return frames

# Ses dosyasını yükle ve normalize et
file_path = "ses12.wav"
normalized_audio_path = "normalized_audio.wav"
rms_normalize(file_path, normalized_audio_path)

# Librosa ile ses dosyasını yükle
audio_data, sample_rate = librosa.load(normalized_audio_path, sr=None)

# Çerçeveleme ve pencerelendirme işlemi
frames = frame_and_window(audio_data)

# Gürültü giderme işlemi
reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.4, stationary=False)

# Gürültü giderilmiş ses dosyasını kaydet
output_file = "reduced_noise.wav"
sf.write(output_file, reduced_noise_audio, sample_rate)

# Transkripsiyon işlemi
result = model.transcribe(
    output_file, 
    language="en",
    fp16=True if device == "cuda" else False, 
    temperature=0.0,  # Daha tutarlı sonuçlar için sıfır sıcaklıkta transkribe ediyoruz
    condition_on_previous_text=True,  # Önceki metne dayalı olarak daha doğru tahmin yapar
    verbose=True  # Süreci takip etmek için ayrıntılı çıktılar sağlar
)

print("Söyledikleriniz = ", result["text"])

# Geçici dosyayı sil
os.remove(output_file)
