# Gerekli paketleri yükleyin
# %%capture
# !pip install git+https://github.com/openai/whisper.git
# !pip install torch
# !pip install noisereduce
# !pip install librosa
# !pip install soundfile
# !pip install pydub
# !apt-get install ffmpeg
import whisper
import torch
import noisereduce as nr
import librosa
import soundfile as sf
from pydub import AudioSegment 
import os

# Cihazı belirle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper modelini yükle
model = whisper.load_model("large").to(device)

# Ses dosyasını yükle ve normalize et
file_path = "/content/drive/MyDrive/05325148851-8503052459-queue-Lipyum-16052024-150817-486604978.wav"
audio = AudioSegment.from_file(file_path)
normalized_audio = audio.normalize()
normalized_audio.export("/content/drive/MyDrive/normalized_audio.wav", format="wav")

# Librosa ile ses dosyasını yükle
audio_data, sample_rate = librosa.load("/content/drive/MyDrive/normalized_audio.wav", sr=None)

# Gürültü giderme işlemi
reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.6)

# Gürültü giderilmiş ses dosyasını geçici bir dosyaya kaydet
output_file = "/content/drive/MyDrive/reduced_noise.wav"
sf.write(output_file, reduced_noise_audio, sample_rate)

# Transkripsiyon işlemi
result = model.transcribe(
    output_file, 
    language="tr", 
    fp16=True if device == "cuda" else False, 
    temperature=0.0,  # Daha tutarlı sonuçlar için sıfır sıcaklıkta transkribe ediyoruz
    condition_on_previous_text=True,  # Önceki metne dayalı olarak daha doğru tahmin yapar
    verbose=True  # Süreci takip etmek için ayrıntılı çıktılar sağlar
)

print("Söyledikleriniz = ", result["text"])

# Geçici dosyayı sil
os.remove(output_file)