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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large").to(device)

file_path = "/content/drive/MyDrive/05325148851-8503052459-queue-Lipyum-16052024-150817-486604978.wav"
audio = AudioSegment.from_file(file_path)
normalized_audio = audio.normalize()
normalized_audio.export("/content/drive/MyDrive/normalized_audio.wav", format="wav")

audio_data, sample_rate = librosa.load("/content/drive/MyDrive/normalized_audio.wav", sr=None)

reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.6)

output_file = "/content/drive/MyDrive/reduced_noise.wav"
sf.write(output_file, reduced_noise_audio, sample_rate)

result = model.transcribe(
    output_file, 
    language="tr", 
    fp16=True if device == "cuda" else False, 
    condition_on_previous_text=True, 
    verbose=True 
)

print("Söyledikleriniz = ", result["text"])
os.remove(output_file)