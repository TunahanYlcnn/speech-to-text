import vosk
import sounddevice as sd
import json
import numpy as np

# Model dosyasının yolunu belirtin
# Modeli yükle
model = vosk.Model("C:/models/models/vosk-model-tr")

# Ses girişini ayarlamak için örnekleme hızı
samplerate = 16000

# Ses verilerini almak için bir fonksiyon
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    
    # indata'yı numpy dizisine dönüştür
    data = np.frombuffer(indata, dtype=np.int16)
    
    # Veriyi bytes formatına çevir ve AcceptWaveform'a gönder
    if not rec.AcceptWaveform(data.tobytes()):
        print(rec.PartialResult(), flush=True)
    else:
        print(rec.Result(), flush=True)


# Vosk için bir tanıyıcı başlat
rec = vosk.KaldiRecognizer(model, samplerate)

# Mikrofon akışını başlat
with sd.RawInputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback, blocksize=8000):
    print("Konuşmaya başlayın...")
    while True:
        try:
            pass  # Burada sürekli dinlemede kalır
        except KeyboardInterrupt:
            break

print("Program sonlandırıldı.")
