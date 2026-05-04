import vosk
import sounddevice as sd
import numpy as np

model = vosk.Model("C:/models/models/vosk-model-tr")
samplerate = 16000

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    
    data = np.frombuffer(indata, dtype=np.int16)
    
    if rec.AcceptWaveform(data.tobytes()):
        print(rec.Result(), flush=True)
    else:
        print(rec.PartialResult(), flush=True)

rec = vosk.KaldiRecognizer(model, samplerate)

with sd.RawInputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback, blocksize=8000):
    print("Konuşmaya başlayın...")
    while True:
        try:
            sd.sleep(1000)
        except KeyboardInterrupt:
            break

print("Program sonlandırıldı.")