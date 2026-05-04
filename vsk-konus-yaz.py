import vosk
import sounddevice as sd
import json

model = vosk.Model("C:/models/models/vosk-model-tr")
samplerate = 16000

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    
    if rec.AcceptWaveform(indata.tobytes()):
        sonuc = json.loads(rec.Result())
        if sonuc["text"]:
            print(f"\nSöylenen: {sonuc['text']}")
    else:
        kismi_sonuc = json.loads(rec.PartialResult())
        if kismi_sonuc["partial"]:
            print(f"Algılanıyor: {kismi_sonuc['partial']}", end="\r")

rec = vosk.KaldiRecognizer(model, samplerate)

try:
    with sd.RawInputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback, blocksize=4000):
        print("Sistem Hazır! Konuşmaya başlayın (Çıkmak için Ctrl+C)...")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nProgram kullanıcı tarafından sonlandırıldı.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")