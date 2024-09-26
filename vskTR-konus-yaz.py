import vosk
import sounddevice as sd
import numpy as np

# Modeli yükle
model = vosk.Model("C:/models/models/vosk-model-tr")

# Örnekleme hızı (samplerate)
samplerate = 16000

# Vosk için bir tanıyıcı başlat
rec = vosk.KaldiRecognizer(model, samplerate)

def recognize_speech():
    print("Listening...")
    with sd.RawInputStream(samplerate=samplerate, channels=1, dtype='int16', blocksize=8000) as stream:
        while True:
            data = stream.read(8000)[0]
            data = np.frombuffer(data, dtype=np.int16)
            
            if rec.AcceptWaveform(data.tobytes()):
                result = rec.Result()
                break
            else:
                result = rec.PartialResult()

    return result

# Konuşma tanıma döngüsü
bitir = 0
while bitir == 0:
    print("Konuşmaya başlayın...")
    text_json = recognize_speech()
    text = eval(text_json)["text"]  # Sonucu JSON'dan metne çevir
    
    if text:
        print(f"Söyledikleriniz = \"{text}\"")
    else:
        print("Anlaşılamadı, lütfen tekrar deneyin.")
    
    bitir = int(input("devam edelim mi (evet için 0 hayir için 1): "))

print("Program sonlandırıldı.")
