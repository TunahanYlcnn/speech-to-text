import vosk
import sounddevice as sd
import numpy as np
import json

# Modeli yükle
model = vosk.Model("C:/models/models/vosk-model-tr")

# Örnekleme hızı (samplerate)
samplerate = 16000

# Vosk için bir tanıyıcı başlat
rec = vosk.KaldiRecognizer(model, samplerate)

def recognize_speech():
    print("Dinleniyor...")
    with sd.RawInputStream(samplerate=samplerate, channels=1, dtype='int16', blocksize=8000) as stream:
        while True:
            # Mikrofondan veri oku
            data = stream.read(8000)[0]
            data = np.frombuffer(data, dtype=np.int16)
            
            # Ses verisini analiz et
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
    
    # eval yerine json.loads kullanarak veriyi güvenli şekilde çekiyoruz
    parsed_data = json.loads(text_json)
    text = parsed_data.get("text", "")
    
    if text:
        print(f"Söyledikleriniz = \"{text}\"")
    else:
        print("Anlaşılamadı, lütfen tekrar deneyin.")
    
    try:
        bitir = int(input("Devam edelim mi? (Evet için 0, Hayır için 1): "))
    except ValueError:
        bitir = 0

print("Program sonlandırıldı.")