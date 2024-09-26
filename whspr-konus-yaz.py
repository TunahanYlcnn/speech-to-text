import whisper
import speech_recognition as sr
import tempfile
import os
import torch

# GPU'yu kullanmak için uygun cihazı belirleyin
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Whisper modelini yükleyin ve GPU'da çalışacak şekilde ayarlayın
model = whisper.load_model("small").to(device)
r = sr.Recognizer()

bitir = 0
while bitir == 0:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("dinleniyor...")
        audio = r.listen(source)
        print("işleniyor...")

        # Geçici bir dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio.get_wav_data())
            temp_filename = temp_audio_file.name

        # Whisper ile transkripsiyon yap
        result = model.transcribe(temp_filename, fp16=True if device == "cuda" else False)
        
        print("Söyledikleriniz = ", result["text"])
        
        # Geçici dosyayı sil
        os.remove(temp_filename)

        bitir = int(input("devam edelim mi (evet için 0 hayır için 1): "))
