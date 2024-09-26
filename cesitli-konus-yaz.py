import speech_recognition as sr
from langdetect import detect  # Bu örnekte langdetect kütüphanesi kullanılacak

def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return None

r = sr.Recognizer()
bitir = 0

while bitir == 0:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("Söyledikleriniz = ", text)
            
            # Dil tanıma işlemi
            detected_language = detect_language(text)
            print("Tespit edilen dil: ", detected_language)
            
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        
        bitir = int(input("devam edelim mi (evet için 0 hayir için 1)"))
