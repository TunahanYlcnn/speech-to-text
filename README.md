# Speech-to-Text: Whisper & Vosk Project

Bu proje, OpenAI'ın **Whisper** modeli ve **Vosk** kütüphanesini kullanarak ses dosyalarını metne dönüştürmek, modelleri özel veri setleri ile eğitmek (fine-tuning) ve ses verilerini optimize etmek amacıyla geliştirilmiştir[cite: 1, 4, 6].

## 🚀 Özellikler

*   **Model Eğitimi (Fine-Tuning):** Kendi ses verilerinizle (Excel/CSV formatında) Whisper modelini Türkçe dili için özelleştirme.
*   **Gelişmiş Transkripsiyon:** Uzun ses dosyalarını 10-15 saniyelik parçalara (chunks) bölerek işleme ve yüksek doğruluklu metne dönüştürme[cite: 4, 6].
*   **Ses Ön İşleme:** Ses normalizasyonu ve gürültü azaltma (noise reduction) teknikleri ile daha temiz ve doğru sonuçlar[cite: 5, 6].
*   **Çoklu Model Desteği:** Yüksek performanslı Whisper Large modelleri ve hafif/hızlı Vosk çözümleri bir arada.

## 🛠️ Kurulum

Projenin çalışması için sisteminizde Python yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu terminalinizde çalıştırın:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers datasets pandas librosa evaluate jiwer accelerate openpyxl
```

## 📂 Proje Yapısı ve Kullanım
1. Model Eğitimi (Fine-Tuning)
whisperModeliFineTune.py dosyası, Whisper modelini kendi Excel verilerinizle eğitmek için tasarlanmıştır.  

Veri Seti: Ses dosyası yollarını ve metin karşılıklarını içeren bir Excel dosyası (kayit.xlsx) kullanır.  

Eğitim: Seq2SeqTrainer yapısı ile modelin Türkçe dilindeki başarısını artırır.  

2. Metne Dönüştürme (Inference)
Eğitilmiş modelleri (checkpoint) kullanarak ses dosyalarını metne çevirmek için:

load_and_preprocess_audio: Sesi normalize eder.  

transcribe_audio_in_chunks: Sesi parçalara bölerek num_beams=10 gibi yüksek doğruluk ayarlarıyla işler.
