import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("/content/whisper-large-hii/checkpoint-9")
model = WhisperForConditionalGeneration.from_pretrained("/content/whisper-large-hii/checkpoint-9")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def transcribe_audio_in_chunks(model, processor, audio, sampling_rate, chunk_length=15):
    chunk_length_samples = int(chunk_length * sampling_rate)
    audio_chunks = [audio[i:i + chunk_length_samples] for i in range(0, len(audio), chunk_length_samples)]

    full_transcription = []

    for chunk in audio_chunks:
        input_features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                no_repeat_ngram_size=2
            )
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        full_transcription.append(" ".join(transcription))

    return " ".join(full_transcription)

audio, sr = librosa.load("/content/drive/MyDrive/indirilen_ses_dosyaları/ses_70.wav", sr=16000)
print("Final Transcription:", transcribe_audio_in_chunks(model, processor, audio, sr))




    
