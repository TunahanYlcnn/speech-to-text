import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("/content/whisper-large-hi/checkpoint-36")
processor = WhisperProcessor.from_pretrained("/content/whisper-large-hi/checkpoint-36")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def load_and_preprocess_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    audio = librosa.util.normalize(audio)
    return audio, sr

def transcribe_audio_in_chunks(model, processor, audio, sampling_rate, chunk_length=10):
    chunk_length_samples = int(chunk_length * sampling_rate)
    audio_chunks = [audio[i:i + chunk_length_samples] for i in range(0, len(audio), chunk_length_samples)]

    full_transcription = []

    for chunk in audio_chunks:

        chunk = librosa.effects.trim(chunk, top_db=20)[0]
        input_features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_features
        input_features = input_features.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=512,
                num_beams=10,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_k=50,
                length_penalty=1.0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        transcription = processor.tokenizer.batch_decode(predicted_ids.sequences, skip_special_tokens=True)
        full_transcription.append(" ".join(transcription))

    return " ".join(full_transcription)


audio, sr = load_and_preprocess_audio("/content/drive/MyDrive/indirilen_ses_dosyaları/ses_70.wav")
print("Final Transcription:", transcribe_audio_in_chunks(model, processor, audio, sr))
