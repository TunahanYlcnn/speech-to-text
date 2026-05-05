from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa

model = WhisperForConditionalGeneration.from_pretrained("/content/whisper-large-hi/checkpoint-36")
processor = WhisperProcessor.from_pretrained("/content/whisper-large-hi/checkpoint-36")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

audio, sr = librosa.load("/content/drive/MyDrive/indirilen_ses_dosyaları/ses_70.wav", sr=16000)
audio = librosa.util.normalize(audio)
audio = librosa.effects.trim(audio, top_db=20)[0]
chunk_length = 10 * sr
audio_chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

full_transcription = []
for chunk in audio_chunks:
    input_features = processor.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding="longest").input_features
    input_features = input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=512,
            num_beams=5,
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

final_transcription = " ".join(full_transcription)

print("Final Transcription:", final_transcription)
