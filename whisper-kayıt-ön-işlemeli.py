import whisper
import torch
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


model = whisper.load_model("large").to(device)

def rms_normalize(audio_path, output_path, target_rms=0.5):

    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    rms = np.sqrt(np.mean(np.square(audio_data)))
    
    scaling_factor = target_rms / rms
    
    normalized_audio = audio_data * scaling_factor
    
    sf.write(output_path, normalized_audio, sample_rate)

def frame_and_window(audio_data, frame_size=2048, hop_size=512):
    frames = []
    num_frames = int(np.ceil(len(audio_data) / float(hop_size)))
    
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + frame_size, len(audio_data))
        frame = audio_data[start:end]
        
        window = np.hamming(len(frame))
        frame *= window
        
        frames.append(frame)
    
    return frames


file_path = "ses12.wav"
normalized_audio_path = "normalized_audio.wav"
rms_normalize(file_path, normalized_audio_path)
audio_data, sample_rate = librosa.load(normalized_audio_path, sr=None)

frames = frame_and_window(audio_data)

reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.4, stationary=False)

output_file = "reduced_noise.wav"
sf.write(output_file, reduced_noise_audio, sample_rate)

result = model.transcribe(
    output_file, 
    language="en",
    fp16=True if device == "cuda" else False, 
    temperature=0.0,  
    condition_on_previous_text=True, 
    verbose=True 
)

print("Söyledikleriniz = ", result["text"])

os.remove(output_file)
