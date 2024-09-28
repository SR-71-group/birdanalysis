import os
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import numpy as np
import noisereduce as nr


class AudioProcessor:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

    def read_wav(self, file_path):
        rate, data = wavfile.read(file_path)
        data = data.astype(np.float32)
        data = data / np.max(np.abs(data))  
        return rate, data

    def reduce_noise(self, data, rate):
        return nr.reduce_noise(y=data, sr=rate)

    def increase_volume(self, data, volume_factor):
        amplified_data = data * volume_factor
        amplified_data = np.clip(amplified_data, -1.0, 1.0)

        return amplified_data

    def process_and_cut_audio(self, file_path, volume_factor, start_ms, end_ms):
        rate, data = self.read_wav(file_path)
        reduced_noise = self.reduce_noise(data, rate)
        amplified_data = self.increase_volume(reduced_noise, volume_factor)
        processed_int16 = (amplified_data * 32767).astype(np.int16)
        temp_wav_file = f"{self.output_directory}/temp.wav"
        wavfile.write(temp_wav_file, rate, processed_int16)
        audio = AudioSegment.from_wav(temp_wav_file)
        cut_audio = audio[start_ms:end_ms]
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

        return cut_audio

def parse_filename(filename):
    parts = filename.split('___')
    time_range = parts[2].replace('s', '') 
    start_time, end_time = map(float, time_range.split('-'))
    return start_time * 1000, end_time * 1000 

input_directory = r"C:\"  
output_directory = r"C:\"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
processor = AudioProcessor(input_directory, output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_file_path = os.path.join(input_directory, filename)
        start_ms, end_ms = parse_filename(filename)
        processed_and_cut_audio = processor.process_and_cut_audio(input_file_path, volume_factor=5.0, start_ms=start_ms, end_ms=end_ms)
        cut_file_path = os.path.join(output_directory, f"{filename}")
        processed_and_cut_audio.export(cut_file_path, format="wav")
        print(f"Processed and cut {filename} -> Saved as {cut_file_path}")
