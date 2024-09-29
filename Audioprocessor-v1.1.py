import os
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import numpy as np
import noisereduce as nr

class AudioProcessor:
    def __init__(self, input_directory, output_directory, noise_reduction_active, volume_increase_active):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.noise_reduction_active = noise_reduction_active
        self.volume_increase_active = volume_increase_active

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

    def read_wav(self, file_path):
        rate, data = wavfile.read(file_path)
        data = data.astype(np.float32)
        data = data / np.max(np.abs(data))  # Normalize audio
        return rate, data

    def reduce_noise(self, data, rate):
        return nr.reduce_noise(y=data, sr=rate)

    def increase_volume(self, data, volume_factor):
        amplified_data = data * volume_factor
        amplified_data = np.clip(amplified_data, -1.0, 1.0)
        return amplified_data

    def process_and_cut_audio(self, file_path, volume_factor, start_ms, end_ms):
        rate, data = self.read_wav(file_path)

        # Apply noise reduction if selected by the user
        if self.noise_reduction_active:
            print("Reducing noise...")
            data = self.reduce_noise(data, rate)
        else:
            print("Skipping noise reduction.")

        # Apply volume increase if selected by the user
        if self.volume_increase_active:
            print("Increasing volume...")
            data = self.increase_volume(data, volume_factor)
        else:
            print("Skipping volume increase.")

        # Convert processed data back to int16 for saving as .wav
        processed_int16 = (data * 32767).astype(np.int16)
        temp_wav_file = f"{self.output_directory}/temp.wav"
        wavfile.write(temp_wav_file, rate, processed_int16)

        # Cut the audio based on the provided time
        audio = AudioSegment.from_wav(temp_wav_file)
        cut_audio = audio[start_ms:end_ms]

        # Clean up the temp file
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

        return cut_audio

def parse_filename(filename):
    parts = filename.split('___')
    time_range = parts[2].replace('s', '')
    start_time, end_time = map(float, time_range.split('-'))
    return start_time * 1000, end_time * 1000

# Ask the user for input_directory and output_directory
input_directory = input("Please enter the input directory path: ")
output_directory = input("Please enter the output directory path: ")

# Ask for noise reduction and volume increase options only once for all files
print("For audio processing, please specify the following options:")
noise_reduction_active = input("Do you want to activate noise reduction for all files? (yes/no): ").lower() == "yes"
volume_increase_active = input("Do you want to activate volume increase for all files? (yes/no): ").lower() == "yes"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize the processor with user inputs
processor = AudioProcessor(input_directory, output_directory, noise_reduction_active, volume_increase_active)

# Process all .wav files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_file_path = os.path.join(input_directory, filename)
        start_ms, end_ms = parse_filename(filename)
        processed_and_cut_audio = processor.process_and_cut_audio(input_file_path, volume_factor=5.0, start_ms=start_ms, end_ms=end_ms)
        cut_file_path = os.path.join(output_directory, f"{filename}")
        processed_and_cut_audio.export(cut_file_path, format="wav")
        print(f"Processed and cut {filename} -> Saved as {cut_file_path}")
