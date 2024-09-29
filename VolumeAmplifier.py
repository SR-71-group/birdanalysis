import os
import scipy.io.wavfile as wavfile
import numpy as np

class VolumeAmplifier:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def read_wav(self, file_path):
        rate, data = wavfile.read(file_path)
        data = data.astype(np.float32)
        data = data / np.max(np.abs(data))
        return rate, data

    def increase_volume(self, data, volume_factor):
        amplified_data = data * volume_factor
        amplified_data = np.clip(amplified_data, -1.0, 1.0)
        return amplified_data

    def save_wav(self, rate, data, output_file_path):
        processed_int16 = (data * 32767).astype(np.int16)
        wavfile.write(output_file_path, rate, processed_int16)

    def process_volume_amplification(self, volume_factor):
        for filename in os.listdir(self.input_directory):
            if filename.endswith(".wav"):
                input_file_path = os.path.join(self.input_directory, filename)
                rate, data = self.read_wav(input_file_path)
                amplified_data = self.increase_volume(data, volume_factor)
                output_file_path = os.path.join(self.output_directory, f"amplified_{filename}")
                self.save_wav(rate, amplified_data, output_file_path)
                print(f"Amplified file saved as {output_file_path}")
