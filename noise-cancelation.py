import os
import scipy.io.wavfile as wavfile
import numpy as np
import noisereduce as nr

class NoiseReducer:
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

    def reduce_noise(self, file_path):
        rate, data = self.read_wav(file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        return rate, reduced_noise

    def save_wav(self, rate, data, output_file_path):
        processed_int16 = (data * 32767).astype(np.int16)
        wavfile.write(output_file_path, rate, processed_int16)

    def process_noise_reduction(self):
        for filename in os.listdir(self.input_directory):
            if filename.endswith(".wav"):
                input_file_path = os.path.join(self.input_directory, filename)
                rate, reduced_data = self.reduce_noise(input_file_path)
                output_file_path = os.path.join(self.output_directory, f"noise_reduced_{filename}")
                self.save_wav(rate, reduced_data, output_file_path)
                print(f"Noise reduced file saved as {output_file_path}")
