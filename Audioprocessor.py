import scipy.io.wavfile as wavfile
import numpy as np
import noisereduce as nr 

class AudioProcessor:
    def __init__(self, file_path):
        # Load the wav file
        self.reduced_noise_data = None
        self.rate, self.data = wavfile.read(file_path)
        self.original_data = self.data.astype(np.float32)
        self.data = self.original_data / np.max(np.abs(self.original_data))  # Normalize to [-1, 1]

    def reduce_noise(self):
        self.reduced_noise_data = nr.reduce_noise(y=self.data, sr=self.rate)

    def increase_volume(self, volume_factor):
        if hasattr(self, 'reduced_noise_data'):
            amplified_data = self.reduced_noise_data * volume_factor
        else:
            amplified_data = self.data * volume_factor

        # Ensure no clipping by limiting the range between -1 and 1
        self.amplified_data = np.clip(amplified_data, -1.0, 1.0)

    def save_amplified(self, output_file):
        # Convert back to original data type 16-bit
        amplified_int16 = (self.amplified_data * 32767).astype(np.int16)
        wavfile.write(output_file, self.rate, amplified_int16)

    def process_and_save(self, output_file, volume_factor=5.0):
        self.reduce_noise()
        self.increase_volume(volume_factor)
        self.save_amplified(output_file)