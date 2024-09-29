import os
from pydub import AudioSegment

class AudioCutter:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def cut_audio(self, file_path, start_ms, end_ms):
        audio = AudioSegment.from_wav(file_path)
        cut_audio = audio[start_ms:end_ms]
        return cut_audio

    def process_audio_cutting(self):
        def parse_filename(filename):
            parts = filename.split('___')
            time_range = parts[2].replace('s', '')
            start_time, end_time = map(float, time_range.split('-'))
            return start_time * 1000, end_time * 1000

        for filename in os.listdir(self.input_directory):
            if filename.endswith(".wav"):
                input_file_path = os.path.join(self.input_directory, filename)
                start_ms, end_ms = parse_filename(filename)
                cut_audio = self.cut_audio(input_file_path, start_ms, end_ms)
                output_file_path = os.path.join(self.output_directory, f"cut_{filename}")
                cut_audio.export(output_file_path, format="wav")
                print(f"Cut file saved as {output_file_path}")
