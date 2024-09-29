import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa

# A function to open the one audio file, manipulate it and send the final file.
# It need a file path, desired value for cutting the audio file (start_cut, end_cut), 
#   sample rate of the audio signal, offset and the duration as an input values. 
# It provides the trimmed audio and sample rate in the variable called trimmed_audio and 
# sr respectively. 
#def open_audio_file(path: str, start_cut: float, end_cut: float, sample_rate=48000, offset=0.0, duration=None, target_length=None):

def open_audio_file(path: str, start_cut: float, end_cut: float, sample_rate=48000, offset=0.0, duration=None):
    a, sr = librosa.load(path, sr=sample_rate, mono=True, offset=offset, duration=duration)
    total_duration = len(a)/sr
    
    if start_cut + end_cut > total_duration:
        raise ValueError(f"Invalid trim times: start_cut ({start_cut}s) and end_cut ({end_cut}s) result in an empty audio.")
    
    start_sample = int(start_cut * sr)
    end_sample = int(end_cut * sr)
    trimmed_audio = a[start_sample:len(a)-end_sample]
    
#    if target_length is not None:
        # current_length = len(trimmed_audio)
        # if current_length < target_length:
        #     total_padding = target_length - current_length
        #     pad_start = total_padding // 2
        #     pad_end = total_padding - pad_start
        #     trimmed_audio = np.pad(trimmed_audio, (pad_start, pad_end))
        # elif current_length > target_length:
        #     trimmed_audio = trimmed_audio[:target_length]
    
    return trimmed_audio, sr

# A function to access the folder in which the audio files has been stored. 
# It takes a folder_path, desired value for cutting the audio file (start_cut, end_cut),
#   and a sample rate of the audio signal as an input values. 
# Additionally, a for loop has been created to load the all audio files from that folder. 
# To load the audio file, it uses the load function that has been created above. 
# As an output, an 3 dimensional array (df) has been created in which it has the trimmed_data
#   information in the form of NumPy array corresponds to their names and sample rate. 

    # uncomment and comment out the process_folder function if required to use the zero padding function. 
    
#def process_folder(folder_path: np.ndarray, start_cut: float, end_cut: float, sample_rate=48000, target_length=None):
def process_folder(folder_path: np.ndarray, start_cut: float, end_cut: float, sample_rate=48000):
    audio_extensions = ('.wav', '.mp3')  # Add other audio formats if needed
    dataset = []
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(audio_extensions):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            #print(i)
            # Load the audio file
                # uncomment and commnet out if required to use the zero padding function
            #trimmed_data, sr = open_audio_file(file_path, start_cut, end_cut, sample_rate=sample_rate, target_length=target_length)
            trimmed_data, sr = open_audio_file(file_path, start_cut, end_cut, sample_rate=sample_rate)
            dataset.append({
                        'filename': filename,
                        'data': np.array(trimmed_data),
                        'sample_rate': sr,
                    })
    df = pd.DataFrame(dataset)
    if df.empty:
        print('No2')
    return df

# A function has been created to plot the signal using matplotlib. 

def plot_signal(AD, sr):
    #print(AD.shape)
    if len(AD.shape) >1:
        AD = np.ravel(AD)
    
    time = np.linspace(0, len(AD)/sr, num=len(AD))
    plt.figure(figsize=(10, 4))
    plt.plot(time, AD)
    plt.show()
    
# A function has been created to generate the spectrogram values.
# The values has been stored in the form of *.npy type.     

def create_spectrogram(AD, sr, file_name=None, save_path=None):
    S = librosa.stft(AD)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)     # Might go into the learning model
    
    base_filename = os.path.splitext(file_name)[0]
    save_file = os.path.join(save_path, f"{base_filename}.npy")
    np.save(save_file, S_dB)
        
    #plt.figure(figsize=(10,6))
    #librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="log")
    #plt.show()