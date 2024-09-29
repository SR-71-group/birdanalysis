from audio import process_folder, create_spectrogram, plot_signal


def main():
    
    
# Different paths to different files
    folder_path = '/home/sarkar/Desktop/sci_camp/birdnet_mini/real_data'
    target_length = 150*48000
    save_path = '/home/sarkar/Desktop/sci_camp/birdnet_mini/data/preprocessed'
    
        # Uncomment below processdata and target_length if required to use the zero padding and comment out the another processdata.
    #processdata = process_folder(folder_path=folder_path, start_cut=0.0, end_cut=0.0, target_length=target_length)
    processdata = process_folder(folder_path=folder_path, start_cut=8.0, end_cut=8.0)
    
    print('processdata:', processdata)
    if not processdata.empty:
        for i , row in processdata.iterrows():
            FN = row['filename']
            AD = row['data']
            SR = row['sample_rate']
            #print("before plotting")
            #AD, sr = open_audio_file(FN, SR)
            #print('AD', FN)
            #plot_signal(AD, SR)
            create_spectrogram(AD, SR, file_name=FN, save_path=save_path)
            
    else:
        print("No")

if __name__=="__main__":
    main()