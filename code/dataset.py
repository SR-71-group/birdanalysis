import pandas as pd
import csv
import os
#2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___sr.wav


def filename_to_annotations(filename):
    filename = filename[:-4]
    annotations = filename.split("_")
    annotations = [an for an in annotations if an != ""]
    annotations[2] = annotations[2][:-3]
    annotations[3] = annotations[3][:-1]
    annotations[2] = annotations[2].split("-")
    annotations[3] = annotations[3].split("-")
    # flatten the list:
    annotations = [item for sublist in annotations for item in (sublist if isinstance(sublist, list) else [sublist])]

    return annotations

def append_rows_tocsv(csv_filepath):
    title = ["time", "loc", "low_freq", "high_freq", "start", "end", "species"]
    audio_filepaths = [file for file in os.listdir("data/dataset/Audiodateien") if file.endswith('.wav')]
    with open(csv_filepath, "w", newline="") as annotations_file:
        wr = csv.writer(annotations_file)
        wr.writerow(title)
        for file in audio_filepaths:
            annotation = filename_to_annotations(file)
            wr.writerow(annotation)
    

if __name__ == "__main__":
    
    append_rows_tocsv("data/annotations.csv")

