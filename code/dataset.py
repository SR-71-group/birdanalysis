import pandas as pd
import csv
import os
from datetime import datetime, timedelta
#2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___sr.wav


def julian_to_datetime(julian_date):
    # Julian day 0 starts on November 24, 4714 BC in the Julian calendar, 
    # which is represented as 24-hour noon, so subtract the half day (0.5)
    julian_start = datetime(4714, 11, 24, 12)  # Julian Day 0
    return julian_start + timedelta(days=julian_date)


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
    annotations = [filename] + annotations

    #converted_datetime = julian_to_datetime(float(annotations[1]))
    #annotations.insert(3, converted_datetime)

    return annotations

def append_rows_tocsv(csv_filepath):
    title = ["filename", "juliandate", "loc", "low_freq", "high_freq", "start", "end", "species"]
    audio_filepaths = [file for file in os.listdir("data/dataset/Audiodateien") if file.endswith('.wav')]
    with open(csv_filepath, "w", newline="") as annotations_file:
        wr = csv.writer(annotations_file)
        wr.writerow(title)
        for file in audio_filepaths:
            annotation = filename_to_annotations(file)
            wr.writerow(annotation)
    

if __name__ == "__main__":
    
    append_rows_tocsv("data/annotations.csv")

