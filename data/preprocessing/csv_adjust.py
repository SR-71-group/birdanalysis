
#
# SCRIPT FOR REMOVING FILES EXTENSION FROM COLUMN WITH FILENAMES
#


import csv
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
input_file = base_dir / 'metadata/sliding_window_annotations.csv'  # file which needed to be adjusted
output_file = base_dir / 'metadata/sl_win_annotations.csv'  # The output CSV file name

# Open the input file for reading and the output file for writing
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    # Create CSV reader and writer objects
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Process each row
    for row in reader:
        # Remove '.wav' from the first column, if it exists
        row[0] = row[0].replace('.wav', '')
        # Write the modified row to the output file
        writer.writerow(row)

print(f"File has been processed and saved as '{output_file}'.")
