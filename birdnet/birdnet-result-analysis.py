'''
Mapping based on:
same name
if birdnet detected two species, but one was correct, second species omitted
birdnet classification with highest confidence is chosen for each file
if birdnet detected species that is not in the true labels, birdnet detection = other species


birdnetsettings:
minimum confidence: 0.5, sensitivity: 1, overlap: 0
'''
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import seaborn as sns


mapping = { # for species that are named differently
    "Common Chaffinch": "Chaffinch",
    "Eurasian Blackbird": "Blackbird",
    "Eurasian Skylark": "Skylark",
    "Eurasian Wigeon": "Wigeon",
    "Eurasian Moorhen": "Moorhen",
    "European Robin": "Robin",
    "Eurasian Bullfinch": "Bullfinch",
    "Eurasian Coot": "Coot",
    "Gray Wagtail": "Grey Wagtail",
    "Gray Heron": "Grey Heron",
    "Black-crowned Night-Heron": "night heron",
    "Greater White-fronted Goose": "White-fronted Goose",
    "Graylag Goose": "Greylag Goose",
    "Common Goldeneye": "Goldeneye",
    "Green-winged Teal": "Teal",
    "Eurasian Eagle-Owl": "Eagle Owl",
    "Northern Lapwing": "Lapwig",
    "Masked Lapwing": "Lapwig",

}


def merge_csv_files(directory):
    # List to hold DataFrames
    df_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

def create_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)

    unique_labels = sorted(set(labels) | set(preds))

    # Create a DataFrame for better readability (optional)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=[f"{label}" for label in unique_labels])

    # Print the confusion matrix
    print(cm_df)

    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(16,12))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("birnet_conf_mat.png")
    plt.show()

def calc_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)

    # Calculate F1-Score (average='macro' for multiclass classification)
    f1 = f1_score(labels, preds, average='macro')  # macro gives equal weight to all classes
    f1_mic = f1_score(labels, preds, average='micro')

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score (Macro): {f1}")
    print(f"F1-Score (Micro): {f1_mic}")

if __name__ == "__main__":

    otherspecies = open("data/other-species-birdnet.txt")
    otherspecies = otherspecies.read()

    df_labels = pd.read_csv("data/annotations.csv", encoding='utf-8', on_bad_lines='skip')

    df_results = merge_csv_files("data/birdnet-output")

    df_results["filename"] = df_results["File"].apply(lambda x: os.path.basename(x)[:-4])

    merged_df = df_labels.merge(df_results, on='filename', how='left')
    columns_to_convert = merged_df.select_dtypes(include=['float', 'int']).columns
    merged_df[columns_to_convert] = merged_df[columns_to_convert].astype('object')
    merged_df.fillna("nopred", inplace=True)
    print("Column names of merged DataFrame:", merged_df.columns.tolist())

    columnstokeep = ["filename", "Scientific name", "Common name", "Confidence", "species"]
    df_filtered = merged_df.copy()
    df_filtered = df_filtered[columnstokeep]

    df_filtered.rename(columns={"Scientific name": "scient_name_birdnet",
                                "Common name": "common_name_birdnet", 
                                "species": "true_label"}, 
                                inplace=True)
    
    # Group by 'filename' and aggregate 'common_name_birdnet' by concatenating unique values
    def unique_concat(series):
        return '/'.join(series.unique())
    
    df_filtered["common_name_birdnet"] = df_filtered["common_name_birdnet"].apply(lambda x: x if x not in otherspecies else "otherspecies")
    df_filtered["common_name_birdnet"] = df_filtered["common_name_birdnet"].apply(lambda x: x if x not in mapping.keys() else mapping[x])
    
    df_filtered["Confidence"] = df_filtered["Confidence"].astype(str)

    idx = df_filtered.groupby('filename')['Confidence'].idxmax()
    # Filter the DataFrame to keep only the rows with the highest Confidence
    df_grouped = df_filtered.loc[idx]

    df_grouped["true_label"] = df_grouped["true_label"].apply(lambda x: x.split("/")[0])
    label_list = np.array(df_grouped["true_label"].tolist())
    pred_list = np.array(df_grouped["common_name_birdnet"].tolist())

    create_confusion_matrix(label_list, pred_list)
    calc_metrics(label_list, pred_list)
    
    #print(df_grouped[["common_name_birdnet"]])
    
    #unique_scientific = sorted(df_grouped["scient_name_birdnet"].unique())
    #unique_common = sorted(df_grouped["common_name_birdnet"].unique())
    #unique_label = sorted(df_grouped["true_label"].unique())
    #unique_conf = df_grouped["Confidence"].unique()
    #unique_label = [label.split("/")[0] for label in unique_label]

        #df_grouped = df_filtered.groupby('filename').agg({
    #    'scient_name_birdnet': 'first',  # Assuming you want to keep the first occurrence
    #    'common_name_birdnet': unique_concat,
    #    'true_label': 'first',  # Assuming you want to keep the first occurrence
    #}).reset_index()

    #for i in range(len(label_list)):
    #    if label_list[i] in pred_list[i].split("/"):
    #        pred_list[i] = label_list[i]
    #        #print("correct prediction")
    #    else:
    #        pred_list[i] = "otherspecies" # this is not ideal!




    


