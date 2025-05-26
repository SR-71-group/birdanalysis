from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


if __name__ == "__main__":

    origin_csv = "data/annotations_noise.csv"
    target_csv_train = "data/annotations_noise_train_resampled.csv"
    target_csv_test = "data/annotations_noise_test.csv"
    oversampling_amount = 0.2 # 20% of the majority class size
    
    df = pd.read_csv(origin_csv, encoding='utf-8', on_bad_lines='skip')

    X = df[["filename"]]
    y = df["species_code"]

    # Perform stratified train-test split based on the species_code column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Determine the desired number of samples for each class
    majority_class_size = y_train.value_counts().max()
    sampling_strategy = {cls: max(count, int(majority_class_size * oversampling_amount)) for cls, count in y_train.value_counts().items()}

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    print(y_train_res.value_counts())

    # Combine the resampled features and labels into a DataFrame
    train_df_res = pd.concat([X_train_res, y_train_res], axis=1)

    # Save the resampled training set to a new CSV file
    train_df_res.to_csv(target_csv_train, index=False, encoding='utf-8')

    # Save the testing set to a new CSV file
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(target_csv_test, index=False, encoding='utf-8')