from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse


def splitTT(dataset, outputfolder, percentage):

    dataset_mod = os.path.basename(dataset)

    output_train = os.path.join(outputfolder, "train-" + dataset_mod)
    output_test = os.path.join(outputfolder, "test-" + dataset_mod)

    df_dataset = pd.read_csv(dataset)

    X = np.asarray(df_dataset.iloc[:, 0])
    y = np.asarray(df_dataset.iloc[:, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=21, stratify=y)

    train_df = pd.DataFrame({df_dataset.columns[0]: X_train, df_dataset.columns[1]: y_train})
    test_df = pd.DataFrame({df_dataset.columns[0]: X_test, df_dataset.columns[1]: y_test})

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)


def splitTVT(dataset, outputfolder, percentageV, percentageT):

    dataset_mod = os.path.basename(dataset)

    output_train = os.path.join(outputfolder, "train-" + dataset_mod)
    output_val = os.path.join(outputfolder, "val-" + dataset_mod)
    output_test = os.path.join(outputfolder, "test-" + dataset_mod)

    df_dataset = pd.read_csv(dataset)

    X = np.asarray(df_dataset.iloc[:, 0])
    y = np.asarray(df_dataset.iloc[:, 1])

    X_trainV, X_test, y_trainV, y_test = train_test_split(X, y, test_size=percentageT, random_state=42, stratify=y) #42, 21, 10
    X_train, X_val, y_train, y_val = train_test_split(X_trainV, y_trainV, test_size=percentageV/(1-percentageT), random_state=42, stratify=y_trainV)

    train_df = pd.DataFrame({df_dataset.columns[0]: X_train, df_dataset.columns[1]: y_train})
    val_df = pd.DataFrame({df_dataset.columns[0]: X_val, df_dataset.columns[1]: y_val})
    test_df = pd.DataFrame({df_dataset.columns[0]: X_test, df_dataset.columns[1]: y_test})

    train_df.to_csv(output_train, index=False)
    val_df.to_csv(output_val, index=False)
    test_df.to_csv(output_test, index=False)

    print("Train - Validation - Test sizes: {} - {} - {}".format(len(y_train), len(y_val), len(y_test)))


def crossVal(dataset, outputfolder, cv):

    dataset_mod = os.path.basename(dataset)

    df_dataset = pd.read_csv(dataset)
    X = np.asarray(df_dataset.iloc[:, 0])
    y = np.asarray(df_dataset.iloc[:, 1])

    # splitting
    skf = StratifiedKFold(n_splits=cv)
    i = 0

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        i_str = '{:02d}'.format(i)

        train_df = pd.DataFrame({df_dataset.columns[0]: X_train, df_dataset.columns[1]: y_train})
        test_df = pd.DataFrame({df_dataset.columns[0]: X_test, df_dataset.columns[1]: y_test})

        output_train = os.path.join(outputfolder, i_str + "-" + "train-" + dataset_mod)
        output_test = os.path.join(outputfolder, i_str + "-" + "test-" + dataset_mod)

        train_df.to_csv(output_train, index=False)
        test_df.to_csv(output_test, index=False)

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset for train, development, and test')
    parser.add_argument('dataset', help='file with ID and annotations')
    parser.add_argument('outputfolder', help='splitted file with ID and annotations')
    parser.add_argument('--cv', help='if speficied means cross validation in 10 folder, if not means 80/20 train-test', type=int)
    parser.add_argument('--val', help='if speficied means train - validation -test split', type=int)

    args = parser.parse_args()
    dataset = args.dataset
    print("Processing dataset {}".format(dataset))
    assert os.path.exists(dataset), "dataset not found"

    outputfolder = args.outputfolder
    print("Output directory: {}".format(outputfolder))
    os.makedirs(outputfolder, exist_ok=True)

    cv = args.cv
    val = args.val
    percentage = 0.20  # test percentage

    if cv:
        crossVal(dataset, outputfolder, cv)
    elif val:
        print("Train - Validation - Test split: {} - {} - {}".format(1 - 2*percentage, percentage, percentage))
        splitTVT(dataset, outputfolder, percentage, percentage)
    else:
        print("Train - Test split: {} - {}".format(1 - percentage, percentage))
        splitTT(dataset, outputfolder, percentage)
