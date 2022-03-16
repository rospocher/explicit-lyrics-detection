from time import gmtime, strftime
import numpy as np
import pandas as pd
import os
import argparse
import gzip
import joblib

def get_data(dataset):
    data = pd.read_csv(dataset)
    return data.iloc[:, 1], data.iloc[:, 0]


def writePrediction(output, test, pred):

    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')


def predict(folder, trainpx, testpx):

    print("Start Prediction: " + strftime("%H:%M:%S", gmtime()))

    for f_name in os.listdir(folder):
        if f_name.startswith(testpx):
            testfile = os.path.join(folder, f_name)

    X_test, y_test = get_data(testfile)

    print("  Loading Model: " + strftime("%H:%M:%S", gmtime()))

    WASABI_PATH = "/data/mrcexp/lyrics/code/wasabi/"
    idf_vectorizer = joblib.load(WASABI_PATH + 'tfidf_vectorizer_explicitness.jl')
    clf = joblib.load(WASABI_PATH + 'logistic_regression_explicitness.jl')

    print("  End Loading Model: " + strftime("%H:%M:%S", gmtime()))
    X_test_vec = idf_vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    print("End Prediction: " + strftime("%H:%M:%S", gmtime()))

    return y_test, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction loading the TF-IDF vectorization and classifier developed for WASABI')
    parser.add_argument('folder', help='the file containing the datasets in a CSV file')
    parser.add_argument('--cv', help='the number of folders for cross-validation', type=int, default=0)

    args = parser.parse_args()

    folder = args.folder
    print("Processing folder: {}".format(folder))
    assert os.path.exists(folder), "input folder not found"

    cv = args.cv

    final_test = list()
    final_pred = list()

    if cv:
        print("Number of CV folders {}".format(cv))
        for i in range(0, 10):
            i_str = '{:02d}'.format(i)
            print("Start iteration " + i_str+ ": " + strftime("%H:%M:%S", gmtime()))
            temp_test, temp_pred = predict(folder, i_str + "-" + "train-", i_str + "-" + "test-")
            print("End iteration " + i_str + ": " + strftime("%H:%M:%S", gmtime()))
            final_test.extend(temp_test)
            final_pred.extend(temp_pred)
    else:
        print("Train and Test".format(cv))
        final_test, final_pred = predict(folder, "train-", "test-")

    output = os.path.join(folder, "eval.gz")
    writePrediction(output, final_test, final_pred)