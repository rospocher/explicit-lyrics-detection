import numpy as np
import pandas as pd
import os
import argparse
import gzip


def get_data(dataset):
    data = pd.read_csv(dataset)
    return data.iloc[:, 1], data.iloc[:, 0]

def writePrediction(output, test, pred):
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')

def badWords(text,bad_words):
  words = set(text.lower().split(" "))
  offensive = set(bad_words)
  return len(words.intersection(offensive)) > 0

def predictMajority(testfile):

    X_test, y_test = get_data(testfile)
    y_pred = [0]*len(y_test)

    return y_test, y_pred

def predictDictionary(testfile, badfile):

    X_test, y_test = get_data(testfile)
    with open(badfile, "r") as f:
        badwords = f.read().splitlines()

    y_pred = [badWords(text,badwords) for text in X_test]
    return y_test, y_pred




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('test', help='the file containing the test')
    parser.add_argument('bad', help='the file containing the badwords')
    parser.add_argument('eval', help='the file where to put the eval data FOLDER')
    args = parser.parse_args()

    testfile = args.test
    print("Processing test: {}".format(testfile))
    assert os.path.exists(testfile), "train not found"

    badfile = args.bad
    print("Processing badfile: {}".format(badfile))
    assert os.path.exists(badfile), "train not found"

    eval = args.eval
    print("Saving in eval: {}".format(eval))

    print("Majority")
    final_test = list()
    final_pred = list()
    final_test, final_pred = predictMajority(testfile)
    writePrediction(eval+"/eval-maj.gz", final_test, final_pred)

    print("Dictionary")
    final_test = list()
    final_pred = list()
    final_test, final_pred = predictDictionary(testfile,badfile)
    writePrediction(eval+"/eval-dic.gz", final_test, final_pred)