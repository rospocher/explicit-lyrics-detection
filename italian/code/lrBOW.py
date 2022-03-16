from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from time import time, gmtime, strftime
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


def predict(trainfile, testfile):

    print("Start Prediction: " + strftime("%H:%M:%S", gmtime()))

    X_train, y_train = get_data(trainfile)
    X_test, y_test = get_data(testfile)

    print("  Training Model: " + strftime("%H:%M:%S", gmtime()))
    start = time()
    idf_vectorizer = TfidfVectorizer(analyzer='word', binary=True, decode_error='strict',
                                     encoding='utf-8', lowercase=True, max_df=0.95,
                                     max_features=100000, min_df=2, ngram_range=(1, 2), norm='l2',
                                     preprocessor=None, smooth_idf=True, stop_words=None, strip_accents=None, sublinear_tf=False,
                                     token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
                                     vocabulary=None)
    X_train_vec = idf_vectorizer.fit_transform(X_train)

    print("  after vectorization: " + strftime("%H:%M:%S", gmtime()))

    clf = LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                             fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                             max_iter=2000, n_jobs=None, penalty='l2',
                             random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                             warm_start=False).fit(X_train_vec, y_train)

    print("  End Training Model: " + strftime("%H:%M:%S", gmtime()))
    print("  Training time: {}".format(time() - start))
    start = time()
    X_test_vec = idf_vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    print("  Testing time: {}".format(time() - start))
    print("End Prediction: " + strftime("%H:%M:%S", gmtime()))

    return y_test, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression with BOW TF-IDF vectorization')
    parser.add_argument('train', help='the file containing the traininig')
    parser.add_argument('test', help='the file containing the test')
    parser.add_argument('eval', help='the file where to put the eval data GZIPPED')
    args = parser.parse_args()

    trainfile = args.train
    print("Processing train: {}".format(trainfile))
    assert os.path.exists(trainfile), "train not found"

    testfile = args.test
    print("Processing test: {}".format(testfile))
    assert os.path.exists(testfile), "train not found"

    eval = args.eval
    print("Saving in eval: {}".format(eval))

    final_test = list()
    final_pred = list()

    print("Train and Test")
    final_test, final_pred = predict(trainfile, testfile)

    writePrediction(eval, final_test, final_pred)