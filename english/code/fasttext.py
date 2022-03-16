import fasttext
import numpy as np
from time import time, gmtime, strftime
import argparse
import gzip
import os


def writePrediction(output, test, pred):

    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')


def predict(folder, trainpx, testpx, modelname=""):

    for f_name in os.listdir(folder):
        if f_name.startswith(trainpx):
            trainfile = os.path.join(folder, f_name)
        elif f_name.startswith(testpx):
            testfile = os.path.join(folder, f_name)

    if not modelname:
        print("Training on {}".format(trainfile))
        start = time()

        # best model
        model = fasttext.train_supervised(input=trainfile, lr=0.041897598020537996, epoch=40, wordNgrams=4, dim=65, loss='hs', minn=4, maxn=6, minCount=1)
        # model with no subword
        #model = fasttext.train_supervised(input=trainfile, lr=0.041897598020537996, epoch=40, wordNgrams = 4, dim = 65, loss = 'hs', minn = 0, maxn = 0, minCount = 1)

        model.save_model(os.path.join(folder, "ft-model.bin"))
        print("Training time: {}".format(time() - start))
    else:
        model = fasttext.load_model(modelname)

    start = time()

    if testfile.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(testfile, 'rt') as f:
        test_desc = f.readlines()

    listPred = []
    listLabel = []
    for line in test_desc:
        if line.startswith("__label__1 "):
            desc = line[len("__label__1 "):]
            label = 1
        elif line.startswith("__label__0 "):
            desc = line[len("__label__0 "):]
            label = 0
        elif line.strip():
            print("<EMPTY?")
            print(line)
            print(">")
        else:
            print("<ERROR reading test")
            print(line)
            print(">")

        predLabel = model.predict(desc.rstrip("\n\r"))[0][0];
        if predLabel == "__label__1":
            pred = 1
        elif predLabel == "__label__0":
            pred = 0
        else:
            print("ERROR in prediction")

        listPred.append(pred)
        listLabel.append(label)

    print("Testing time: {}".format(time() - start))
    return listLabel, listPred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastText code for predicting song lyrics')
    parser.add_argument('folder', help='the file containing the datasets in fasttext format')
    parser.add_argument('--cv', help='the number of folders for cross-validation', type=int, default=0)
    parser.add_argument('--model', help='load an existing model instead of training', type=str, default="")
    args = parser.parse_args()

    folder = args.folder
    print("Processing folder: {}".format(folder))
    assert os.path.exists(folder), "input folder not found"

    cv = args.cv

    model = args.model

    final_test = list()
    final_pred = list()

    if cv:
        print("Number of CV folders {}".format(cv))
        for i in range(0, 10):
            i_str = '{:02d}'.format(i)
            print("Start iteration " + i_str+ ": " + strftime("%H:%M:%S", gmtime()))
            temp_test, temp_pred = predict(folder, i_str + "-" + "train-", i_str + "-" + "test-", model)
            print("End iteration " + i_str + ": " + strftime("%H:%M:%S", gmtime()))
            final_test.extend(temp_test)
            final_pred.extend(temp_pred)
    else:
        print("Train and Test")
        final_test, final_pred = predict(folder, "train-", "test-", model)

    output = os.path.join(folder, "eval.gz")
    writePrediction(output, final_test, final_pred)