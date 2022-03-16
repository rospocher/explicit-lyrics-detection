import fasttext
import numpy as np
from time import time, gmtime, strftime
import argparse
import gzip
import os


def writePrediction(output, test, pred):

    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')


def predict(trainfile, testfile, modelname="", save=""):

    if not modelname:
        print("Training on {}".format(trainfile))
        start = time()

        model = fasttext.train_supervised(input=trainfile, lr=0.041897598020537996, epoch=40, wordNgrams=4, dim=300,
                                          loss='hs', minn=4, maxn=6, minCount=1, pretrainedVectors='fastText/model/cc.it.300.vec')

        if save:
            model.save_model(save)
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
    parser.add_argument('train', help='the file containing the traininig')
    parser.add_argument('test', help='the file containing the test')
    parser.add_argument('eval', help='the file where to put the eval data GZIPPED')
    parser.add_argument('--save', help='save the model', type=str, default="")
    parser.add_argument('--model', help='load an existing model instead of training', type=str, default="")
    args = parser.parse_args()

    trainfile = args.train
    print("Processing train: {}".format(trainfile))
    assert os.path.exists(trainfile), "train not found"

    testfile = args.test
    print("Processing test: {}".format(testfile))
    assert os.path.exists(testfile), "train not found"

    eval = args.eval
    print("Saving in eval: {}".format(eval))

    model = args.model
    save = args.save

    final_test = list()
    final_pred = list()


    print("Train and Test")
    final_test, final_pred = predict(trainfile,testfile, model, save)

    writePrediction(eval, final_test, final_pred)