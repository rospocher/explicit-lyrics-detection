!pip install fasttext
!pip install tensorflow-addons

from __future__ import print_function

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
import tensorflow_addons as tfa

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report
import pandas as pd
from time import time
import argparse
import os
import numpy as np
import gzip
import fasttext
from numpy import argmax


def get_data(dataset):
    data = pd.read_csv(dataset)
    return data.iloc[:, 0], data.iloc[:, 1]


def writePrediction(output, test, pred):
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')

#PARAM
MAX_NUM_WORDS = 40000
MAX_SEQUENCE_LENGTH = 750
VALIDATION_SPLIT = 0.05
EMBEDDING_DIM = 65

tasks = ["explicit", "strong", "abuse", "sexual", "violence", "discriminatory"]

def getScore(report):
  return report["precision"], report["recall"], report["f1-score"]

input_folder = "/LYRICS/fine-grained/rawdata/cv10/"
output_folder = "/LYRICS/fine-grained/experiments/1DCNN-cv10/"

trainList = ["empty"]
testList= ["empty"]
for i in range(10):
  input_train = os.path.join(input_folder, '{:02d}'.format(i)+"train.csv")
  input_test = os.path.join(input_folder, '{:02d}'.format(i)+"test.csv")
  train = pd.read_csv(input_train)
  test = pd.read_csv(input_test)
  train.columns = ["text", "explicit", "strong", "abuse", "sexual", "violence", "discriminatory"]
  test.columns = ["text", "explicit", "strong", "abuse", "sexual", "violence", "discriminatory"]

  trainList.append(train)
  testList.append(test)

modelfile = "/LYRICS/ft-model.bin"

print("Processing Model: {}".format(modelfile))
assert os.path.exists(modelfile), "model not found"

#print("Loading fastText model")
model=fasttext.load_model(modelfile)


for task in tasks:

  task_metrics = []
  task_predictions= []

  for i in range(1,limit+1):
    trainfile = os.path.join(input_folder, task, '{:02d}'.format(i)+"train.csv")
    print("Processing train: {}".format(trainfile))
    assert os.path.exists(trainfile), "train not found"
    
    testfile = os.path.join(input_folder, task, '{:02d}'.format(i)+"test.csv")
    print("Processing test: {}".format(testfile))
    assert os.path.exists(testfile), "test not found"


    output = os.path.join(output_folder)
    os.makedirs(output, exist_ok=True)

    eval = os.path.join(output, task+"-{:02d}eval.txt".format(i))
    print("Saving eval in: {}".format(eval))

    print("Loading data and models")
    X_train_input, y_train_input = trainList[i]["text"],trainList[i][task]
    X_test_input, y_test_input =  testList[i]["text"],testList[i][task]
    #len(y_train_input)-sum(y_train_input)

    start = time()

    X_train_text = list(X_train_input.iloc[0:])
    X_test_text = list(X_test_input.iloc[0:])
    y_train_label = list(y_train_input.iloc[0:])
    y_test_label = list(y_test_input.iloc[0:])

    texts = list(X_train_text)
    texts.extend(X_test_text)
    labels = list(y_train_label)
    labels.extend(y_test_label)
    labels_index = {0: 0, 1: 1}

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print("Tokenizing time: {}".format(time() - start))
    start = time()

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    data_train = data[:len(y_train_label)]
    data_test = data[len(y_train_label):]

    labels_train = labels[:len(y_train_label)]
    labels_test = labels[len(y_train_label):]

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    #indices = np.arange(data_train.shape[0])
    #np.random.shuffle(indices)
    #data_train = data_train[indices]
    #labels_train = labels_train[indices]
    #num_validation_samples = int(VALIDATION_SPLIT * data_train.shape[0])

    #x_train = data_train[:-num_validation_samples]
    #y_train = labels_train[:-num_validation_samples]
    #x_val = data_train[-num_validation_samples:]
    #y_val = labels_train[-num_validation_samples:]

    x_train = data_train
    y_train = labels_train
    x_val = data_test
    y_val = labels_test

    print('Preparing embedding matrix.')

    # prepare embedding matrix

    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, j in word_index.items():
        if j >= MAX_NUM_WORDS:
            continue
        embedding_vector = model.get_word_vector(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[j] = embedding_vector

    print("Embedding matrix: {}".format(time() - start))
    start = time()

    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print("Embedding layer: {}".format(time() - start))


    #NEW
    start = time()

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    modelCNN = Model(sequence_input, preds)
    modelCNN.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=[F1Score(num_classes=2, average='macro')])

    checkpoint_filepath = output+"/"+task+"-{:02d}".format(i)+'weights.best.h5'
    #checkpoint_filepath = os.path.join(output,task+"-weights.best.h5")
    model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_f1_score',
            mode='max',
            #verbose=1,
            save_best_only=True,
            save_weights_only=True)

    modelCNN.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              validation_data=(x_val, y_val),
              callbacks=[model_checkpoint_callback])

    modelCNN.load_weights(checkpoint_filepath)

    print("Training time: {}".format(time() - start))

    #model.save_model(savecnn)

    start = time()

    predictions = modelCNN.predict(data_test)

    y_pred = argmax(predictions, axis=1)

    print("Prediction Time: {}".format(time() - start))
    start = time()

    report = classification_report(y_test_label, y_pred, digits=3, output_dict=True, target_names=['0', "1"])
    
    bacc = report["accuracy"]
    bMp,bMr,bMf1 = getScore(report["macro avg"])
    bwp,bwr,bwf1 = getScore(report["weighted avg"])
    bp0,br0,bf10 = getScore(report["0"])
    bp1,br1,bf11 = getScore(report["1"])

    scores = {
        'bacc': bacc,
        'bMap': bMp,
        'bMar': bMr,
        'bMaf1': bMf1,
        'bwp': bwp,
        'bwr': bwr,
        'bwf1': bwf1,       
        'bp0': bp0,
        'br0': br0,
        'bf10': bf10,
        'bp1': bp1,
        'br1': br1,
        'bf11': bf11
    }

    task_metrics.append(scores)
    if i==1:
      task_predictions = predictions[:,1]
    else:
      task_predictions = np.append(task_predictions, predictions[:,1], axis=0)

    
  df_metrics=pd.DataFrame(task_metrics)
  df_metrics.index += 1
  df_metrics.to_csv(os.path.join(output,task+"-results.csv"), index=None)
  
  #write predictions
  df_outputs=pd.DataFrame(task_predictions)
  df_outputs.columns=[task]
  df_outputs.to_csv(os.path.join(output,task+"-outputs.csv"), index=None)

  print("###############################################################")
  print("TASK: "+task)
  print(df_metrics)
  print(df_metrics.mean().to_frame().T)
  print(df_metrics.std().to_frame().T)
  print("###############################################################")