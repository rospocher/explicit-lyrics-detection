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
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
import tensorflow_addons as tfa
import pandas as pd
from time import time
import argparse
import os
import numpy as np
import gzip
import fasttext
from numpy import argmax
from tensorflow.python.keras.metrics import CategoricalAccuracy
from tensorflow_addons.metrics import F1Score


def get_data(dataset):
    data = pd.read_csv(dataset)
    return data.iloc[:, 0], data.iloc[:, 1]


def writePrediction(output, test, pred):
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')

base_folder = "/lyrics/"

trainfile = base_folder+"train.csv.gz"
print("Processing train: {}".format(trainfile))
assert os.path.exists(trainfile), "train not found"

testfile = base_folder+"test.csv.gz"
print("Processing test: {}".format(testfile))
assert os.path.exists(testfile), "test not found"

modelfile = "cc.it.300.bin"
print("Processing FastText Model: {}".format(modelfile))
#assert os.path.exists(modelfile), "model not found"

output = base_folder +"1DCNN/"
#output = base_folder +"{}/".format(sizeTrain)
os.makedirs(output, exist_ok=True)

eval = output+"eval.gz"
print("Saving eval in: {}".format(eval))

#print("Loading fastText model")
model = fasttext.load_model(modelfile)

print("Loading data and models")
X_train_input, y_train_input = get_data(trainfile)
X_test_input, y_test_input = get_data(testfile)

#PARAM
MAX_NUM_WORDS = 40000
MAX_SEQUENCE_LENGTH = 500
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300

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
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_train = data_train[indices]
labels_train = labels_train[indices]
num_validation_samples = int(VALIDATION_SPLIT * data_train.shape[0])

x_train = data_train[:-num_validation_samples]
y_train = labels_train[:-num_validation_samples]
x_val = data_train[-num_validation_samples:]
y_val = labels_train[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = model.get_word_vector(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Embedding matrix: {}".format(time() - start))
start = time()

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print("Embedding layer: {}".format(time() - start))

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

#saving weights of the best model
checkpoint_filepath = eval+'weights.{epoch:02d}-{val_f1_score:.3f}.h5'
model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)

modelCNN.fit(x_train, y_train,
           batch_size=128,
           epochs=5,
           validation_data=(x_val, y_val),
           callbacks=[model_checkpoint_callback])

print("Training time: {}".format(time() - start))

start = time()

y_pred = argmax(modelCNN.predict(data_test), axis=1)

print("Prediction Time: {}".format(time() - start))
start = time()

output = os.path.join(eval)
writePrediction(output, y_test_label, y_pred)

report = classification_report(y_test_label, y_pred, digits=3)
print(report)
with open(output + ".score.txt", "wt") as f:
    f.write(report)