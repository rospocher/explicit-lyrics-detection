import pandas as pd, os
import numpy as np
from datasets import DatasetDict, Dataset, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split

base_folder = "/lyrics/"
train = pd.read_csv(base_folder + "rawdata/train.csv.gz")
test = pd.read_csv(base_folder + "rawdata/test.csv.gz")

train.columns = ["text", "label"]
test.columns = ["text", "label"]

dataset_train = Dataset.from_pandas(train, features=Features(
    {"label": ClassLabel(names=['not explicit', 'explicit']), "text": Value(dtype='string')}))
dataset_test = Dataset.from_pandas(test, features=Features(
    {"label": ClassLabel(names=['not explicit', 'explicit']), "text": Value(dtype='string')}))

datasets = DatasetDict([("train", dataset_train), ("test", dataset_test)])
print(datasets)

datasets.save_to_disk(base_folder + "datasetNLM/")