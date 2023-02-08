!nvidia-smi
!pip install transformers datasets pandas torch

import transformers
import datasets
from datasets import DatasetDict, Dataset, Features, ClassLabel, Value

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)
import pandas as pd
from sklearn.metrics import f1_score ,classification_report

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    #device = torch.device("cpu")
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print(f"Running on transformers v{transformers.__version__} and datasets v{datasets.__version__}")

base_folder = "LYRICS/fine-grained/"
input_folder = base_folder +"rawdata/cv10/"
model_ckpt = "distilbert-base-uncased"

metric_name = "Maf1"
batch_size = 16
threshold = 0.4

folder = base_folder + "experiments/"+model_ckpt+"-cv10-"+str(threshold)+"/"
os.makedirs(folder, exist_ok=True)

num_labels=6

datasets = []
for i in range(10):
  input_train = os.path.join(input_folder, '{:02d}'.format(i)+"train.csv")
  input_test = os.path.join(input_folder, '{:02d}'.format(i)+"test.csv")
  train = pd.read_csv(input_train)
  test = pd.read_csv(input_test)

  train.columns = ["text", "explicit", "strong", "abuse", "sexual", "violence", "discriminatory"]
  test.columns = ["text", "explicit", "strong", "abuse", "sexual", "violence", "discriminatory"]

  dataset_train = Dataset.from_pandas(train)
  dataset_test = Dataset.from_pandas(test)

  dataset = DatasetDict([("train", dataset_train), ("test", dataset_test)])
  print(dataset)
  datasets.append(dataset)

for i in range(10):
  # create labels column
  cols = datasets[i]["train"].column_names
  datasets[i] = datasets[i].map(lambda x : {"labels": [x[c] for c in cols if c != "text"]})

tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="multi_label_classification")
def tokenize_and_encode(examples):
  return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
  
datasets_enc = []
for i in range(10):  
  datasets[i]
  cols = datasets[i]["train"].column_names
  cols.remove("labels")
  ds_enc = datasets[i].map(tokenize_and_encode, batched=True, remove_columns=cols)

  # cast label IDs to floats
  ds_enc.set_format("torch")
  ds_enc = (ds_enc
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))
  datasets_enc.append(ds_enc)

training_args = TrainingArguments(folder+"/training_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    overwrite_output_dir=True,
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model=metric_name,
    weight_decay=0.01,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

def getScore(report):
  return report["precision"], report["recall"], report["f1-score"]


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_metrics(pred):
    
    logits, labels = pred
    preds = np.where(sigmoid(logits)>threshold,1.,0.)

    #print binary
    print(classification_report(labels[:,0], preds[:,0], digits=3, output_dict=False, target_names=['non explicit', "explicit"]))
    #print multi
    print(classification_report(labels[:,1:], preds[:,1:], digits=3, output_dict=False, target_names=['strong', 'abuse', 'sexual', 'violence', 'discriminatory']))

    reportExp = classification_report(labels[:,0], preds[:,0], digits=3, output_dict=True, target_names=['non explicit', "explicit"])
    report = classification_report(labels[:,1:], preds[:,1:], digits=3, output_dict=True, target_names=['strong', 'abuse', 'sexual', 'violence', 'discriminatory'])

    p0,r0,f10 = getScore(report["strong"])
    p1,r1,f11 = getScore(report["abuse"])
    p2,r2,f12 = getScore(report["sexual"])
    p3,r3,f13 = getScore(report["violence"])
    p4,r4,f14 = getScore(report["discriminatory"])
    mp,mr,mf1 = getScore(report["micro avg"])
    Mp,Mr,Mf1 = getScore(report["macro avg"])
    wp,wr,wf1 = getScore(report["weighted avg"])
    sp,sr,sf1 = getScore(report["samples avg"])

    bacc = reportExp["accuracy"]
    bMp,bMr,bMf1 = getScore(reportExp["macro avg"])
    bwp,bwr,bwf1 = getScore(reportExp["weighted avg"])
    bp0,br0,bf10 = getScore(reportExp["non explicit"])
    bp1,br1,bf11 = getScore(reportExp["explicit"])

    return ({
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
        'bf11': bf11,        
        'mp': mp,
        'mr': mr,
        'mf1': mf1,
        'Map': Mp,
        'Mar': Mr,
        'Maf1': Mf1,
        'wp': wp,
        'wr': wr,
        'wf1': wf1,
        'sp': sp,
        'sr': sr,
        'sf1': sf1,
        'p0': p0,
        'r0': r0,
        'f10': f10,
        'p1': p1,
        'r1': r1,
        'f11': f11,        
        'p2': p2,
        'r2': r2,
        'f12': f12,        
        'p3': p3,
        'r3': r3,
        'f13': f13,        
        'p4': p4,
        'r4': r4,
        'f14': f14,                        
    })


metrics = []
for i in range(10):  
  trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, problem_type="multi_label_classification"),
    args=training_args,
    train_dataset=datasets_enc[i]["train"],
    eval_dataset=datasets_enc[i]["test"],
    compute_metrics=compute_metrics,
  )
  trainer.train()
  output = trainer.predict(datasets_enc[i]["test"])
  if i==0:
    outputs = np.where(sigmoid(output.predictions)>threshold,1.,0.).astype(int)
  else:
    outputs = np.append(outputs, np.where(sigmoid(output.predictions)>threshold,1.,0.).astype(int), axis=0)
  metrics.append(output.metrics)

trainer.save_model(folder+"/bestmodel")


df_outputs=pd.DataFrame(outputs)
df_outputs.columns=['explicit','strong', 'abuse', 'sexual', 'violence', 'discriminatory']
df_outputs.to_csv(folder+"outputs.csv", index=None)

df_metrics=pd.DataFrame(metrics)
df_metrics.index += 1
df_metrics.to_csv(folder+"results.csv", index=None)

df_metrics

print(df_metrics.mean())
print(df_metrics.std())
print(df_metrics.describe())