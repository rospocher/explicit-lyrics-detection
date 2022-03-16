import gzip, os
import numpy as np
from datasets import DatasetDict, Dataset, load_from_disk, load_metric
from time import time
from datasets import load_dataset
import torch
from transformers import DataCollatorWithPadding, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report


def writePrediction(output, test, pred):
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')


def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
    outputs = model(**inputs)
    #print("Logits:", outputs.logits)
    return outputs.logits.argmax(-1).tolist()[0]

# If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        # device = torch.device("cpu")
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

base_folder = "/"

#checkpoint = "bert-base-uncased"
checkpoint = "distilbert-base-uncased"
#checkpoint = "roberta-base"
#checkpoint = "xlnet-base-cased"
#checkpoint = "microsoft/deberta-base"

folder = base_folder + "fine-tuning/"+checkpoint

datasets = load_from_disk(base_folder+"datasets/")
print(datasets)

save_folder = base_folder + "fine-tuning-eval/"+checkpoint+"/"
os.makedirs(save_folder, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(folder)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text_list = datasets["test"]["text"][:]
labels = datasets["test"]["label"][:]

startTime = time()
pred_list = []
start = 0
counter = start
for text in text_list:
    pred_list.append(get_prediction(text))
    counter += 1
    if counter % 10000 == 0:
        print("Done {:06d}".format(counter))

print("Done {} Testing time: {}".format(size, time() - startTime))
writePrediction(save_folder + "eval.gz",labels,pred_list)

report = classification_report(labels, pred_list, digits=3)
print(report)
with open(save_folder + "eval.score.txt".format(size), "wt") as f:
    f.write(report)    