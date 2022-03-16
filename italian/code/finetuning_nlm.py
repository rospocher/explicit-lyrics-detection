import torch, os, gzip
import numpy as np
from time import time
from sklearn.metrics import f1_score ,classification_report
from transformers import DataCollatorWithPadding, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_from_disk

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

base_folder = "/lyrics/"
datasets = load_from_disk(base_folder+"datasetNLM/")

whole_train = datasets["train"].train_test_split(0.20, seed=21)

#checkpoint = "Musixmatch/umberto-wikipedia-uncased-v1"
#checkpoint = "Musixmatch/umberto-commoncrawl-cased-v1"
#checkpoint = "bert-base-multilingual-uncased"
#checkpoint = "dbmdz/bert-base-italian-uncased"
#checkpoint = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alberto"
checkpoint = "idb-ita/gilberto-uncased-from-camembert"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")
  #return tokenizer(batch["text"], truncation=True, padding="max_length")

# Encode the input data
dataset = datasets.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset.rename_column_("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

metric_name = "mf1"
batch_size = 12

folder = base_folder + "fine-tuning/"+checkpoint+"/"
os.makedirs(folder, exist_ok=True)

training_args = TrainingArguments(folder+"/training_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    overwrite_output_dir=True,
    save_strategy="epoch",
    metric_for_best_model=metric_name,
    weight_decay=0.01,
    load_best_model_at_end=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def getScore(report):
  return report["precision"], report["recall"], report["f1-score"]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function

    report = classification_report(labels, preds, digits=3, output_dict=True)

    p0,r0,f10 = getScore(report["0"])
    p1,r1,f11 = getScore(report["1"])
    acc = report["accuracy"]
    mp,mr,mf1 = getScore(report["macro avg"])
    wp,wr,wf1 = getScore(report["weighted avg"])

    return {
        'p0': p0,
        'r0': r0,
        'f10': f10,
        'p1': p1,
        'r1': r1,
        'f11': f11,        
        'acc':acc,
        'mp': mp,
        'mr': mr,
        'mf1': mf1,
        'wp': wp,
        'wr': wr,
        'wf1': wf1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=whole_train["train"],
    eval_dataset=whole_train["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(folder)

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    outputs = model(**inputs)
    #print("Logits:", outputs.logits)
    return outputs.logits.argmax(-1).tolist()[0]

def writePrediction(output, test, pred):
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (test, pred), fmt='%i')

print("EVAL")

text_list = datasets["test"]["text"][:]
labels = datasets["test"]["label"][:]

startTime = time()
pred_list = []
start = 0
counter = start
for text in text_list:
  pred_list.append(get_prediction(text))

print("Done Testing time: {}".format(time() - startTime))
writePrediction(folder + "eval.gz",labels,pred_list)

report = classification_report(labels, pred_list, digits=3)
print(report)
with open(folder + "eval.score.txt", "wt") as f:
  f.write(report)