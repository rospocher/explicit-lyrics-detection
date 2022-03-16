import torch, os
from sklearn.metrics import f1_score
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

base_folder = "/"
datasets = load_from_disk(base_folder+"datasets/")
print(datasets)

#checkpoint = "bert-base-uncased"
checkpoint = "distilbert-base-uncased"
#checkpoint = "roberta-base"
#checkpoint = "xlnet-base-cased"
#checkpoint = "microsoft/deberta-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

# Encode the input data
dataset = datasets.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset.rename_column_("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

metric_name = "mf1"
batch_size = 16

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
    remove_unused_columns=False,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    mf1 = f1_score(labels, preds, average='macro')
    wf1 = f1_score(labels, preds, average='weighted')
    return {
        'mf1': mf1,
        'wf1': wf1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(folder)