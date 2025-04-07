##!pip install transformers datasets
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from datasets import Dataset
import datasets
import random
import pandas as pd
from IPython.display import display, HTML
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os, argparse
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
task = "sst2"
model_checkpoint = "distilbert-base-uncased"

dataset = pd.read_csv('drugsCom/drugsComTrain_raw.csv', encoding='ISO-8859-1')

dataset = dataset.dropna()
dataset = dataset.drop_duplicates(subset='review')

dataset =dataset[['review','rating','usefulCount']]
def label_rating(rating):
    if 1 <= rating <= 5:
        return 0
    elif 6 <= rating <= 10:
        return 1
dataset['label'] = dataset['rating'].apply(label_rating)
data=dataset[['review','label','usefulCount']]

top_1000_per_label = data.sort_values('usefulCount', ascending=False).groupby('label').head(10000)
label_counts = top_1000_per_label['label'].value_counts()
datas=top_1000_per_label[['review','label']]
datas = data[['review', 'label']]


import numpy as np
texts = np.array(datas['review'])
labels = np.array(datas['label'])
data = Dataset.from_pandas(datas)
data = data.remove_columns(['__index_level_0__'])
datass=data.train_test_split(test_size=0.1)
actual_task = "mnli" if task == "mnli-mm" else task



batch_size = 16
metric = load_metric('accuracy')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True,trust_remote_code=True,ignore_mismatched_sizes=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
sentence_key= ('review')

def preprocess_function(examples):
    return tokenizer(examples[sentence_key], truncation=True)

encoded_dataset = datass.map(preprocess_function, batched=True)


num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,trust_remote_code=True,ignore_mismatched_sizes=True)

print(f"param_amout of model: {sum(param.numel() for param in model.parameters())}")


num_train_epochs = 1
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
output_dir = f"{num_train_epochs}_{model_checkpoint}"
args = TrainingArguments(
    #"test-finance/1",
    output_dir,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print('############## evaluate #################')
res = trainer.evaluate()
print(f"acc before tunning : {res['eval_accuracy']*100:.2f}%")
print('##########################################')



model = model.to(device)
print('############## trainning #################')
trainer.train()
print('##########################################')

print('############## evaluate #################')
res = trainer.evaluate()
print(f"acc after tunning : {res['eval_accuracy']*100:.2f}%")
print('##########################################')



def creat_dataset(df):
    data = Dataset.from_pandas(df)
    data = data.remove_columns(['__index_level_0__'])
    #data=data.train_test_split(test_size=0.1)
    return datass

test = pd.read_csv('drugsCom/drugsComTest_raw.csv', encoding='ISO-8859-1')

test = test.dropna()
test = test.drop_duplicates(subset='review')
test=test[['review','rating']]
def label_rating(rating):
    if 1 <= rating <= 5:
        return 0
    elif 6 <= rating <= 10:
        return 1
# 使用 apply 函数来创建新的 'label' 列
test['label'] = test['rating'].apply(label_rating)
test=test[['review','label']]
testt = creat_dataset(test)
testt = testt.map(preprocess_function)

testt_test = creat_dataset(test[:1000])
testt_test = testt_test.map(preprocess_function)


print('###############################')
#print(trainer.evaluate(testt_test))

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=testt_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print(trainer.evaluate())