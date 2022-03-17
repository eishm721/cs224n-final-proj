import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, BertTokenizer
import json
from transformers import TrainerCallback, EarlyStoppingCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
# df = pd.read_csv('https://jacobdanovitch.blob.core.windows.net/datasets/twtc.csv')
# df.head()

# df = df.drop(['name', 'key_mlbam', 'key_fangraphs', 'age', 'year', 'primary_position', 'eta', 'Arm', 'report', 'Changeup', 'Control', 'Curveball', 'Cutter', 'Fastball', 'Field', 'Hit', 'Power', 'Run', 'Slider', 'Splitter', 'source', 'birthdate', 'mlb_played_first', 'debut_age', 'report'], axis=1)
# df = df.drop(df[df['label'] == -1].index)

train = pd.read_csv('/home/asingh11/224Nfinproj/data/train_shuffle_sentences.csv')
test = pd.read_csv('/home/asingh11/224Nfinproj/data/test.csv') 

# train, test = train_test_split(df, test_size=0.2)

np.savetxt('./preds.csv', test['label'], delimiter=',', header = 'label')

custom_dataset_train = Dataset.from_pandas(train)
custom_dataset_test = Dataset.from_pandas(test)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train = custom_dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding=True), batched=True)
train = train.remove_columns(["text", "__index_level_0__"])
test = custom_dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding=True), batched=True)
test = test.remove_columns(["text", "__index_level_0__"])

# wiki = "amanm27/bert-base-uncased-wiki"
# sports = "amanm27/bert-base-uncased-sports"
# scouting = "amanm27/bert-base-uncased-scouting"
# wiki_sports = "amanm27/bert-base-uncased-wiki-sports"
# wiki_scouting = "amanm27/bert-base-uncased-wiki-scouting"
# sports_scouting = "amanm27/bert-base-uncased-sports-scouting"
# wiki_sports_scouting = "amanm27/bert-base-uncased-wiki-sports-scouting"

# model_wiki = BertForSequenceClassification.from_pretrained(wiki, num_labels=2)
# model_sports = BertForSequenceClassification.from_pretrained(sports, num_labels=2)
# model_scouting = BertForSequenceClassification.from_pretrained(scouting, num_labels=2)
# model_wiki_sports = BertForSequenceClassification.from_pretrained(wiki_sports, num_labels=2)
# model_wiki_scouting = BertForSequenceClassification.from_pretrained(wiki_scouting, num_labels=2)
# model_sports_scouting = BertForSequenceClassification.from_pretrained(sports_scouting, num_labels=2)
# model_wiki_sports_scouting = BertForSequenceClassification.from_pretrained(wiki_sports_scouting, num_labels=2) 


pretrains = ["amanm27/bert-base-uncased-wiki", "amanm27/bert-base-uncased-sports", "amanm27/bert-base-uncased-scouting", "amanm27/bert-base-uncased-wiki-sports", "amanm27/bert-base-uncased-wiki-scouting", "amanm27/bert-base-uncased-sports-scouting", "amanm27/bert-base-uncased-wiki-sports-scouting"]
for pre in pretrains:
    preds_csv = pd.read_csv('./preds.csv')
    model = BertForSequenceClassification.from_pretrained(pre, num_labels=2) 
    print("loaded model ...")
    pth = pre.split("/")[1]
    if not os.path.exists(pth):
        os.makedirs(pth)
    arguments = TrainingArguments(
        output_dir=pth,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )

    def compute_metrics(eval_pred):
        # labels = pred.label_ids
        # preds = pred.predictions.argmax(-1)
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1) 
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=test, # change to test when you do your final evaluation!
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    class LoggingCallback(TrainerCallback):
        def __init__(self, log_path):
            self.log_path = log_path
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(logs) + "\n")

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback(pth+"/log.jsonl"))

    trainer.train()

    #Evaluation
    predictions = trainer.predict(test)

    preds = np.argmax(predictions.predictions, axis=-1)
    print(preds.shape)
    print(preds)

    preds_csv.insert(loc=len(preds_csv.columns)-1, column=pth, value=preds.tolist())
    preds_csv.to_csv("./preds.csv")
    print("COMPLETED ", pre)
    def compute_fin_metrics(preds, labels):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    print(compute_fin_metrics(preds, np.array(test['label'])))
