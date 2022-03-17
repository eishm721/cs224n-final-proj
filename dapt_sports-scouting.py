#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import os
from huggingface_hub import notebook_login
import transformers
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
import math
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling


# In[35]:


block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# In[36]:


print("starting notebook login")
notebook_login()


# In[37]:


print("done with notebook login")
wiki_train = "wikitext"
wiki_val = "wikitext-2-raw-v1"
sports_train = "data/sports_article_data.csv"
sports_val = "data/sports_article_data.csv"
scouting_train = "data/unlabeled_scouting.csv"
scouting_val = "data/unlabeled_scouting.csv"
base_model = "bert-base-uncased" # replace with best base model as determined by Amol's baseline experiments


# In[38]:


def run_pretraining_stage(datasets, model_loc, stage_name):
    if model_loc == "amanm27/" + base_model:
        # first pre-training stage
        model_checkpoint = base_model
    else:
        # already pre-trained on something, so resume where we left off
        model_checkpoint = model_loc
    print("Reading model from " + model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(base_model, use_fast=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(model_name + stage_name, evaluation_strategy = "epoch", learning_rate=2e-5, weight_decay=0.01, push_to_hub=True)
    model_loc += stage_name 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = Trainer(model=model, args=training_args, train_dataset=lm_datasets["train"], eval_dataset=lm_datasets["validation"], data_collator=data_collator)
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.push_to_hub()
    print("Done with " + stage_name + " stage")
    return model_loc


# In[39]:


def run_pretraining_experiment(include_wiki, include_sports, include_scouting):
    #model_loc = "amanm27/"
    #model_loc += base_model
    model_loc = "amanm27/bert-base-uncased-sports"
    #if include_wiki:
    #    datasets = load_dataset(wiki_train, wiki_val)
    #    model_loc = run_pretraining_stage(datasets, model_loc, "-wiki")
    #    print("Saved model checkpoint to " + model_loc)
    #if include_sports:
    #    datasets = load_dataset("text", data_files={"train": sports_train, "validation": sports_val})
    #    model_loc = run_pretraining_stage(datasets, model_loc, "-sports")
    #    print("Saved model checkpoint to " + model_loc)
    if include_scouting:
        datasets = load_dataset("text", data_files={"train": scouting_train, "validation": scouting_val})
        model_loc = run_pretraining_stage(datasets, model_loc, "-scouting")
        print("Saved model checkpoint to " + model_loc)
    print("Saved final model checkpoint to " + model_loc)
          

# In[ ]:


print("Starting experiment: sports --> scouting")
run_pretraining_experiment(False, True, True)
print("Done with experiment: sports --> scouting")

