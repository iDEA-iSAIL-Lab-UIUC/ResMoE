# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import sys
import wandb
import evaluate
sys.path.append("./")

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, SwitchTransformersForConditionalGeneration,T5ForSequenceClassification

from switch_transformers import SwitchForSequenceClassification, SwitchTransformersConfig

# Setting up the device for GPU usage
from torch import cuda
# device = 'cuda:7' if cuda.is_available() else 'cpu'

from data.utils import (
  TASK_MAPPING_DATASET_ARGUMENTS,
  my_dataset_map,
  mySet,
  Config,
  eval_map,
  mySeq2SeqSet
)

from utils import(
  train,
  validate,
  train_val,
  freeze_switch_routers_and_experts_for_finetuning,
  freeze_switch_routers_for_finetuning,
  validate_seq2seq,
  validate_head
)

import logging
# warm-up
from transformers import get_linear_schedule_with_warmup,get_scheduler
from argparse import ArgumentParser

from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device

parser = ArgumentParser()
parser.add_argument("--dataset", default="sst2", type=str, help="dataset_name")

args = parser.parse_args()
dataset_name = args.dataset


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(levelname)s| %(message)s',
                    filename='finetuning.log',  # Log to this file
                    filemode='a')  # Append to the file, 'w' to overwrite


def train_val(epoch, model, device, train_loader, optimizer,val_loader,scheduler,accelerator):
     
  
    model.train()
    for _,data in enumerate(train_loader, 0):
        
        labels = data['target_ids_y'].to(dtype = torch.long)
        ids = data['source_ids'].to(dtype = torch.long)
        mask = data['source_mask'].to(dtype = torch.long)
        
        outputs = model(input_ids = ids, attention_mask = mask, labels=labels,return_dict=True)
        train_loss = outputs[0]      
        wandb.log({"train_loss": train_loss.item()})    
        
        
        accelerator.backward(train_loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        
        
    model.eval()
    with torch.no_grad():  # No need to track the gradients
        for _, data in enumerate(val_loader, 0):
            
            labels = data['target_ids_y'].to(dtype=torch.long)
            ids = data['source_ids'].to(dtype=torch.long)
            mask = data['source_mask'].to(dtype=torch.long)
            
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
            loss = outputs.loss
            wandb.log({"val_loss": loss.item()})    

    logging.info(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {loss}")
    return train_loss, loss
  
  
config = Config()       # Initialize config
config.TRAIN_BATCH_SIZE = 64   # input batch size for training (default: 64)
config.VALID_BATCH_SIZE = 100   # input batch size for testing (default: 1000)
config.TRAIN_EPOCHS = 10      # number of epochs to train (default: 10)
config.VAL_EPOCHS = 1 
config.LEARNING_RATE = 2e-4    # learning rate (default: 0.01)
config.SEED = 42               # random seed (default: 42)
config.INPUT_MAX_LEN = 100
config.OUT_MAX_LEN = 10
    
    


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")   


dataset_name = "mnli"
    
encoded_dataset = my_dataset_map(dataset_name)    
train_dataset=encoded_dataset["train"]
val_dataset=encoded_dataset["validation_matched"] if dataset_name == "mnli" else encoded_dataset["validation"]
test_dataset=encoded_dataset["test"]
  
  
# Creating the Training and Validation dataset for further creation of Dataloader
train_set = mySet(train_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)
val_set = mySet(val_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)
test_set = mySet(test_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': config.TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 2
    }

val_params = {
    'batch_size': config.VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 2
    }

train_loader = DataLoader(train_set, **train_params)
val_loader = DataLoader(val_set, **val_params)
test_loader = DataLoader(test_set, **val_params)



switch_config = SwitchTransformersConfig.from_pretrained(
        "google/switch-base-8",
        num_labels=3 if dataset_name == "mnli" else 2, 
        finetuning_task=dataset_name
)

model = SwitchForSequenceClassification.from_pretrained(
            "google/switch-base-8",
            config=switch_config,
            torch_dtype = torch.bfloat16
)


model = freeze_switch_routers_and_experts_for_finetuning(model)


no_decay = ["bias", "layer_norm.weight", "LayerNorm", "layernorm", "layer_norm", "ln"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(params = optimizer_grouped_parameters, lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)

model, optimizer, train_loader, val_dataloader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

num_steps_per_epoch = len(train_loader)
num_training_steps = num_steps_per_epoch * config.TRAIN_EPOCHS


scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps= 8 * num_steps_per_epoch,
    num_training_steps=num_training_steps
)




wandb.init(
    # set the wandb project where this run will be logged
    project=f"switch-base-8-{dataset_name}",
    config={
    "learning_rate": config.LEARNING_RATE,
    "batch_size": config.TRAIN_BATCH_SIZE,
    "epochs": config.TRAIN_EPOCHS,
    }
)
  
# Training loop
metric = evaluate.load("accuracy")
print('Initiating Fine-Tuning for the model')

logging.info(f"BEGINING THE FINETUNING FOR switch-base-8-{dataset_name}-epoch{str(config.TRAIN_EPOCHS)}-batch{str(config.TRAIN_BATCH_SIZE)}-lr{str(config.LEARNING_RATE)}")
for epoch in range(config.TRAIN_EPOCHS):
    train_val(epoch,  model, device, train_loader, optimizer,val_loader,scheduler,accelerator)    
    
    predictions, actuals = validate_head(model, device, val_loader)
    
    result = metric.compute(predictions=predictions, references=actuals)
    
    print("Epoch: ",epoch,"\n",result)
  
  
model.save_pretrained(f"./switch-base-8-{dataset_name}", from_pt=True) 