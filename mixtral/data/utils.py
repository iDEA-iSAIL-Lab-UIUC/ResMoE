from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch




TASK_MAPPING_DATASET_ARGUMENTS = {
    "winogrande": ["winogrande", "winogrande_xs"],
    "hellaswag": ["hellaswag"],
    "piqa": ["piqa"],
    "lambada": ["lambada"]
}

def my_dataset_map(dataset_name):
  dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[dataset_name])

  
    
  if dataset_name == "winogrande":
    
    def preprend(example):
      context1 = example['sentence'].split('_')[0] + example['option1']
      context2 = example['sentence'].split('_')[0] + example['option2']
      cont = example['sentence'].split('_')[1].strip(" ")
      return {"context1":context1,"context2":context2,"cont":cont}
  
  elif dataset_name == "hellaswag":
    
    def preprend(example):
      end1,end2,end3,end4 = example['endings']
      context = example['activity_label']+ ": " +example['ctx']
      return {"end1":end1,"end2":end2,"end3":end3,"end4":end4,"context":context}
    
  elif dataset_name == "piqa":
    
    def preprend(example):
      context = f"Question: {example['goal']}\nAnswer:"
      return {"context":context}
    
  elif dataset_name == "lambada":
    def preprend(example):
      prompts = f"{' '.join(example['text'].split(' ')[:-1])}"
      answer = f"{example['text'].split(' ')[-1]}"
      return {"context":prompts,"answer":answer}
    
    
  
    
  encoded_dataset = dataset.map(preprend)
    
  return encoded_dataset


     
def eval_map(val_dataset,dataset_name):
  if dataset_name == "copa":
    id2choices = {
        item['idx']: [item['choice1'], item['choice2']] for item in val_dataset
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }
  elif dataset_name == "mrpc":
    id2choices = {
        idx: ["Unequivalent","Equivalent"] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }
  elif dataset_name == "sst2":
    id2choices = {
        item['idx']: ["negative","positive"] for item in val_dataset
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }
    
  return id2choices,id2references