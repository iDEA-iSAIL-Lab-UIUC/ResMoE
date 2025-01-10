# Importing stock libraries
import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import torch.nn as nn

import scipy.sparse as sp



# Importing the T5 modules from huggingface/transformers
from transformers import AutoTokenizer

from transformers import (
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP,
    SwitchForSequenceClassification
)
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# torch.cuda.set_device(device)

from datasets import load_dataset
import torch

from wb import (
  get_optimal_permutation,
  get_approx_loss,
  permute_weights,
  weights_permute
)


import ot

from utils import(
  validate,
  validate_head
)
from data.utils import(
  eval_map,
  mySet,
  my_dataset_map
)

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--dataset", default="sst2", type=str, help="dataset_name")

args = parser.parse_args()
dataset_name = args.dataset

sparsity = 0.75


def extract_and_compress(
  MLP: SwitchTransformersSparseMLP,
  sparsity,
  wd_extract = None,
  T = None
):
    
  expert_indices = MLP.experts.keys()
  n,p = MLP.experts['expert_0'].wi.weight.shape
  
  # There's no bias in switch-transformer
  
  with torch.no_grad():    
    w1 = torch.stack([(MLP.experts[idx].wi.weight) for idx in expert_indices])
    w2 = torch.stack([(MLP.experts[idx].wo.weight) for idx in expert_indices])    
    wd = torch.cat((w1,w2.transpose(1, 2)),dim=2)    
    wd_avg = torch.mean((wd),dim=0)
    wd_weights = torch.stack([torch.full((n,), 1.0 / n) 
                                    for _ in range(len(expert_indices))])    
   
    
    we1, we2 = weights_permute(w1,w2,T)
    wd = torch.cat((we1,we2.transpose(1, 2)),dim=2)    
  
  
  def res_pruning(wd_extract):    
    for i,idx in enumerate(expert_indices):
      ept = MLP.experts[idx]        
      wdx = wd[i]
      
      wd_res_layer = nn.Linear(2*p, n)        
      wd_res_layer.weight.data = wdx - wd_extract
      
      
      prune.l1_unstructured(wd_res_layer, name='weight', amount=sparsity)
      
      
      ept.wi.weight.data = ((wd_res_layer.weight.data + wd_extract)[:,:p])
      ept.wo.weight.data = (((wd_res_layer.weight.data + wd_extract)[:,p:]).T)
        
          
            
      
  res_pruning(wd_extract)
  return 

def expert_pruning(blocks,type,k,sparsity):
  for i in range(len(blocks)):
    if i<12-k:continue
    if i & 1:
      if type == "encoder":
        mlp = blocks[i].layer[1].mlp
      elif type == "decoder":
        mlp = blocks[i].layer[2].mlp
        
      print(f"{type}-block{i}")
      
      wd_extract = torch.load(f'./extract_saved-{q}/wd-{type}-layer{i}-ot.pt').to(device)

      T = torch.load(f'./extract_saved-{q}/T-{type}-layer{i}-ot.pt')      
      
      extract_and_compress(MLP=mlp,sparsity=sparsity,wd_extract=wd_extract,T=T)
      
    else:
      continue


val_dataset=my_dataset_map(dataset_name)["validation"]
val_set = mySet(val_dataset, tokenizer, 150, 20, dataset_name)

val_params = {
    'batch_size': 100,
    'shuffle': False,
    'num_workers': 2
}

val_loader = DataLoader(val_set, **val_params)

model = SwitchForSequenceClassification.from_pretrained(f"./switch-base-8-{dataset_name}").to(device)

expert_pruning(model.transformer.encoder.block,"encoder",8,sparsity)
expert_pruning(model.transformer.decoder.block,"decoder",8,sparsity)  

predictions, actuals = validate_head(model, device, val_loader)    
evaluation_result = metric.compute(predictions=predictions, references=actuals)

print(dataset_name,'\n',evaluation_result)