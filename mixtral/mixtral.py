import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

import sys
sys.path.append('./')

import torch

from datasets import load_dataset, load_metric

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from tqdm import tqdm


import itertools
import time

import torch.nn as nn

from data.utils import(
  my_dataset_map
)

from utils import (
  evaluate_model,
  ppl
)

from wb import (
  get_optimal_permutation,
  permute_weights
)



def extract_and_compress(
  MLP,
  sparsity
):
  
  n,p = MLP.experts[0].w1.weight.shape
  working_device = 'cuda' if cuda.is_available() else 'cpu'
  
  
  # There's no bias in mixtral, also no bias in mixtral
  
    
  def res_compress(wd_extract):          
    wd_extract = wd_extract.to(working_device)
    for idx in range(8):
      ept = MLP.experts[idx]
      current_device = ept.w1.weight.device
      ept = ept.to(working_device)
      
      wdx = torch.cat((ept.w1.weight,ept.w3.weight,ept.w2.weight.T),1)
      
      T = torch.load(f'./extract_saved-0/T-layer{layer}-{extract_type}.pt')
      wdx = weights_permute(wdx,T[idx])
      
      wd_res_layer = nn.Linear(3*p, n)
      wd_res_layer.weight.data = wdx - wd_extract
      
      prune.l1_unstructured(wd_res_layer, name='weight', amount=sparsity)
      
      wd_now = (wd_res_layer.weight + wd_extract).to(current_device)
        
      
      ept.w1.weight.data = wd_now[:,:p]
      ept.w3.weight.data = wd_now[:,p:2*p]
      ept.w2.weight.data = wd_now[:,2*p:].T
      
      del wdx,wd_now,wd_res_layer
      torch.cuda.empty_cache()
      
      ept = ept.to(current_device)
    
    del wd_extract
    torch.cuda.empty_cache()
  
  
  wd_extract = torch.load(f'./extract_saved-0/layer{layer}-{extract_type}.pt').to(torch.float16)
  
    
  res_compress(wd_extract)  
  
  return 
  

def expert_pruning(k,sparsity,ex_type=None,cp_type=None):
  for i in range(32):
    if i<32-k:
      continue
    else:
      print(f"Layer-{i}")
      
      
      extract_and_compress(
        MLP=model.model.layers[i].block_sparse_moe,
        sparsity=sparsity)
      




model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load dataset
winogrande = "winogrande"
winogrande_set = my_dataset_map(winogrande)['validation']

piqa = "piqa"
piqa_set = my_dataset_map(piqa)['validation']

test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

lambada = "lambada"
lambada_set = my_dataset_map(lambada)['test']



from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--k", default="24", type=int, help="topk")
parser.add_argument("--s", default="0.75", type=float, help="sparsity")

args = parser.parse_args()
k = args.k
sparsity = args.s


result = {}

  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16)
model.eval()
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

expert_pruning(k,sparsity)  

t = time.time()

wino_result = evaluate_model(model=model, tokenizer=tokenizer, dataset=winogrande_set,dataset_name=winogrande,batch_size=64)
piqa_result = evaluate_model(model=model, tokenizer=tokenizer, dataset=piqa_set,dataset_name=piqa,batch_size=64)
wiki_result = ppl(model,encodings)
ld_result = evaluate_model(model=model, tokenizer=tokenizer, dataset=lambada_set,dataset_name=lambada,batch_size=64)

result[f"wino-k{k}sparse{sparsity}"]=wino_result
result[f"piqa-k{k}sparse{sparsity}"]=piqa_result
result[f"wiki-k{k}sparse{sparsity}"]=wiki_result
result[f"lamb-k{k}sparse{sparsity}"]=ld_result

print(f"take {time.time()-t} time.")

print(result)