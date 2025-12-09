import torch
from transformers import AutoModelForCausalLM
import os


model_id = "mistralai/Mixtral-8x7B-v0.1"


model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)

os.makedirs('./wd_layer/', exist_ok=True)

for i in range(len(model.model.layers)):
  wdk = []
  MLP = model.model.layers[i].block_sparse_moe
  print(MLP)
  for idx in range(len(MLP.experts)):
    wdk.append(torch.cat((
      MLP.experts[idx].w1.weight,        
      MLP.experts[idx].w3.weight,
      MLP.experts[idx].w2.weight.T),1))
  wd = torch.stack(wdk)
    
  torch.save(wd,f'./wd_layer/layer{i}.pt')
    