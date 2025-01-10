# Importing stock libraries
import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.utils.prune as prune
import torch.nn as nn
import scipy.sparse as sp




# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoConfig

from transformers import (
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP,
    SwitchTransformersTop1Router,
    PretrainedConfig
)
from torch import cuda
device = 'cuda:3' if cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

import ot
from data.utils import (
  TASK_MAPPING_DATASET_ARGUMENTS,
  my_dataset_map
)

from utils import(
  validate,
)
from data.utils import(
  eval_map,
  mySet
)

from wb import (
  weights_permute,
  get_optimal_permutation
)

  
avg_loss = 0
ot_without_permute_loss = 0
ot_loss = 0

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--q", default="0", type=int, help="expert")

args = parser.parse_args()
q = args.q


for layer in [5,7,9,11]:
  for type in ['encoder','decoder']:

    wd = torch.load(f'./switch-base-8-wd/{type}-layer{layer}.pt')    
    wd_avg = torch.mean((wd),dim=0)   
    wd_weights = torch.stack([torch.full((3072,), 1.0 / 3072) 
                                    for _ in range(16)])
                    
        
    wd_extract = ot.lp.free_support_barycenter(measures_locations=wd, measures_weights=wd_weights, X_init=wd[q], numItermax = 100,numThreads="max",stopThr=1e-14,verbose=True)

    torch.save(wd_extract,f'./extract_saved_8-{q}/wd-{type}-layer{layer}-ot.pt')

    T = get_optimal_permutation(wd, wd_weights, wd_extract)

    torch.save(T,f'./extract_saved_8-{q}/T-{type}-layer{layer}-ot.pt')