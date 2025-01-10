import torch
import ot

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--q", default="1", type=int, help="starting expert")
parser.add_argument("--layer", default="1", type=int, help="layer")

args = parser.parse_args()
layer = args.layer
q = args.q

import sys
sys.path.append('./')
from wb import(
  get_optimal_permutation
)



n = 14336
x = torch.load(f'./wd_layer/layer-{layer}.pt')


wd = torch.stack([x[i] for i in range(8)]).to(f'cuda')

del x
torch.cuda.empty_cache()

wd_weights = torch.stack([torch.full((n,), (1.0 / n),dtype=torch.float32) 
                                    for _ in range(8)]).to(f'cuda')

wd_extract = ot.lp.free_support_barycenter(measures_locations=wd, measures_weights=wd_weights, X_init=wd[q], numItermax = 100, numThreads='max')

torch.save(wd_extract,f'./extract_saved-{q}/wd-layer{layer}-ot.pt')

T = get_optimal_permutation(wd, wd_weights, wd_extract)

torch.save(T,f'./extract_saved-{q}/T-layer{layer}-ot.pt')
