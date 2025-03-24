import os
import sys
import numpy as np
import cv2
import torch
import sklearn
from tqdm import tqdm
from sklearn.manifold import TSNE

pth_path = sys.argv[1]

pths = os.listdir(pth_path)
pths = [i for i in pths if '.pt' in i]

# all_pths = []
all_maxs = -3000000000
all_mins = 30000000000

# # for i_pt in tqdm(pths[:50]):
# for i_pt in tqdm(pths):
#     cur_pt = torch.load(os.path.join(pth_path, i_pt))['feat']
#     # cur_pt = torch.mean(cur_pt, dim=0)
    
#     all_pths.append(cur_pt)
# import pdb;pdb.set_trace()
# all_pths = torch.cat(all_pths, dim=0).numpy()

# x_tsne = TSNE(n_components=2).fit_transform(all_pths)

# for i_pt in tqdm(pths[:50]):
for i_pt in tqdm(pths):
    cur_pt = torch.load(os.path.join(pth_path, i_pt))['feat'].float()
    if len(cur_pt) < 2:
        continue
    # cur_pt = torch.mean(cur_pt, dim=0)
    cur_pt = cur_pt.mean(dim=-1)
    
    cur_min = torch.min(cur_pt).item()
    cur_max = torch.max(cur_pt).item()
    
    all_maxs = all_maxs if all_maxs > cur_max else cur_max
    
    all_mins = all_mins if all_mins < cur_min else cur_min
    
print(all_mins, all_maxs)

