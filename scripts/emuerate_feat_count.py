import os
import sys
import numpy as np
import cv2
import torch
import sklearn
from tqdm import tqdm
from sklearn.manifold import TSNE
import pandas as pd

dataset = sys.argv[1]

count_bins = int(sys.argv[2])

pth_path = sys.argv[3]

save_name = sys.argv[4]

pths = os.listdir(pth_path)
pths = [i for i in pths if '.pt' in i]

# all_pths = []
all_maxs = -3000000000
all_mins = 30000000000

# scannet_lseg_min = -0.008756677620112896
# scannet_lseg_max = 0.015098048374056816

# scannet_dinov2_min = -0.05177316814661026
# scannet_dinov2_max = 0.0815395712852478

# scannet_sd_min = -11.126303672790527
# scannet_sd_max = 1.4777133464813232

scannet_lseg_min = -0.01
scannet_lseg_max = 0.02

scannet_dinov2_min = -0.08
scannet_dinov2_max = 0.1

scannet_sd_min = -12.0
scannet_sd_max = 1.8

# # for i_pt in tqdm(pths[:50]):
# for i_pt in tqdm(pths):
#     cur_pt = torch.load(os.path.join(pth_path, i_pt))['feat']
#     # cur_pt = torch.mean(cur_pt, dim=0)
    
#     all_pths.append(cur_pt)

# all_pths = torch.cat(all_pths, dim=0).numpy()

# x_tsne = TSNE(n_components=2).fit_transform(all_pths)

# for i_pt in tqdm(pths[:50]):

all_counts = np.zeros((count_bins-1, 1))

for i_pt in tqdm(pths[:]):
    cur_pt = torch.load(os.path.join(pth_path, i_pt))['feat'].float()
    # cur_pt = torch.mean(cur_pt, dim=0)
    
    # import pdb;pdb.set_trace()
    
    cur_pt = cur_pt.mean(dim=-1)
    
    # import pdb;pdb.set_trace()
    
    # cur_groups = pd.cut(cur_pt, bins=[-0.00875668, -0.00610615, -0.00345563, -0.0008051 ,  0.00184542,
    #     0.00449595,  0.00714647,  0.009797  ,  0.01244752,  0.01509805])
    # print(cur_groups.value_counts())
    
    # cur_groups = pd.cut(cur_pt, bins=[-0.05177317, -0.03696064, -0.02214811, -0.00733559,  0.00747694,
    #     0.02228946,  0.03710199,  0.05191452,  0.06672704,  0.08153957])
    # print(cur_groups.value_counts())
    
    # # cur_groups = pd.cut(cur_pt, bins=list(np.linspace(-11.126303672790527, 1.4777133464813232, 50)))
    # cur_groups = pd.cut(cur_pt, bins=list(np.linspace(-12, 2.0, 100)))
    # print(cur_groups.value_counts())
    
    if dataset == 'lseg':
        cur_groups = pd.cut(cur_pt, bins=list(np.linspace(scannet_lseg_min, scannet_lseg_max, count_bins)))
    elif dataset == 'dinov2':
        cur_groups = pd.cut(cur_pt, bins=list(np.linspace(scannet_dinov2_min, scannet_dinov2_max, count_bins)))
    elif dataset == 'sd':
        cur_groups = pd.cut(cur_pt, bins=list(np.linspace(scannet_sd_min, scannet_sd_max, count_bins)))
    else:
        raise ValueError("Not imple yet ...")
    # print(all_counts.shape)
    
    print(cur_groups.value_counts())
    print(cur_groups.value_counts().values.shape)
    
    cur_t = cur_groups.value_counts().values[..., None]
    
    all_counts = np.concatenate([all_counts, cur_t], axis=1)

    # import pdb;pdb.set_trace()

all_sum_count = np.sum(all_counts, axis=1)
print(all_sum_count, all_sum_count.shape)
np.save(save_name + ".npy", all_counts)

