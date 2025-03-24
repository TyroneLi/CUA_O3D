# import os
# import random
# import numpy as np
# import logging
# import argparse
# import urllib

# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from util import metric
# from torch.utils import model_zoo

# from MinkowskiEngine import SparseTensor
# from util import config
# from util.util import export_pointcloud, get_palette, \
#     convert_labels_with_palette, extract_text_feature, visualize_labels
# from tqdm import tqdm
# # from run.distill import get_model
# from run.self_distill import get_model

# from dataset.label_constants import *


# def get_parser():
#     '''Parse the config file.'''

#     parser = argparse.ArgumentParser(description='OpenScene evaluation')
#     parser.add_argument('--config', type=str,
#                     default='config/scannet/eval_openseg.yaml',
#                     help='config file')
#     parser.add_argument('opts',
#                     default=None,
#                     help='see config/scannet/test_ours_openseg.yaml for all options',
#                     nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     if args.opts is not None:
#         cfg = config.merge_cfg_from_list(cfg, args.opts)
#     return cfg


# def get_logger():
#     '''Define logger.'''

#     logger_name = "main-logger"
#     logger_in = logging.getLogger(logger_name)
#     logger_in.setLevel(logging.DEBUG)
#     handler = logging.StreamHandler()
#     fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
#     handler.setFormatter(logging.Formatter(fmt))
#     logger_in.addHandler(handler)
#     return logger_in

# def is_url(url):
#     scheme = urllib.parse.urlparse(url).scheme
#     return scheme in ('http', 'https')

# def main_process():
#     return not args.multiprocessing_distributed or (
#             args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

# def precompute_text_related_properties(labelset_name):
#     '''pre-compute text features, labelset, palette, and mapper.'''
#     if 'scannet_3d_200' == labelset_name:
#         labelset = list(SCANNET_LABELS_200)
#         palette = get_palette(colormap='scannet200')
#     elif 'scannet_3d' in labelset_name:
#         labelset = list(SCANNET_LABELS_20)
#         labelset[-1] = 'other' # change 'other furniture' to 'other'
#         palette = get_palette(colormap='scannet')
#     elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
#         labelset = list(MATTERPORT_LABELS_21)
#         palette = get_palette(colormap='matterport')
#     elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
#         labelset = list(MATTERPORT_LABELS_40)
#         palette = get_palette(colormap='matterport_160')
#     elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
#         labelset = list(MATTERPORT_LABELS_80)
#         palette = get_palette(colormap='matterport_160')
#     elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
#         labelset = list(MATTERPORT_LABELS_160)
#         palette = get_palette(colormap='matterport_160')
#     elif 'nuscenes' in labelset_name:
#         labelset = list(NUSCENES_LABELS_16)
#         palette = get_palette(colormap='nuscenes16')
#     else: # an arbitrary dataset, just use a large labelset
#         labelset = list(MATTERPORT_LABELS_160)
#         palette = get_palette(colormap='matterport_160')

#     mapper = None
#     if hasattr(args, 'map_nuscenes_details'):
#         labelset = list(NUSCENES_LABELS_DETAILS)
#         mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

#     text_features = extract_text_feature(labelset, args)
#     # labelset.append('unknown')
#     labelset.append('unlabeled')
#     return text_features, labelset, mapper, palette

# def main():
#     '''Main function.'''

#     args = get_parser()

#     # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
#     cudnn.benchmark = True
#     if args.manual_seed is not None:
#         random.seed(args.manual_seed)
#         np.random.seed(args.manual_seed)
#         torch.manual_seed(args.manual_seed)
#         torch.cuda.manual_seed(args.manual_seed)
#         torch.cuda.manual_seed_all(args.manual_seed)

#     print(
#         'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
#             torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
#     args.ngpus_per_node = len(args.test_gpu)
#     if len(args.test_gpu) == 1:
#         args.sync_bn = False
#         args.distributed = False
#         args.multiprocessing_distributed = False
#         args.use_apex = False

    
#     # By default we do not use shared memory for evaluation
#     if not hasattr(args, 'use_shm'):
#         args.use_shm = False
#     if args.use_shm:
#         if args.multiprocessing_distributed:
#             args.world_size = args.ngpus_per_node * args.world_size
#             mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
#     else:
#         main_worker(args.test_gpu, args.ngpus_per_node, args)


# def main_worker(gpu, ngpus_per_node, argss):
#     global args
#     args = argss
#     if args.distributed:
#         if args.multiprocessing_distributed:
#             args.rank = args.rank * ngpus_per_node + gpu
#         dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
#                                 rank=args.rank)

#     model = get_model(args)
#     if main_process():
#         global logger
#         logger = get_logger()
#         logger.info(args)

#     if args.distributed:
#         torch.cuda.set_device(gpu)
#         args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
#         args.test_workers = int(args.test_workers / ngpus_per_node)
#         model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
#     else:
#         model = model.cuda()

#     if args.feature_type == 'fusion':
#         pass # do not need to load weight
#     elif is_url(args.model_path): # load from url
#         checkpoint = model_zoo.load_url(args.model_path, progress=True)
#         model.load_state_dict(checkpoint['state_dict'], strict=True)
    
#     elif args.model_path is not None and os.path.isfile(args.model_path):
#         # # load from directory
#         # if main_process():
#         #     logger.info("=> loading checkpoint '{}'".format(args.model_path))
#         # checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
#         # try:
#         #     model.load_state_dict(checkpoint['state_dict'], strict=True)
#         # except Exception as ex:
#         #     # The model was trained in a parallel manner, so need to be loaded differently
#         #     from collections import OrderedDict
#         #     new_state_dict = OrderedDict()
#         #     for k, v in checkpoint['state_dict'].items():
#         #         if k.startswith('module.'):
#         #             # remove module
#         #             k = k[7:]
#         #         else:
#         #             # add module
#         #             k = 'module.' + k

#         #         new_state_dict[k]=v
#         #     model.load_state_dict(new_state_dict, strict=True)
#         #     logger.info('Loaded a parallel model')
        
#         # load from directory
#         if main_process():
#             logger.info("=> loading checkpoint '{}'".format(args.model_path))
#             if os.path.exists((args.model_path).rsplit("/", 1)[0]+"/ema_model_best.pth.tar"):
#                 args.model_path = (args.model_path).rsplit("/", 1)[0]+"/ema_model_best.pth.tar"
#                 logger.info("=> change loading checkpoint '{}'".format(args.model_path))
#         checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
#         try:
#             # model.load_state_dict(checkpoint['state_dict'], strict=True)
#             print("loading ema model ...")
#             model.load_state_dict(checkpoint['ema_state_dict'], strict=True)
#             print("best iou during training : {}".format(checkpoint['best_iou']))
#             # print("best ema iou during training : {}".format(checkpoint['best_ema_iou']))
#         except Exception as ex:
#             # The model was trained in a parallel manner, so need to be loaded differently
#             from collections import OrderedDict
#             new_state_dict = OrderedDict()
#             for k, v in checkpoint['state_dict'].items():
#                 if k.startswith('module.'):
#                     # remove module
#                     k = k[7:]
#                 else:
#                     # add module
#                     k = 'module.' + k

#                 new_state_dict[k]=v
#             model.load_state_dict(new_state_dict, strict=True)
#             logger.info('Loaded a parallel model')

#         if main_process():
#             logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
#     else:
#         raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

#     # ####################### Data Loader ####################### #
#     if not hasattr(args, 'input_color'):
#         # by default we do not use the point color as input
#         args.input_color = False
    
#     from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
#     val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
#                                 datapath_prefix_feat=args.data_root_2d_fused_feature,
#                                 voxel_size=args.voxel_size, 
#                                 split=args.split, aug=False,
#                                 memcache_init=args.use_shm, eval_all=True, identifier=6797,
#                                 input_color=args.input_color)
#     val_sampler = None
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
#                                                 shuffle=False, num_workers=args.test_workers, pin_memory=True,
#                                                 drop_last=False, collate_fn=collation_fn_eval_all,
#                                                 sampler=val_sampler)

#     # ####################### Test ####################### #
#     # import pdb;pdb.set_trace()
#     labelset_name = args.data_root.split('/')[-1]
#     if hasattr(args, 'labelset'):
#         # if the labelset is specified
#         labelset_name = args.labelset

#     evaluate(model, val_loader, labelset_name)

# def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
#     '''Evaluate our OpenScene model.'''

#     torch.backends.cudnn.enabled = False

#     if not os.path.exists(args.save_folder):
#         os.makedirs(args.save_folder, exist_ok=True)

#     if args.save_feature_as_numpy: # save point features to folder
#         out_root = os.path.commonprefix([args.save_folder, args.model_path])
#         saved_feature_folder = os.path.join(out_root, 'saved_feature')
#         os.makedirs(saved_feature_folder, exist_ok=True)

#     # short hands
#     save_folder = args.save_folder
#     feature_type = args.feature_type
#     eval_iou = True
#     if hasattr(args, 'eval_iou'):
#         eval_iou = args.eval_iou
#     mark_no_feature_to_unknown = False
#     if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
#         # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
#         mark_no_feature_to_unknown = True
#     vis_input = False
#     if hasattr(args, 'vis_input') and args.vis_input:
#         vis_input = True
#     vis_pred = False
#     if hasattr(args, 'vis_pred') and args.vis_pred:
#         vis_pred = True
#     vis_gt = False
#     if hasattr(args, 'vis_gt') and args.vis_gt:
#         vis_gt = True

#     text_features, labelset, mapper, palette = \
#         precompute_text_related_properties(labelset_name)
    
#     print("evaluating labelset : ", labelset_name)
#     # import pdb;pdb.set_trace()
    
#     with torch.no_grad():
#         model.eval()
#         store = 0.0
#         args.test_repeats = 5
#         for rep_i in range(args.test_repeats):
#             preds, gts = [], []
#             val_data_loader.dataset.offset = rep_i
#             if main_process():
#                 logger.info(
#                     "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

#             # repeat the evaluation process
#             # to account for the randomness in MinkowskiNet voxelization
#             if rep_i>0:
#                 seed = np.random.randint(10000)
#                 random.seed(seed)
#                 np.random.seed(seed)
#                 torch.manual_seed(seed)
#                 torch.cuda.manual_seed(seed)
#                 torch.cuda.manual_seed_all(seed)

#             if mark_no_feature_to_unknown:
#                 masks = []

#             for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
#                 sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
#                 coords = coords[inds_reverse, :]
#                 pcl = coords[:, 1:].cpu().numpy()
                
#                 if feature_type == 'distill':
#                     predictions, _ = model(sinput)
#                     predictions = predictions[inds_reverse, :]
#                     pred = predictions.half() @ text_features.t()
#                     logits_pred = torch.max(pred, 1)[1].cpu()
#                 elif feature_type == 'fusion':
#                     predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
#                     pred = predictions.half() @ text_features.t()
#                     logits_pred = torch.max(pred, 1)[1].detach().cpu()
#                     if mark_no_feature_to_unknown:
#                     # some points do not have 2D features from 2D feature fusion.
#                     # Directly assign 'unknown' label to those points during inference.
#                         logits_pred[~mask[inds_reverse]] = len(labelset)-1

#                 elif feature_type == 'ensemble':
#                     feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
#                     # pred_fusion = feat_fuse.half() @ text_features.t()
#                     pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

#                     predictions = model(sinput)
#                     predictions = predictions[inds_reverse, :]
#                     # pred_distill = predictions.half() @ text_features.t()
#                     pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

#                     # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
#                     # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
#                     # pred = pred_distill
#                     # pred[mask_ensem] = pred_fusion[mask_ensem]
#                     # logits_pred = torch.max(pred, 1)[1].detach().cpu()

#                     feat_ensemble = predictions.clone().half()
#                     mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
#                     # feat_ensemble[mask_] = feat_fuse[mask_]
#                     feat_ensemble = ((feat_fuse + feat_ensemble.float()) / 2.0).half()
#                     pred = feat_ensemble @ text_features.t()
#                     logits_pred = torch.max(pred, 1)[1].detach().cpu()

#                     predictions = feat_ensemble # if we need to save the features
#                 else:
#                     raise NotImplementedError

#                 if args.save_feature_as_numpy:
#                     scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
#                     np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())
                
#                 # Visualize the input, predictions and GT

#                 # special case for nuScenes, evaluation points are only a subset of input
#                 if 'nuscenes' in labelset_name:
#                     label_mask = (label!=255)
#                     label = label[label_mask]
#                     logits_pred = logits_pred[label_mask]
#                     pred = pred[label_mask]
#                     if vis_pred:
#                         pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]

#                 save = False

#                 # if vis_input:
#                 if save:
#                     input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
#                     export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

#                 # if vis_pred:
#                 if save:
#                     if mapper is not None:
#                         pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
#                         export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
#                     else:
#                         pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
#                         export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
#                         visualize_labels(list(np.unique(logits_pred.numpy())),
#                                     labelset,
#                                     palette,
#                                     os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

#                 # Visualize GT labels
#                 # if vis_gt:
#                 if save:
#                     # for points not evaluating
#                     label[label==255] = len(labelset)-1
#                     gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
#                     export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
#                     visualize_labels(list(np.unique(label.cpu().numpy())),
#                                 labelset,
#                                 palette,
#                                 os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

#                     if 'nuscenes' in labelset_name:
#                         all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
#                         labelset = list(NUSCENES_LABELS_16)
#                         labelset[4] = 'construct. vehicle'
#                         labelset[10] = 'road'
#                         visualize_labels(list(all_digits), labelset, 
#                             palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=all_digits.shape[0])

#                 if eval_iou:
#                     if mark_no_feature_to_unknown:
#                         if "nuscenes" in labelset_name: # special case
#                             masks.append(mask[inds_reverse][label_mask])
#                         else:
#                             masks.append(mask[inds_reverse])

#                     if args.test_repeats==1:
#                         # save directly the logits
#                         preds.append(logits_pred)
#                     else:
#                         # only save the dot-product results, for repeat prediction
#                         preds.append(pred.cpu())

#                     gts.append(label.cpu())

#             if eval_iou:
#                 gt = torch.cat(gts)
#                 pred = torch.cat(preds)

#                 pred_logit = pred
#                 if args.test_repeats>1:
#                     pred_logit = pred.float().max(1)[1]

#                 if mapper is not None:
#                     pred_logit = mapper[pred_logit]

#                 if mark_no_feature_to_unknown:
#                     mask = torch.cat(masks)
#                     pred_logit[~mask] = 256

#                 if args.test_repeats==1:
#                     current_iou = metric.evaluate(pred_logit.numpy(),
#                                                 gt.numpy(),
#                                                 dataset=labelset_name,
#                                                 stdout=True)
#                 if args.test_repeats > 1:
#                     store = pred + store
#                     store_logit = store.float().max(1)[1]
#                     if mapper is not None:
#                         store_logit = mapper[store_logit]

#                     if mark_no_feature_to_unknown:
#                         store_logit[~mask] = 256
#                     accumu_iou = metric.evaluate(store_logit.numpy(),
#                                                 gt.numpy(),
#                                                 stdout=True,
#                                                 dataset=labelset_name)

# if __name__ == '__main__':
#     main()

import os
import random
import numpy as np
import logging
import argparse
import urllib
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from run.self_distill import get_model

from dataset.label_constants import *

from pointnet2 import knn as pointnet_knn
from pointnet2 import grouping_operation

SCANNET_NAMES = [
    'wall',
    'floor',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'desk',
    'curtain',
    'refrigerator',
    'shower curtain',
    'toilet',
    'sink',
    'bathtub',
    'otherfurniture',
]

def calc_entropy(x, step=0):
    b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
    # b = torch.argmax(x, dim=1)[:, None] * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum(dim=1)
    return b

def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    if hasattr(args, 'map_nuscenes_details'):
        labelset = list(NUSCENES_LABELS_DETAILS)
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette

def main():
    '''Main function.'''

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    
    # By default we do not use shared memory for evaluation
    if not hasattr(args, 'use_shm'):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == 'fusion':
        pass # do not need to load weight
    elif is_url(args.model_path): # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            if os.path.exists((args.model_path).rsplit("/", 1)[0]+"/ema_model_best.pth.tar"):
                args.model_path = (args.model_path).rsplit("/", 1)[0]+"/ema_model_best.pth.tar"
                logger.info("=> change loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            # model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("loading ema model ...")
            model.load_state_dict(checkpoint['ema_state_dict'], strict=True)
            print("best iou during training : {}".format(checkpoint['best_iou']))
            # print("best ema iou during training : {}".format(checkpoint['best_ema_iou']))
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')

        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    
    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset

    evaluate(model, val_loader, labelset_name)

def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy: # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, 'vis_input') and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, 'vis_pred') and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, 'vis_gt') and args.vis_gt:
        vis_gt = True

    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)

    print("evaluating labelset : ", labelset_name)

    with torch.no_grad():
        model.eval()
        store = 0.0
        args.test_repeats = 1
        
        bin_values = [
                [0.0, 0.05],
                [0.05, 0.10],
                [0.10, 0.15],
                [0.15, 0.20],
                [0.20, 0.25],
                [0.25, 0.30],
                [0.30, 0.35],
                [0.35, 0.40],
                [0.40, 0.45],
                [0.45, 0.50],
                [0.50, 0.55],
                [0.55, 0.60],
                [0.60, 0.65],
                [0.65, 0.70],
                [0.70, 0.75],
                [0.75, 0.80],
                [0.80, 0.85],
                [0.85, 0.90],
                [0.90, 0.95],
                [0.95, 1.0]
            ]

        bin_seg_mious = []
        bin_seg_macc = []
        
        for cur_bin in bin_values:
            print("cur_bin ... ", cur_bin)
            
            for rep_i in range(args.test_repeats):
                preds, gts = [], []
                val_data_loader.dataset.offset = rep_i
                if main_process():
                    logger.info(
                        "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

                # repeat the evaluation process
                # to account for the randomness in MinkowskiNet voxelization
                if rep_i>0:
                    seed = np.random.randint(10000)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                if mark_no_feature_to_unknown:
                    masks = []

                print("K : ", args.K)

                for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                    sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                    coords = coords[inds_reverse, :]
                    coords = coords.cuda(non_blocking=True)
                    pcl = coords[:, 1:].cpu().numpy()

                    # if i > 100:
                    #     break

                    if feature_type == 'distill':
                        predictions, U_3D = model(sinput)
                        orig_u_3d = U_3D.clone()
                        orig_u_3d = orig_u_3d[inds_reverse, :]

                        U_3D = (U_3D - U_3D.min()) / (U_3D.max() - U_3D.min())

                        # predictions *= (1.0 - U_3D)
                        # predictions *= U_3D
                        
                        # predictions *= torch.exp(-1.0 * U_3D)
                        
                        # print("here")

                        # U_3D = U_3D[inds_reverse, :]
                        
                        low_ucer_bin, high_uncer_bin = cur_bin
                        eval_point_mask = ((U_3D >= low_ucer_bin) & (U_3D < high_uncer_bin))[:, 0]
                        eval_point_mask = eval_point_mask[inds_reverse]
                        
                        predictions = predictions[inds_reverse, :]

                        # fused_feat = feat_3d.cuda(non_blocking=True)[inds_reverse, :]

                        # cosine_loss = (1.0 - torch.nn.CosineSimilarity()(predictions, fused_feat))
                        # pred_loss = torch.nn.L1Loss(reduction='none')(cosine_loss[..., None], orig_u_3d)
                        # pred_loss = (pred_loss - pred_loss.min()) / (pred_loss.max() - pred_loss.min())

                        # # predictions *= (1.0 - pred_loss)

                        pred = predictions.half() @ text_features.t()

                        
                        logits_pred = torch.max(pred, 1)[1].cpu()
                        
                        label = label[eval_point_mask]
                        pred = pred[eval_point_mask]
                        logits_pred = logits_pred[eval_point_mask]
                        
                    elif feature_type == 'fusion':
                        predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                        pred = predictions.half() @ text_features.t()
                        logits_pred = torch.max(pred, 1)[1].detach().cpu()
                        if mark_no_feature_to_unknown:
                        # some points do not have 2D features from 2D feature fusion.
                        # Directly assign 'unknown' label to those points during inference.
                            logits_pred[~mask[inds_reverse]] = len(labelset)-1

                    elif feature_type == 'ensemble':
                        feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                        # pred_fusion = feat_fuse.half() @ text_features.t()
                        pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                        predictions, U_3D = model(sinput)

                        # U_3D = (U_3D - U_3D.min()) / (U_3D.max() - U_3D.min())
                        # predictions *= (1.0 - U_3D)
                        
                        predictions *= torch.exp(-1.0 * U_3D)
                        
                        predictions = predictions[inds_reverse, :]
                        # pred_distill = predictions.half() @ text_features.t()
                        pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                        # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
                        # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
                        # pred = pred_distill
                        # pred[mask_ensem] = pred_fusion[mask_ensem]
                        # logits_pred = torch.max(pred, 1)[1].detach().cpu()

                        feat_ensemble = predictions.clone().half()
                        mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                        # feat_ensemble[mask_] = feat_fuse[mask_]
                        feat_ensemble = ((feat_fuse + feat_ensemble.float()) / 2.0).half()
                        pred = feat_ensemble @ text_features.t()
                        logits_pred = torch.max(pred, 1)[1].detach().cpu()

                        # # import pdb;pdb.set_trace()
                        # ###########################################################################
                        # # feat_ensemble = predictions.clone()

                        # _, idx = pointnet_knn(args.K,
                        #                       coords[:, 1:].float().contiguous()[None, ...],
                        #                       coords[:, 1:].float().contiguous()[None, ...])
                        
                        # original_idx = torch.arange(coords.shape[0]).unsqueeze(0).unsqueeze(-1).to(idx.device)
                        
                        # # import pdb;pdb.set_trace()
                        # idx = torch.cat([idx, original_idx], dim=-1)
                        
                        # '''
                        # # [1, 20, 100, 10]
                        # new_pred_distill = grouping_operation(pred_distill.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(),
                        #                                      idx.detach())

                        # # [1, 20, 100, 10]
                        # # new_predictions = knn_find(coords[:, 1:].float().contiguous()[None, ...],
                        # #                            predictions.float()[None, ...])

                        # # [20, 100, 10]
                        # new_pred_distill = new_pred_distill[0]
                        # # [100, 20, 10]
                        # new_pred_distill = new_pred_distill.permute(1, 0, 2)
                        # # # # [100, 20]
                        # new_pred_distill = new_pred_distill.mean(dim=-1)
                        # # # [100, 1]
                        # new_pred_distill_entropy = calc_entropy(new_pred_distill, rep_i)
                        # # new_pred_distill_entropy = calc_entropy_instances(new_pred_distill)

                        # '''
                        # '''
                        # # [1, 20, 100, 10]
                        # new_pred_fusion = grouping_operation(pred_fusion.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(),
                        #                                      idx.detach())

                        # # [1, 20, 100, 10]
                        # # new_predictions = knn_find(coords[:, 1:].float().contiguous()[None, ...],
                        # #                            predictions.float()[None, ...])

                        # # [20, 100, 10]
                        # new_pred_fusion = new_pred_fusion[0]
                        # # [100, 20, 10]
                        # new_pred_fusion = new_pred_fusion.permute(1, 0, 2)
                        # # # [100, 20]
                        # new_pred_fusion = new_pred_fusion.mean(dim=-1)
                        # # # [100, 1]
                        # new_pred_fusion_entropy = calc_entropy(new_pred_fusion, rep_i)
                        # # new_pred_fusion_entropy = calc_entropy_instances(new_pred_fusion)
                        # # import pdb;pdb.set_trace()
                        # '''
                        # '''
                        # # new_pred_distill_entropy = torch.nn.functional.normalize(new_pred_distill_entropy[..., None], dim=-1)
                        # # new_pred_fusion_entropy = torch.nn.functional.normalize(new_pred_fusion_entropy[..., None], dim=-1)

                        # new_pred_distill_entropy = (new_pred_distill_entropy - new_pred_distill_entropy.min()) / (new_pred_distill_entropy.max(0 - new_pred_distill_entropy.min()))

                        # # mask_pick = new_pred_distill_entropy > new_pred_fusion_entropy
                        # # feat_ensemble[mask_pick] = feat_fuse[mask_pick].half()
                        # # import pdb;pdb.set_trace()

                        # new_pred = grouping_operation(feat_ensemble.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(),
                        #                                      idx.detach())[0]
                        
                        # new_pred = new_pred.permute(1, 0, 2).contiguous()

                        # uncertainty_3d = grouping_operation(U_3D[inds_reverse, :].unsqueeze(dim=0).permute(0, 2, 1),
                        #                                     idx.detach())[0]
                        # uncertainty_3d = uncertainty_3d.permute(1, 0, 2).contiguous()

                        # uncertainty_sem_3d = grouping_operation(new_pred_distill_entropy.unsqueeze(dim=0).unsqueeze(dim=2).permute(0, 2, 1),
                        #                                      idx.detach())[0]
                        # uncertainty_sem_3d = uncertainty_sem_3d.permute(1, 0, 2).contiguous()

                        # new_pred = new_pred * (1.0 - uncertainty_3d) * (1.0 - uncertainty_sem_3d)
                        # # import pdb;pdb.set_trace()
                        # '''
                        # new_pred = grouping_operation(feat_ensemble.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(), idx.detach())[0]
                        # feat_ensemble = new_pred.mean(dim=-1)
                        # feat_ensemble = feat_ensemble.permute(1, 0).contiguous()

                        # # new_pred_2d = grouping_operation(feat_fuse.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(), idx.detach())[0]
                        # # feat_fuse = new_pred_2d.mean(dim=-1)
                        # # feat_fuse = feat_fuse.permute(1, 0).contiguous()

                        # # feat_ensemble = feat_ensemble * new_pred_distill_entropy[:, None]
                        # # feat_fuse = feat_fuse * new_pred_fusion_entropy[:, None]
                        # feat_ensemble = ((feat_ensemble + feat_fuse) / 2.0).half()
                        # # feat_ensemble = (raw_feat_ensemble + feat_ensemble) / 2.0
                        # # feat_ensemble = (torch.min(feat_ensemble.float(), feat_fuse)).half()
                        # # shower curtain : idx 15
                        # # import pdb;pdb.set_trace()
                        # pred = feat_ensemble @ text_features.t()
                        
                        # # # import pdb;pdb.set_trace()
                        # # uncertainty_sem_3d = grouping_operation(new_pred_distill_entropy.unsqueeze(dim=0).unsqueeze(dim=2).permute(0, 2, 1),
                        # #                                      idx.detach())[0]
                        # # uncertainty_sem_2d = grouping_operation(new_pred_fusion_entropy.unsqueeze(dim=0).unsqueeze(dim=2).permute(0, 2, 1),
                        # #                                      idx.detach())[0]
                        # # uncertainty_3d = grouping_operation(uncertainty_pred.unsqueeze(dim=0).permute(0, 2, 1),
                        # #                                     idx.detach())[0]
                        
                        
                        # # # import pdb;pdb.set_trace()
                        # # new_pred = grouping_operation(pred.unsqueeze(dim=0).permute(0, 2, 1).contiguous().float(),
                        # #                                      idx.detach())[0]
                        # # # import pdb;pdb.set_trace()
                        # # # uncertainty_sem = F.normalize(uncertainty_sem, dim=-1)
                        # # uncertainty_sem = (uncertainty_sem - uncertainty_sem.min()) / (uncertainty_sem.max() - uncertainty_sem.min())
                        # # # uncertainty_3d = F.normalize(uncertainty_3d, dim=-1)
                        # # # final_uncertain = uncertainty_sem * uncertainty_3d
                        # # final_uncertain = uncertainty_sem * uncertainty_3d
                        # # # final_uncertain = F.normalize(final_uncertain, dim=-1)
                        # # final_pred = ((1.0 - final_uncertain) * new_pred.contiguous()).mean(dim=-1)
                        # # # final_pred = (new_pred).mean(dim=-1)
                        # # # final_pred = new_pred.max(dim=-1)[0]
                        # # # final_pred = ((uncertainty_sem) * new_pred.contiguous()).mean(dim=-1)
                        # # # final_pred = ((1.0 - uncertainty_3d) * new_pred).mean(dim=-1)
                        # # # final_pred

                        # # # import pdb;pdb.set_trace()
                        # # pred = final_pred.permute(1, 0).contiguous()
                        # # pred = pred.half() @ text_features.t()

                        # # pred = torch.max(pred, final_pred)
                        # # pred = feat_ensemble @ text_features.t()
                        # logits_pred = torch.max(pred, 1)[1].detach().cpu()
                        # ###########################################################################

                        predictions = feat_ensemble # if we need to save the features
                    else:
                        raise NotImplementedError

                    if args.save_feature_as_numpy:
                        scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                        np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())
                    
                    # Visualize the input, predictions and GT

                    # special case for nuScenes, evaluation points are only a subset of input
                    if 'nuscenes' in labelset_name:
                        label_mask = (label!=255)
                        label = label[label_mask]
                        logits_pred = logits_pred[label_mask]
                        pred = pred[label_mask]
                        if vis_pred:
                            pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]

                    # # if vis_input
                    # # np.save(os.path.join(save_folder, '{}_raw_pred.npy'.format(i)), predictions.cpu().numpy())
                    # np.save(os.path.join(save_folder, '{}_text_pred.npy'.format(i)), pred.cpu().numpy())
                    # np.save(os.path.join(save_folder, '{}_pred_cosine_loss.npy'.format(i)), cosine_loss.cpu().numpy())
                    # np.save(os.path.join(save_folder, '{}_pred_pred_loss.npy'.format(i)), pred_loss.cpu().numpy())
                    # np.save(os.path.join(save_folder, '{}_raw_U3D_pred.npy'.format(i)), orig_u_3d.cpu().numpy())
                    # np.save(os.path.join(save_folder, '{}_mask_matching.npy'.format(i)), mask[inds_reverse].cpu().numpy())

                    save = False

                    if save:
                        input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                        export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

                    # if vis_pred:
                    if save:
                        if mapper is not None:
                            pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
                            export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                        else:
                            pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
                            export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                            visualize_labels(list(np.unique(logits_pred.numpy())),
                                        labelset,
                                        palette,
                                        os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

                    # Visualize GT labels
                    # if vis_gt:
                    if save:
                        # for points not evaluating
                        ori_label = label.clone()
                        ori_label[ori_label==255] = len(labelset)-1
                        gt_label_color = convert_labels_with_palette(ori_label.cpu().numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
                        visualize_labels(list(np.unique(ori_label.cpu().numpy())),
                                    labelset,
                                    palette,
                                    os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

                        if 'nuscenes' in labelset_name:
                            all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                            labelset = list(NUSCENES_LABELS_16)
                            labelset[4] = 'construct. vehicle'
                            labelset[10] = 'road'
                            visualize_labels(list(all_digits), labelset, 
                                palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=all_digits.shape[0])

                    if eval_iou:
                        if mark_no_feature_to_unknown:
                            if "nuscenes" in labelset_name: # special case
                                masks.append(mask[inds_reverse][label_mask])
                            else:
                                masks.append(mask[inds_reverse])

                        if args.test_repeats==1:
                            # save directly the logits
                            preds.append(logits_pred)
                        else:
                            # only save the dot-product results, for repeat prediction
                            preds.append(pred.cpu())

                        gts.append(label.cpu())

                if eval_iou:
                    gt = torch.cat(gts)
                    pred = torch.cat(preds)

                    pred_logit = pred
                    if args.test_repeats>1:
                        pred_logit = pred.float().max(1)[1]

                    if mapper is not None:
                        pred_logit = mapper[pred_logit]

                    if mark_no_feature_to_unknown:
                        mask = torch.cat(masks)
                        pred_logit[~mask] = 256

                    if args.test_repeats==1:
                        current_iou, current_acc = metric.evaluate(pred_logit.numpy(),
                                                                gt.numpy(),
                                                                dataset=labelset_name,
                                                                stdout=True,
                                                                return_both=True)
                        # import pdb;pdb.set_trace()
                        # assert len(current_iou) == 1
                        # assert len(current_acc) == 1
                        bin_seg_mious.append(current_iou)
                        bin_seg_macc.append(current_acc)
                        
                    if args.test_repeats > 1:
                        store = pred + store
                        store_logit = store.float().max(1)[1]
                        if mapper is not None:
                            store_logit = mapper[store_logit]

                        if mark_no_feature_to_unknown:
                            store_logit[~mask] = 256
                        accumu_iou, accumu_acc = metric.evaluate(store_logit.numpy(),
                                                                gt.numpy(),
                                                                stdout=True,
                                                                dataset=labelset_name)
                        import pdb;pdb.set_trace()

        # print(bin_seg_mious)
        # print(bin_seg_macc)
        # import pdb;pdb.set_trace()
        plt.figure()
        bin_vs = list(range(len(bin_values)))
        plt.plot(bin_vs, 
                 bin_seg_mious, 
                 c='red', 
                 label="mIoU",
                 linewidth=3)
        plt.plot(bin_vs, 
                 bin_seg_macc, 
                 c='blue', 
                 label="mAcc",
                 linewidth=3)
        plt.xlabel("Uncertainty Bin Quantization [0.0-1.0]", fontsize=12)
        plt.ylabel("Eval mIoU (%)", fontsize=12)
        
        
        # plt.show()
        # plt.savefig("U3D_mIoU_20240122.png", dpi=512)
        bin_vs_labels = [
            '0.00-0.05','0.05-0.10','0.10-0.15','0.15-0.20','0.20-0.25',
            '0.25-0.30','0.30-0.35','0.35-0.40','0.40-0.45','0.45-0.50',
            '0.50-0.55','0.55-0.60','0.60-0.65','0.65-0.70','0.70-0.75',
            '0.75-0.80','0.80-0.85','0.85-0.90','0.90-0.95','0.95-1.00',
        ]
        plt.xticks(bin_vs, labels=bin_vs_labels, rotation=45, fontsize=8)
        plt.title("Uncertainty Calibration from ScanNet20 to ScanNet200", fontsize=12)
        plt.xlabel("Uncertainty Bin Quantization [0.0-1.0]", fontsize=12)
        plt.legend(loc='upper right')
        plt.savefig("scannet20_lseg_to_scannet200_val.png", dpi=512)
        
if __name__ == '__main__':
    main()
