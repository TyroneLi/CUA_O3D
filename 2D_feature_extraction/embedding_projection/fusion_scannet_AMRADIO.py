import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
# add the parent directory to the path
import sys
sys.path.append('..')

from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from encoding.models.sseg import BaseNet
import torchvision.transforms as transforms

from fusion_util import (extract_lseg_img_feature, 
                         PointCloudToImageMapper, 
                         save_fused_feature_with_locs, 
                         adjust_intrinsic, 
                         make_intrinsic, 
                         save_fused_feature_with_locs_noAligned,
                         save_fused_feature,
                         sameAs_save_fused_feature_scannet)


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of LSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--scan_dir', type=str, default="/home/jinlong.li/datasets/ScanNetV2/scannetv2_download/scans", help='')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='')
    parser.add_argument('--lseg_model', 
                        type=str, 
                        default='./lexicon3d/demo_e200.ckpt', help='Where is the LSeg checkpoint')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='AMRADIO', help='prefix for the output file')
    parser.add_argument('--save_aligned', type=bool, default=True, help='whether to save aligned features')
    

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):

    # short hand
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    keep_features_in_memory = args.keep_features_in_memory
    evaluator = args.evaluator
    # transform = args.transform


    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    if exists(join(out_dir, args.prefix+'_features', scene_id + '.pt')):
        print(scene_id +'.pt' + ' already exists, skip!')
        return 1

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and evaluator is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_lseg_img_feature(img_dir, transform, evaluator))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):

        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0:
            # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        
        image = Image.open(img_dir).convert('RGB')
        image = pil_to_tensor(image).to(dtype=torch.float32, device='cuda')
        image.div_(255.0)
        image = image.unsqueeze(dim=0)
        
        with torch.no_grad():
            nearest_res = evaluator.get_nearest_supported_resolution(*image.shape[-2:])
            image = F.interpolate(image, nearest_res, mode='bilinear', align_corners=False) 
            summary, spatial_features = evaluator(image, feature_fmt='NCHW')
            assert spatial_features.ndim == 4
            feat_2d = spatial_features

        feat_2d = torch.nn.functional.interpolate(
            feat_2d, 
            size=(240, 320), 
            mode='bicubic', 
            align_corners=False).squeeze(0) # 240, 320, 1024
        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)

        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    
    sameAs_save_fused_feature_scannet(feat_bank, point_ids, n_points, out_dir, scene_id, args)


def main(args):   
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    img_dim = (320, 240)
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx=319.5
    my=239.5
    #######################################
    # threshold for the visibility check
    visibility_threshold = 0.25

    args.depth_scale = depth_scale
    # do not use the features on the image boundary
    args.cut_num_pixel_boundary = 10
    # keep image features in the memory, very expensive
    args.keep_features_in_memory = False

    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'scannet_3d')
    data_root_2d = join(data_dir,'scannet_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    args.feat_dim = 1024 # AMROIDIO feature dimension
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    # for training set, export a chunk of point cloud
    if split== 'train':
        args.n_split_points = 300000
        args.num_rand_file_per_scene = 5
    else:
        # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000
        args.num_rand_file_per_scene = 1


    # model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
    model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
    #model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
    #model_version="e-radio_v2" # for E-RADIO
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
    model.cuda().eval()

    args.evaluator = model

    # calculate image pixel-3D points correspondances
    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim)


    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)
