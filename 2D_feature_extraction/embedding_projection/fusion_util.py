import os
import torch
import glob
import math
import numpy as np
from PIL import Image

def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def extract_lseg_img_feature(img_dir, transform, evaluator, label=''):
    # load RGB image
    image = Image.open(img_dir)
    image = np.array(image)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = evaluator.parallel_forward(image, label)
        feat_2d = outputs[0][0].half()

    return feat_2d


def pc2voxel():
    pc_dir = 'dataset/lseg_features'
    scan_dir = 'dataset/ScanNet/scans'
    save_dir = 'dataset/lseg_voxels'
    voxel_size = 0.05

    os.makedirs(save_dir, exist_ok=True)
    
    pc_pos_aligned_lengths = []
    for id, scene_id in enumerate(os.listdir(pc_dir)):
        if id % 10 == 0:
            print('Processing %d-th scene...' % id)
        pc_pos = torch.load(os.path.join(pc_dir, scene_id, 'pcd_pos.pt'))
        pc_pos = np.array(pc_pos)
        meta_file = open(os.path.join(scan_dir, scene_id, scene_id + '.txt'), 'r').readlines()
        axis_align_matrix = None
        for line in meta_file:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        if axis_align_matrix != None:
            axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        axis_align_matrix = axis_align_matrix if axis_align_matrix is not None else np.eye(4)

        pc_pos_4 = np.concatenate([pc_pos, np.ones((pc_pos.shape[0], 1))], axis=1)
        pc_pos_aligned = pc_pos_4 @ axis_align_matrix.transpose()
        pc_pos_aligned = pc_pos_aligned[:, :3]
        if pc_pos_aligned.shape[0] != 0:
            pc_pos_aligned = voxelize_pc(pc_pos_aligned, voxel_size)

        # save voxelized point cloud
        np.save(os.path.join(save_dir, scene_id + '.npy'), pc_pos_aligned)


def voxelize_pc(pc_pos_aligned, voxel_size=0.05):
    '''pc_pos_aligned: array [3]'''
    # translate point with smallest coordinate to origin
    pc_pos_aligned = pc_pos_aligned - pc_pos_aligned.min(axis=0)
    # voxelization 
    pc_pos_aligned = np.floor(pc_pos_aligned / voxel_size)
    return pc_pos_aligned


def save_fused_feature_with_locs(feat_bank, point_ids, locs_in, n_points, out_dir, scene_id, args):
    '''Save features and locations and aligned voxels.'''

    if n_points < args.n_split_points:
        n_points_cur = n_points  # to handle point cloud numbers less than n_split_points
    else:
        n_points_cur = args.n_split_points

    rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

    mask_entire = torch.zeros(n_points, dtype=torch.bool)
    mask_entire[rand_ind] = True
    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True
    mask_entire = mask_entire & mask

    # read in axis alignment matrix
    meta_file = open(os.path.join(args.scan_dir, scene_id, scene_id + '.txt'), 'r').readlines()
    axis_align_matrix = None
    for line in meta_file:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    axis_align_matrix = axis_align_matrix if axis_align_matrix is not None else np.eye(4)

    pcd_pos = locs_in[mask_entire]
    pcd_pos_4 = np.concatenate([pcd_pos, np.ones((pcd_pos.shape[0], 1))], axis=1)
    pc_pos_aligned = pcd_pos_4 @ axis_align_matrix.transpose()
    pc_pos_aligned = pc_pos_aligned[:, :3]
    if pc_pos_aligned.shape[0] != 0:
        pcd_pos_vox = voxelize_pc(pc_pos_aligned, args.voxel_size)
    else:
        pcd_pos_vox = np.zeros((0, 3))

    out_dir_features = os.path.join(out_dir, args.prefix+'_features')
    out_dir_voxels = os.path.join(out_dir, args.prefix+'_voxels')
    out_dir_points = os.path.join(out_dir, args.prefix+'_points')
    os.makedirs(out_dir_features, exist_ok=True)
    os.makedirs(out_dir_voxels, exist_ok=True)
    os.makedirs(out_dir_points, exist_ok=True)
    torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                "mask_full": mask_entire
    },  os.path.join(out_dir_features, scene_id+'.pt'))
    np.save(os.path.join(out_dir_voxels, scene_id+'.npy'), pcd_pos_vox)
    np.save(os.path.join(out_dir_points, scene_id+'.npy'), pcd_pos)
    print('Scene {} is saved!'.format(scene_id))



def save_fused_feature_with_locs_noAligned(feat_bank, point_ids, locs_in, n_points, out_dir, scene_id, args):
    '''Save features and locations and aligned voxels.'''

    if n_points < args.n_split_points:
        n_points_cur = n_points  # to handle point cloud numbers less than n_split_points
    else:
        n_points_cur = args.n_split_points

    rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

    mask_entire = torch.zeros(n_points, dtype=torch.bool)
    mask_entire[rand_ind] = True
    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True
    mask_entire = mask_entire & mask

    pcd_pos = locs_in[mask_entire]

    pc_pos_aligned = pcd_pos[:, :3]
    
    if pc_pos_aligned.shape[0] != 0:
        pcd_pos_vox = voxelize_pc(pc_pos_aligned, args.voxel_size)
    else:
        pcd_pos_vox = np.zeros((0, 3))

    out_dir_features = os.path.join(out_dir, args.prefix+'_features')
    out_dir_voxels = os.path.join(out_dir, args.prefix+'_voxels')
    out_dir_points = os.path.join(out_dir, args.prefix+'_points')
    os.makedirs(out_dir_features, exist_ok=True)
    os.makedirs(out_dir_voxels, exist_ok=True)
    os.makedirs(out_dir_points, exist_ok=True)
    torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                "mask_full": mask_entire
    },  os.path.join(out_dir_features, scene_id+'.pt'))
    np.save(os.path.join(out_dir_voxels, scene_id+'.npy'), pcd_pos_vox)
    np.save(os.path.join(out_dir_points, scene_id+'.npy'), pcd_pos)
    print('Scene {} is saved!'.format(scene_id))

def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')

def save_fused_feature_matterport(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')

def sameAs_save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points
        
        # TODO:
        # extract the same point indices or positions as original Lseg/OpenSeg point features 
        # need to modify as personal path
        # previous_path = "/leonardo_work/IscrC_bal/OV3D/datas/my_reExtract_scannet_lseg"
        previous_path = "/mhug/mhug-dataset/jinlong_li_datasets/data_from_yoda_2d_embedding"
        
        previous_full_path = os.path.join(previous_path, scene_id +'_%d.pt'%(n))
        assert os.path.exists(previous_full_path)
        mask_entire = torch.load(previous_full_path)['mask_full']
        
        
        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')

def sameAs_save_fused_feature_scannet(feat_bank, point_ids, n_points, out_dir, scene_id, args, previous_path=None):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        # TODO: 
        if not previous_path:
            # extract the same point indices or positions as original Lseg/OpenSeg point features
            # need to modify as personal path
            previous_path = "/leonardo_work/IscrC_bal/OV3D/datas/my_reExtract_scannet_lseg"
            # previous_path = "/nfs/datasets/jinlong_li_datasets/data_from_yoda_2d_embedding/my_reExtract_matterport_lseg"
            # previous_path = "/mhug/mhug-dataset/jinlong_li_datasets/data_from_yoda_2d_embedding/my_reExtract_matterport_lseg"
            # previous_path = "/mhug/mhug-dataset/jinlong_li_datasets/data_from_yoda_2d_embedding/re_my_extraction_scannet_lseg"
            
            previous_full_path = os.path.join(previous_path, scene_id +'_%d.pt'%(n))
            assert os.path.exists(previous_full_path)
            mask_entire = torch.load(previous_full_path)['mask_full']
            
            
            torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                        "mask_full": mask_entire
            },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
            
        else:

            rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

            mask_entire = torch.zeros(n_points, dtype=torch.bool)
            mask_entire[rand_ind] = True
            mask = torch.zeros(n_points, dtype=torch.bool)
            mask[point_ids] = True
            mask_entire = mask_entire & mask

            torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                        "mask_full": mask_entire
            },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
            
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')

def sameAs_save_fused_feature_matterport(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        # TODO: 
        # extract the same point indices or positions as original Lseg/OpenSeg point features
        # need to modify as personal path
        # previous_path = "/leonardo_work/IscrC_bal/OV3D/datas/my_reExtract_matterport_lseg"
        # previous_path = "/nfs/datasets/jinlong_li_datasets/data_from_yoda_2d_embedding/my_reExtract_matterport_lseg"
        previous_path = "/mhug/mhug-dataset/jinlong_li_datasets/data_from_yoda_2d_embedding/my_reExtract_matterport_lseg"
        previous_full_path = os.path.join(previous_path, scene_id +'_%d.pt'%(n))
        assert os.path.exists(previous_full_path)
        mask_entire = torch.load(previous_full_path)['mask_full']
        
        
        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')

class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob.glob(img_dir+'/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name+'.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name+'.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related infomation of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                    (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                    (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        #! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l+bbox_h)/2
        dist_centroid = np.linalg.norm(cam_loc-centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img