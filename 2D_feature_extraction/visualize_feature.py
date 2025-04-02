import os
import sys
import time
import numpy as np
import torch
import open3d as o3d
from open3d.visualization import rendering
import imageio
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster
import umap

def visualize_pc(pcd, pc_feat, out_dir_root, out_dir_name, scene_id):
    out_dir = os.path.join(out_dir_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # visualize pcd with color using open3d
    mtl_points = o3d.visualization.rendering.MaterialRecord()
    mtl_points.shader = "defaultUnlit"
    mtl_points.point_size = 4

    # render.scene.add_geometry('point cloud', pcd, mtl_points)
    cam_dist = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8

    if scene_id in ['scene0031_00', 'scene0031_01']:
        pitch, yaw = 45, 180
    else:
        pitch, yaw = 45, 0

    cam_x = cam_dist * np.cos(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
    cam_y = cam_dist * np.cos(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
    cam_z = cam_dist * np.sin(np.deg2rad(pitch))
    # render.setup_camera(60.0, pcd.get_center(), pcd.get_center()+[cam_x, cam_y, cam_z], [0, 0, 1])

    # render the image and save to out_dir
    # img = render.render_to_image()
    # img = np.array(img)
    # img = img[:, :, :3]
    # img = img.astype(np.uint8)
    # img_path = os.path.join(out_dir, scene_id + '.png')
    # imageio.imwrite(img_path, img)
    # print(f"Saved {img_path}")
    # remove the point cloud from the scene
    # render.scene.remove_geometry('point cloud')


def visualize_pca(pcd, pc_feat, out_dir_root, out_dir_name, scene_id):
    out_dir = os.path.join(out_dir_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # visualize pcd with color using open3d
    mtl_points = o3d.visualization.rendering.MaterialRecord()
    mtl_points.shader = "defaultUnlit"
    mtl_points.point_size = 4

    if scene_id in ['scene0031_00', 'scene0031_01']:
        pitch, yaw = 45, 180
    else:
        pitch, yaw = 45, 0

    cam_dist = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8
    cam_x = cam_dist * np.cos(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
    cam_y = cam_dist * np.cos(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
    cam_z = cam_dist * np.sin(np.deg2rad(pitch))

    # Visualize all 6 combinations of orderings of the pcd.colors (RGB, RBG, GRB, GBR, BRG, BGR)
    for i, order in enumerate([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]):
        pcd.colors = o3d.utility.Vector3dVector(pc_feat[:, order])
        # render.scene.add_geometry('point cloud', pcd, mtl_points)
        # render.setup_camera(60.0, pcd.get_center(), pcd.get_center()+[cam_x, cam_y, cam_z], [0, 0, 1])

        # render the image and save to out_dir
        # img = render.render_to_image()
        # img = np.array(img)
        # img = img[:, :, :3]
        # img = img.astype(np.uint8)
        # img_path = os.path.join(out_dir, scene_id + f'_order_{i}.png')
        # imageio.imwrite(img_path, img)
        # print(f"Saved {img_path}")
        # remove the point cloud from the scene
        # render.scene.remove_geometry('point cloud')


def main():
    w, h = 512, 512

    # prefix = 'clip'
    # prefix = 'dinov2'
    prefix = sys.argv[1]
    
    pc_dir =        './datasets/scannet_marzola_noAligned/{prefix}_points'.format(prefix=prefix) 
    feat_dir =      './datasets/scannet_marzola_noAligned/{prefix}_features'.format(prefix=prefix)
    scan_dir =      '/Volumes/Expansion/datasets/scannet/scannetv2_download/scans'
    out_dir_root =  './datasets/scannet_marzola_noAligned/{prefix}_visualization'.format(prefix=prefix)
    os.makedirs(out_dir_root, exist_ok=True)

    # scene_ids = ['scene0000_00']
    # scene_ids = ['scene0000_02']
    scene_ids = ['scene0109_00']
    # scene_ids = ['scene0449_00']
    
    # for scene_id in scene_ids:

    pc_pos = np.load(sys.argv[1])
    # print(pc_pos)
    # import pdb;pdb.set_trace()
    # pc_feat = torch.load(os.path.join(feat_dir, scene_id + '_0.pt'))
    pc_feat = torch.load(sys.argv[2])
    print("feat shape : ", pc_feat['feat'].shape)

    pc_pos_aligned = pc_pos[:, :3]

    # set the color to (2) the kmeans cluster of the features, using different colors for different clusters
    # num_clusters = 10
    num_clusters = int(sys.argv[3])
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(pc_feat['feat'].numpy())
    # import pdb;pdb.set_trace()
    clusters = kmeans.labels_ # cluster index for each point
    unique_colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_clusters))[:, :3]
    # unique_colors = plt.get_cmap('rainbow')(np.linspace(0, 1, num_clusters))[:, :3]
    cluster_colors = np.array([unique_colors[cluster] for cluster in clusters])
    pc_with_feature_kmeans = o3d.geometry.PointCloud()
    pc_with_feature_kmeans.points = o3d.utility.Vector3dVector(pc_pos_aligned)
    pc_with_feature_kmeans.colors = o3d.utility.Vector3dVector(cluster_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200,height=800)
    vis.add_geometry(pc_with_feature_kmeans)
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([pc_with_feature_kmeans])
    
    # set the color to (3) the pca of the features
    umap_t = umap.UMAP(n_components=3)
    umap_feat = umap_t.fit_transform(pc_feat['feat'].numpy())
    umap_feat = (umap_feat - umap_feat.min(axis=0)) / (umap_feat.max(axis=0) - umap_feat.min(axis=0))
    pc_with_feature_umap = o3d.geometry.PointCloud()
    pc_with_feature_umap.points = o3d.utility.Vector3dVector(pc_pos_aligned)
    pc_with_feature_umap.colors = o3d.utility.Vector3dVector(umap_feat)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200,height=800)
    vis.add_geometry(pc_with_feature_umap)
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([pc_with_feature_umap])


if __name__ == '__main__':
    main()
    