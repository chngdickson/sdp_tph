import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# Method
# PFPH to get the features
# Ransac to get the features on the tree

def dbscan(tree_cloud, eps=0.5, min_points=10):
    tree_cloud = tree_cloud.voxel_down_sample(0.04)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            tree_cloud.cluster_dbscan(eps=1.0, min_points=10, print_progress=True))

    max_label = labels.max()

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    tree_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([tree_cloud])
    

def save_pointcloud(tree_cloud, filename):
    o3d.io.write_point_cloud(filename, tree_cloud)