import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# Method
# PFPH to get the features
# Ransac to get the features on the tree

def dbscan(outlier_cloud, eps=0.5, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
    ## eps = 0.4 min_points=7
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return outlier_cloud , labels