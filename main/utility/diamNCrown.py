import sys
sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm/src')

import numpy as np
import open3d as o3d

# Personal libs
import adTreeutils.tree_utils as tree_utils
import adTreeutils.o3d_utils as o3d_utils
from labels import Labels
from config import Paths
from .o3d_extras import dbscan

class AdTree_cls():
    def __init__(self):
        self.adTree_exe = Paths.get_adtree()
        
    def reconstruct_skeleton(self, tree_cloud):
        labels = self.leafwood_classificiation(tree_cloud, method='surface_variation')
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        skeleton = tree_utils.reconstruct_skeleton(wood_cloud, self.adTree_exe)
        tree_utils.show_tree(tree_cloud, labels, skeleton)
    
    def surface_variation_filter(self, pcd, radius, threshold):
        """Compute surface variation of point cloud."""
        pcd.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=5))
        eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
        eig_val = np.sort(eig_val, axis=1)
        sv = eig_val[:,0] / eig_val.sum(axis=1)
        mask = sv < threshold
        return mask
    
    def leafwood_classificiation(self, tree_cloud, method):
        """Leaf-wood classification."""

        labels = np.full(len(tree_cloud.points), Labels.LEAF, dtype=int)

        # outlier removal
        pcd_, _, trace = tree_cloud.voxel_down_sample_and_trace(0.02,
                                        tree_cloud.get_min_bound(),
                                        tree_cloud.get_max_bound())
        pcd_, ind_ = pcd_.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
        ind_ = np.asarray(ind_)

        # classify
        if method == 'curvature':
            mask = o3d_utils.curvature_filter(pcd_, .075, min1=20, min2=35)
            ind = np.hstack([trace[i] for i in ind_[mask]])
        else:
            mask = self.surface_variation_filter(pcd_, .74, .3)
            ind = np.hstack([trace[i] for i in ind_[mask]])

        labels[ind] = Labels.WOOD

        return labels
    
    def segment_tree(self, tree_cloud):
        stem_cloud, crown_cloud = self.tree_separate(tree_cloud, self.adTree_exe, filter_leaves="surface_variation")
        # crown_mesh_hull, volume = tree_utils.crown_to_mesh(crown_cloud, method='alphashape')
        # stem_mesh_hull, volume = tree_utils.crown_to_mesh(stem_cloud, method='alphashape')
        o3d.visualization.draw_geometries([stem_cloud])
        o3d.visualization.draw_geometries([crown_cloud])
        
    def tree_separate(self, tree_cloud, adTree_exe, filter_leaves="surface_variation"):
        """Function to split stem from o3d tree point cloud."""

        # 1. Classify and filter leaves (optional)
        labels = np.ones(len(tree_cloud.points), dtype=int)
        labels = self.leafwood_classificiation(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])

        # 2. Skeleton reconstruction
        print("Reconstructing tree skeleton...")
        skeleton = tree_utils.reconstruct_skeleton(wood_cloud, adTree_exe)
        print("num_vertices",len(skeleton['vertices']))
        print("len_graph", len(skeleton['graph']))
        tree_utils.show_tree(tree_cloud, labels, skeleton)
        # 3. Stem-crow splitting
        print("Splitting stem form crown...")
        mask = tree_utils.skeleton_split(tree_cloud, skeleton['graph'])
        labels[mask] = Labels.STEM
        print(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

        stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
        crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)

        return stem_cloud, crown_cloud
    
    def separate_via_dbscan(self, tree_cloud):
        dbscan(tree_cloud)
        
    