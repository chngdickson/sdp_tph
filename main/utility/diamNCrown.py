import sys
sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm')

import numpy as np
import open3d as o3d
import src.utils.tree_utils as tree_utils
import src.utils.o3d_utils as o3d_utils
from labels import Labels
from config import Paths
adTree_exe = Paths.get_adtree()


class AdTree_cls():
    def __init__(self):
        self.adTree_exe = Paths.get_adtree()
        
    def reconstruct_skeleton(self, tree_cloud):
        labels = tree_utils.leafwood_classificiation(tree_cloud, method='curvature')
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        skeleton = tree_utils.reconstruct_skeleton(wood_cloud, self.adTree_exe)
        tree_utils.show_tree(tree_cloud, labels, skeleton)
    
    def segment_tree(self, tree_cloud):
        pass