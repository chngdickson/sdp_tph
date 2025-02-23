import sys
sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm/src')

import numpy as np
import open3d as o3d
import utils.tree_utils as tree_utils
import utils.o3d_utils as o3d_utils
from labels import Labels
from config import Paths
adTree_exe = Paths.get_adtree()

def reconstruct_skeleton()