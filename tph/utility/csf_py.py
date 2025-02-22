import CSF
import numpy as np

def csf_py(pcd, 
           return_non_ground: str,
           bsloopSmooth:bool, 
           cloth_res: float,
           threshold:float = 0.5,
           rigidness:int = 3,
           time_step:float = 0.65,
           iterations:int = 500
           ):
    # Load point cloud to CSF
    xyz = np.asarray(pcd.points)[:,:3]
    csf = CSF.CSF()
    csf.setPointCloud(xyz)
    
    # Parameter settings
    # Must Set
    csf.params.bSloopSmooth = bsloopSmooth
    csf.params.cloth_resolution = cloth_res
    csf.params.class_threshold = threshold
    
    # CAN REMAIN DEFAULT
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.iterations = iterations
    
    # Do filtering
    grd_ind, non_grd_ind = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(grd_ind, non_grd_ind)
    
    if return_non_ground == 'ground':
        return pcd.select_by_index(grd_ind)
    elif return_non_ground == 'non_ground':
        return pcd.select_by_index(non_grd_ind)
    else:
        return pcd.select_by_index(grd_ind), pcd.select_by_index(non_grd_ind)
    # Return non_ground or return ground pointcloud
    # return pcd.select_by_index(non_grd_ind) if return_non_ground\
    #     else pcd.select_by_index(grd_ind)