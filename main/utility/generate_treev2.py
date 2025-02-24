from .pcd2img import *
from .get_coords import *
from .generate_tree import get_h_from_each_tree_slice, get_tree_from_coord
from .yolo_detect import Detect
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import math
import statistics
from scipy.cluster.vq import kmeans2, kmeans
from sklearn.cluster import DBSCAN
from .csf_py import csf_py
"""
1. Bounding Box Done
2. Perform object detection Done
3. Do stuff if object detection is successful
    - CSF Filter
    - Find the center via clustering of points x,y
    - x,y radius removal
4. CSF filter
5. Reconstruct Tree
    - Visualize
6. Separate to Cylinder and 
"""
# crop_pcd_to_many
# get_h_from_each_tree_slice
def crop_tree_for_obj_det():
    pass


# Under the assumption that the library works
def find_centroid_from_Trees(grd_pcd, coord:tuple, radius_expand:int=3, zminmax:list=[-15,15], iters:int=0):
    xc, yc = coord[0], -coord[1]
    ex = radius_expand
    zmin, zmax = zminmax
    min_bound = (xc-ex, yc-ex, zmin)
    max_bound = (xc+ex, yc+ex, zmax)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    tree_with_gnd = grd_pcd.crop(bbox)
    tree_with_gnd = tree_with_gnd.remove_non_finite_points()
    if len(np.asarray(tree_with_gnd.points)) < 1000:
        return None
    grd, non_grd = csf_py(
        tree_with_gnd, 
        return_non_ground = "both", 
        bsloopSmooth = True, 
        cloth_res = 15.0, 
        threshold= 2.0, 
        rigidness=2,
        iterations=1000
    )
    xyz = np.asarray(non_grd.points)
    if not xyz.size:
        return None
    z_min_non_gnd = xyz[:,2].min()
    non_grd = non_grd.select_by_index(np.where(xyz[:,2]<z_min_non_gnd+2)[0])
    xyz = np.asarray(non_grd.points)
    xyz = xyz[:, np.isfinite(xyz).any(axis=0)]
    if not xyz.size:
        return None
    else:
        try:
            centroid, label_ = kmeans(xyz[:,0:2],k=1)
            xnew,ynew = centroid[0]
        except:
            print(f"is finite {np.isfinite(xyz.all())}")

        if iters < 1:
            return find_centroid_from_Trees(grd_pcd, (xnew, -ynew), 2, zminmax, iters+1)
        else:
            return (xnew, -ynew)
    # if centroid is None:
    #     return None
    # xnew,ynew = centroid[0]
    # if iters < 1:
    #     find_centroid_from_Trees(grd_pcd, (xnew, -ynew), 2, zminmax, iters+1)
    # else:
    #     return (xnew, -ynew)

def regenerate_Tree(grd_pcd, center_coord:tuple, radius_expand:int=5, zminmax:list=[-15,15]):
    xc, yc = center_coord[0], -center_coord[1]
    ex = radius_expand
    zmin, zmax = zminmax
    min_bound = (xc-ex, yc-ex, zmin)
    max_bound = (xc+ex, yc+ex, zmax)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    tree_with_gnd = grd_pcd.crop(bbox)
    distances = np.linalg.norm(np.asarray(tree_with_gnd.points)[:,0:2] - np.array([xc, yc]), axis=1)
    tree_with_gnd = tree_with_gnd.select_by_index(np.where(distances<=radius_expand)[0])
    grd, non_grd = csf_py(
        tree_with_gnd, 
        return_non_ground = "both", 
        bsloopSmooth = True, 
        cloth_res = 15.0, 
        threshold= 2.0, 
        rigidness=2,
        iterations=1000
    )     
    o3d.cuda.pybind.visualization.draw_geometries([non_grd])
    
class TreeGen():
    def __init__(self, yml_data, sideViewOut, pcd_name):
        self.pcd_name = pcd_name
        self.min_points_per_tree = 1500
        self.sideViewOut = sideViewOut
        
        side_view_model_pth = yml_data["yolov5"]["sideView"]["model_pth"]
        self.side_view_step_size = yml_data["yolov5"]["sideView"]["stepsize"]
        self.side_view_img_size = tuple(yml_data["yolov5"]["sideView"]["imgSize"])
        self.side_view_img_size_tall = tuple(yml_data["yolov5"]["sideView"]["imgSizeTall"])
        self.ex_w, self.ex_h = (dim*self.side_view_step_size for dim in self.side_view_img_size)
        min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]
        yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
        self.obj_det_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size)
        self.obj_det_tall = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size_tall)
        
    
    def process_each_coord(self, pcd, grd_pcd, coords, w_lin_pcd, h_lin_pcd):
        # Init
        h_arr_pcd, h_increment = h_lin_pcd
        w_arr_pcd, w_increment = w_lin_pcd
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        
        coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
        for index, coord in enumerate(coord_loop):
            n_detected = 0
            confi_list = []
            coord_list = []
            h_im_list = []
            
            # Split each coord to multi-sections and find the one with highest confidence
            h_loop = h_arr_pcd[:-1] 
            w_loop = w_arr_pcd[:-1]
            coord = find_centroid_from_Trees(pcd,coord,3, [z_min, z_max])
            if coord is None:
                continue
            for i, h in enumerate(h_loop):
                for j,w in enumerate(w_loop):
                    min_x, max_x = w, w+w_increment+w_increment/4
                    min_y, max_y = h, h+h_increment+h_increment/4 
                    minbound = (min_x, min_y, z_min)
                    maxbound = (max_x, max_y, z_max)
                    coords_x_bool = (coord[0] >= min_x) & (coord[0] <= max_x)
                    coords_y_bool = (-coord[1] >= min_y) & (-coord[1] <= max_y)
                    
                    new_x, new_y = statistics.mean([min_x, max_x]), statistics.mean([min_y, max_y])
                    new_coord = (new_x, new_y)
                    if coords_x_bool & coords_y_bool:
                        section_tree_pcd = pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
                        section_grd_pcd = grd_pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
                        almost_tree = get_tree_from_coord(pcd, grd_pcd, coord, expand_x_y=[self.ex_w,self.ex_w], expand_z=[z_min, z_max])
                        h, im , confi = get_h_from_each_tree_slice(
                            tree = almost_tree,
                            model_short = self.obj_det_short,
                            model_tall = self.obj_det_tall,
                            img_size = self.side_view_img_size, 
                            stepsize = self.side_view_step_size,
                            img_dir = f"{self.sideViewOut}/{self.pcd_name}_{index}_",
                            gen_undetected_img = False,
                            img_with_h = True,
                            min_no_points = self.min_points_per_tree
                            )
                        if h > 0:
                            confi_list.append(confi)
                            coord_list.append(coord)
                            h_im_list.append(im)
                            n_detected += 1
                        
            if n_detected <= 0:
                continue
            else:
                print("h_detected",h>0)
                # Perform Operations
                # new_coord = find_centroid_from_Trees(pcd,coord_list[0],3, [z_min, z_max])
                regenerate_Tree(pcd, coord, 5, [z_min, z_max])