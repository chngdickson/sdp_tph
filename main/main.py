import logging

logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)
import os
import sys
import cv2 
import numpy as np
from utility.yolo_detect import Detect
from utility.pcd2img import pcd2img_np
from utility.get_coords import scale_pred_to_xy_point_cloud, draw_coord_on_img, scale_coord, get_strides
from utility.generate_tree import get_h_from_each_tree_slice, crop_pcd_to_many
from utility.csf_py import csf_py
from utility.encode_decode import img_b64_to_arr

# Standard Libraries
import yaml
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import laspy

def get_args(path_directory, input_file, input_file_type):
    logger.info(f"Inputs:\n   path_directory  : [{path_directory}]\n   input_file_name : [{input_file}]\n   input_file_type : [{input_file_type}]\n")
    input_img_pth = path_directory + input_file + input_file_type
    # output_img_pth = path_directory + input_file +"2" + input_file_type
    assert os.path.exists(input_img_pth), f"the path or file [{input_img_pth}] does not exists"
    # print("input_img_pth", input_img_pth)
    # print("output_img_pth", output_img_pth)
    return None

def main(path_directory, pcd_name, input_file_type):
    get_args(path_directory, pcd_name, input_file_type)
    
    
    #################################################
    ######## 1 File Generation from PCD #############
    #################################################
    logger.info("Step 1: Reading pcd file...")
    
    # Load Yaml
    with open("config/config.yaml","r") as ymlfile:
        yml_data = yaml.load(ymlfile, Loader = yaml.FullLoader)
    
    # Input Folder Location
    curr_dir = os.getcwd()
    folder_loc = path_directory
    pcd_filename = pcd_name+input_file_type
    
    # Output Folder Location
    output_folder = folder_loc + pcd_name +"/"
    topViewOut = output_folder + yml_data["output"]["topView"]["folder_location"]
    sideViewOut = output_folder + yml_data["output"]["sideView"]["folder_location"]
    csvOut = output_folder + pcd_name +".csv"
    
    accepted_file_types = [".las",".laz",".txt",".pcd",".ply"]
    assert input_file_type in accepted_file_types,f"Filetype must be {accepted_file_types}"
    
    # Read pcd
    if input_file_type in [".las",".laz"]:
        with laspy.open(folder_loc+pcd_filename) as fh:
            las = fh.read()
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    elif input_file_type == ".txt":
        format = 'xyz'
        pcd = o3d.io.read_point_cloud(folder_loc+pcd_filename, format=format)
    else:
        format = 'auto'
        pcd = o3d.io.read_point_cloud(folder_loc+pcd_filename, format=format)

    assert len(pcd.points) >= 1,f"Failed to Read Point Cloud file [{pcd_filename}], it's Empty or broken"

    logger.info(f"Reading {input_file_type} file successful, Generating stuff")
    for path in [output_folder, topViewOut, sideViewOut]:
        if not os.path.exists(path):
            os.mkdir(path)
    ###################################################
    ######## END File Generation from PCD #############
    ###################################################
    
    
    ##########################################    
    ######## 2 CSF and Rasterize #############
    ##########################################
    logger.info("Step 2: CSF and Rasterize")
    
    # Yaml Params
    topViewStepsize = yml_data["yolov5"]["topView"]["stepsize"]
    top_view_model_pth = yml_data["yolov5"]["topView"]["model_pth"]
    yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
    ideal_img_size = yml_data["yolov5"]["topView"]["imgSize"]

    # 1. Generate Top View Yolov5 Model
    topViewModel = Detect(yolov5_folder_pth, top_view_model_pth, img_size=ideal_img_size)

    grd, non_grd = csf_py(
        pcd, 
        return_non_ground = "both", 
        bsloopSmooth = True, 
        cloth_res = 15.0, 
        threshold= 2.0, 
        rigidness=1
    )
    # 2. Create img from CSF
    non_ground_img = pcd2img_np(non_grd,"z",topViewStepsize)
    ############################################
    ######## END CSF and Rasterize #############
    ############################################  
    

    ####################################################
    ####### 3. Get Coordinates from Top View ###########
    ####################################################
    logger.info("Step 3: Create Visualization from NN")
    
    coordinates = []

    # 1. Calculate spacing for image splitting
    h_s, w_s = get_strides(non_ground_img.shape, ideal_img_size)
    h_arr, h_incre = np.linspace(0, non_ground_img.shape[0], h_s+1, retstep=True)
    w_arr, w_incre = np.linspace(0, non_ground_img.shape[1], w_s+1, retstep=True)
    
    # 1.b Calculate spacing for PCD splitting.
    x_min_pcd, y_min_pcd, z_min = non_grd.get_min_bound()
    x_max_pcd, y_max_pcd, z_max = non_grd.get_max_bound()
    h_arr_pcd , h_incre_pcd = np.linspace(y_min_pcd, y_max_pcd, h_s+1, retstep=True)
    w_arr_pcd , w_incre_pcd = np.linspace(x_min_pcd, x_max_pcd, w_s+1, retstep=True)
    # logger.info(f"\ny_min: [{y_min}]\ny_max: [{y_max}]\nh_arr_pcd: {h_arr_pcd}")
    # logger.info(f"\nx_min: [{x_min}]\nx_max: [{x_max}]\nw_arr_pcd: {w_arr_pcd}\n\n")
    
    img_shape = non_ground_img.shape
    # 2. Split images 
    for i, h in enumerate(h_arr[:-1]):
        for j, w in enumerate(w_arr[:-1]):
            img = non_ground_img[int(round(h)):int(round(h+h_incre+h_incre/4)), int(round(w)):int(round(w+w_incre+w_incre/4))]
            preds = topViewModel.predict(
                img,
                convert_to_gray=False,
                confi_thres = 0.13,
                iou_thres = 0.02
                )
            coordinates.extend(scale_pred_to_xy_point_cloud(preds, 1, w, h))
            del img, preds

    logger.info("Step 3.1: Performing Clustering")
    # 2.b Remove extra coordinates generated on step 2.a via clustering
    total_dist = 0
    coordinates = np.array(coordinates)
    tree = KDTree(coordinates[:,0:2]) #Location 1
    for i in range(len(coordinates)):
        distances , _ = tree.query(coordinates[i,0:2], k=2, workers=-1) # Location2
        total_dist += distances[1]
    mean_dist = total_dist/len(coordinates)

    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=mean_dist, compute_distances=True)
    clustering.fit(coordinates)

    true_coordinates = []
    for i in range(max(clustering.labels_)):
        each_cluster = np.where(clustering.labels_==i)[0]
        n = len(each_cluster)
        if n<2:
            true_coordinates.append(coordinates[each_cluster[0]][0:2])
        else:
            pts_in_cluster = [coordinates[ind] for ind in each_cluster]
            pts_in_cluster = np.vstack(pts_in_cluster)
            center = pts_in_cluster[np.where(pts_in_cluster[:,2].max())][0][0:2]
            #center = np.mean(pts_in_cluster, axis=0)
            true_coordinates.append(center)
    coordinates = np.vstack(true_coordinates)
    del true_coordinates

    # 2c Visualization Purpose
    img_with_coord = draw_coord_on_img(non_ground_img, np.asarray(coordinates), circle_size=10)
    cv2.imwrite(f"{topViewOut}/{pcd_name}_coor.png", img_with_coord)

    # 3. Scale 2D to 3D
    xmin, ymin, zmin = non_grd.get_min_bound()
    xmax, ymax, zmax = non_grd.get_max_bound()
    range_x, range_y, range_z = xmax-xmin, ymax-ymin, zmax-zmin

    height, width = non_ground_img.shape
    coordinates = scale_coord(
        coordinates, 
        scale=(range_x/width, range_y/height), 
        offset=(xmin,-ymax)
        )

    # 4. Clear unused memory
    del topViewModel
    del non_grd
    ####################################################
    ##### END  Get Coordinates from Top View ###########
    ####################################################
    
    
    ####################################################
    ####### 4. Generate Height from Each Tree ##########
    ####################################################
    logger.info("Step 4. Generate Height ")
    
    # Yaml Params
    side_view_model_pth = yml_data["yolov5"]["sideView"]["model_pth"]
    side_view_step_size = yml_data["yolov5"]["sideView"]["stepsize"]
    side_view_img_size = tuple(yml_data["yolov5"]["sideView"]["imgSize"])
    side_view_img_size_tall = tuple(yml_data["yolov5"]["sideView"]["imgSizeTall"])
    min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]

    # Init SideViewYolo Model
    sideViewModel_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=side_view_img_size)
    sideViewModel_tall = Detect(yolov5_folder_pth, side_view_model_pth, img_size=side_view_img_size_tall)
    
    coords_hs = []
    ex_w, ex_h = (dim*side_view_step_size for dim in side_view_img_size)

    with tqdm(
        crop_pcd_to_many(
            pcd = pcd,grd_pcd = grd,
            coords = coordinates,
            w_lin_pcd = (w_arr_pcd,w_incre_pcd),
            h_lin_pcd = (h_arr_pcd,h_incre_pcd),
            expand_x_y=[ex_w,ex_w]
        ), 
        unit ="n", 
        bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}'
        ) as t:
        for index, value in enumerate(t):
            t.set_description(f"Processing Trees")
            tree, _, x,y = value
            h, im , confi = get_h_from_each_tree_slice(
                tree = tree,
                model_short = sideViewModel_short,
                model_tall = sideViewModel_tall,
                img_size = side_view_img_size, 
                stepsize = side_view_step_size,
                img_dir = f"{sideViewOut}/{pcd_name}_{index}_",
                gen_undetected_img = False,
                img_with_h = True,
                min_no_points = min_points_per_tree
                )
            coords_hs.append((x,y,h,index,im, confi)) if h > 0 else None
    del sideViewModel_short, pcd, grd
    
    
    # Removing Duplicates
    coordinates = np.array(coords_hs.copy())[:,0:2]
    values, cluster_id = np.unique(coordinates, axis=0, return_inverse=True)

    clustered_coord_hs = []
    for i in range(max(cluster_id)):
        this_cluster = np.where(cluster_id==i)[0]
        n = len(this_cluster)
        if n < 2:
            clustered_coord_hs.append(coords_hs[this_cluster[0]])
        else:
            this_cluster_coord_hs = [coords_hs[ind] for ind in this_cluster]
            clustered_coord_hs.append(this_cluster_coord_hs[np.argmax(np.array(this_cluster_coord_hs.copy())[:,-1])])
    coords_hs = clustered_coord_hs
    
    ####################################################
    ####### END Generate Height from Each Tree #########
    ####################################################
    
    
    ####################################################
    ########### Step 5: Miscellaneous Work #############
    ####################################################
    logger.info("Step 5: Miscellaneous Finishing touches")
    
    # Create dataframe
    df = pd.DataFrame(coords_hs, columns=["x","y","h","index","numpy_image","confidence"])
    #df.to_csv(csvOut)
    del coords_hs
    
    # Draw heights of Side View
    logger.info("Step 5.1: Generate Images for Side View")
    for i, side_img in enumerate(df["numpy_image"]):
        side_img = img_b64_to_arr(side_img)
        cv2.imwrite(f"{sideViewOut}/{pcd_name}_{i}_.png", side_img)
    
    
    # Draw Height on TOp View
    def draw_height_on_coord(img, x_y_h:np.ndarray, on_left:str="tape"): #(xc,yc,confidence,label)
        img = img.copy()
        if not x_y_h.size:
            return img
        red = (255,0,0)
        color = red
        font = cv2.FONT_HERSHEY_SIMPLEX
        for stuff in x_y_h:
            x,y,h,index = int(stuff[0]), int(stuff[1]), stuff[2], stuff[3]
            img = cv2.circle(img,(x,y), 1, color,5)
            img = cv2.putText(img, f"{h:.1f}", (x,y),font, 0.7,red,2,cv2.LINE_AA)
            img = cv2.putText(img, f"{int(index)}", (x-16,y-20),font, 0.7,red,2,cv2.LINE_AA)
        # Writing image name
        img = cv2.putText(img, on_left, (50,50), font, 2, red, 2, cv2.LINE_AA)
        return img

    new_pd = np.asarray([df.x, df.y, df.h, df.index]).transpose()
    new_pd = scale_coord(new_pd, (1/topViewStepsize, 1/topViewStepsize), (-xmin/topViewStepsize,ymax/topViewStepsize))
    img_lidar = draw_height_on_coord(non_ground_img, new_pd, f"Lidar {pcd_name}")

    cv2.imwrite(f"{topViewOut}/{pcd_name}_lidar.png", img_lidar)


    # Save Dataframe with Mean
    df2 = pd.DataFrame([["Mean_height", df.h.mean(axis=0)]], columns=["x","h"])
    df3 = pd.DataFrame([["Confidence %", df.confidence.mean(axis=0)]], columns=["x","h"])
    df = df[["x","y","h","index","numpy_image"]]
    df  = pd.concat([df, df3, df2], ignore_index=True)
    df.to_csv(csvOut)
    
    ####################################################
    ########### END 5 : Miscellaneous Work #############
    ####################################################
    logger.info("Done!")
    
if __name__ == '__main__':
    logger.info("Done Loading Libraries\n")
    logger.info(f"Current dir: [{os.getcwd()}]")
    main(*sys.argv[1:])
    