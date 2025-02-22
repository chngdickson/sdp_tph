print("Importing Files")
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
from utility.generate_tree import get_tree_many_from_coords, get_h_from_each_tree_slice
from utility.csf_py import csf_py

# Standard Libraries
import yaml
import open3d as o3d
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering

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
    
    assert input_file_type in [".txt",".pcd",".ply"],f"Filetype must be .txt, .pcd or .ply"
    
    # Read pcd
    if input_file_type == ".txt":
        format = 'xyz'
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

    # 2. Split images 
    for i, h in enumerate(h_arr[:-1]):
        for j, w in enumerate(w_arr[:-1]):
            img = non_ground_img[int(round(h)):int(round(h+h_incre)), int(round(w)):int(round(w+w_incre))]
            preds = topViewModel.predict(
                img,
                convert_to_gray=False,
                confi_thres = 0.134,
                iou_thres = 0.02
                )
            coordinates.extend(scale_pred_to_xy_point_cloud(preds, 1, w, h))
            del img, preds

    logger.info("Step 3.1: Performing Clustering")
    # 2.b Remove extra coordinates generated on step 2.a via clustering
    total_dist = 0
    coordinates = np.array(coordinates)
    #print(coordinates[:,0:2])
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
    min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]

    # Init SideViewYolo Model
    sideViewModel = Detect(yolov5_folder_pth, side_view_model_pth, img_size=side_view_img_size)
    coords_hs = []
    ex_w, ex_h = (dim*side_view_step_size for dim in side_view_img_size)
    
    for index, value in enumerate(get_tree_many_from_coords(pcd, grd, coordinates,expand=[ex_w,ex_w,ex_h])):
        tree, _, x,y = value
        h, im = get_h_from_each_tree_slice(
            tree = tree,
            model = sideViewModel,
            img_size = side_view_img_size, 
            stepsize = side_view_step_size,
            img_dir = f"{sideViewOut}/{pcd_name}_{index}_",
            gen_img = False,
            img_with_h = True,
            min_no_points = min_points_per_tree
            )
        coords_hs.append((x,y,h,index,im)) if h > 0 else None

    del sideViewModel, pcd, grd
    
    ####################################################
    ####### END Generate Height from Each Tree #########
    ####################################################
    
    
    ####################################################
    ########### Step 5: Miscellaneous Work #############
    ####################################################
    logger.info("Step 5: Miscellaneous Finishing touches")
    
    # Yaml Params
    df = pd.DataFrame(coords_hs, columns=["x","y","h","index","numpy_image"])
    df.to_csv(csvOut)
    
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

    ####################################################
    ########### END 5 : Miscellaneous Work #############
    ####################################################
    logger.info("Done!")
    
if __name__ == '__main__':
    logger.info("Done Loading Libraries\n")
    logger.info(f"Current dir: [{os.getcwd()}]")
    main(*sys.argv[1:])
    