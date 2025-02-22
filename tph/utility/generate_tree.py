import open3d
import cv2
from .pcd2img import *
from .get_coords import *
import random
import base64
import io
import numpy as np
import codecs
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

def get_tree_from_coord(pcd, grd_pcd, coord:list, expand:list=[10.0,10.0,40.0]):
    # CAREFUL THE Y IS ACTUALLY NEGATIVE
    xc, yc, zc = coord[0], -coord[1], 0.0
    
    l,w,h = expand
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-l/2,yc-w/2,-40),max_bound=(xc+l/2,yc+w/2,40))
    ground = grd_pcd.crop(bbox)
    
    zmin = ground.get_min_bound()[2]
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-l/2,yc-w/2,zmin),max_bound=(xc+l/2,yc+w/2,zmin+h))
    tree = pcd.crop(bbox)
    return tree

def generate_boundingbox(tree):
    c1 = tree.get_min_bound()
    c2 = tree.get_max_bound()
    
    points = [[c1[0],c1[1],c1[2]],[c2[0],c1[1],c1[2]],[c1[0],c2[1],c1[2]],[c2[0],c2[1],c1[2]],
              [c1[0],c1[1],c2[2]],[c2[0],c1[1],c2[2]],[c1[0],c2[1],c2[2]],[c2[0],c2[1],c2[2]]]
    lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],
             [5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )

    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def get_tree_many_from_coords(pcd, grd_pcd, coords, expand:list=[6.0,6.0,40.0]):
    trees = []
    bBoxes = []
    for coord in coords:
        x,y,z = coord[0], coord[1], 0.0
        tree = get_tree_from_coord(pcd, grd_pcd, [x,y,z], expand=expand)
        bbox = generate_boundingbox(tree)
        trees.append(tree)
        bBoxes.append(bbox)
        
    return np.stack((np.array(trees), np.array(bBoxes), coords[:,0], coords[:,1]), axis=-1)


def calculate_height(coords_with_pred, scale):
    # Scale the points at Z 
    coords_with_pred[:,1] = coords_with_pred[:,1]*scale
    
    x_or_y = coords_with_pred[:,0]
    z = coords_with_pred[:,1]
    conf = coords_with_pred[:,2]
    labels = coords_with_pred[:,3]
    
    unique = np.unique(labels, return_counts=True)
    
    # calculate height from current image 
    # If there are less than 2 labels, return 0
    # If there are more or equal to 1 per label, Get the data with higher confidence.
    if len(unique[1]) < 2:
        return 0
    else:
        label0_z = z[np.where(conf==np.amax(conf[(labels == 0)]))]
        label1_z = z[np.where(conf==np.amax(conf[(labels == 1)]))]
        height = (label1_z-label0_z)[0]
        return height if height > 0 else 0
 
        
def img_arr_to_b64(img_arr):
    """Grayscale"""
    img_pil = PIL.Image.fromarray(np.uint8(img_arr)).convert('RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

def get_h_from_each_tree_slice(tree, model, img_size:tuple, stepsize, img_dir, gen_img=False, img_with_h = True ,min_no_points:int=1000, circle_size = 2) -> tuple:
    c1 = tree.get_min_bound()
    c2 = tree.get_max_bound()
    xmin, xmax = c1[0], c2[0]
    ymin, ymax = c1[1], c2[1]
    zmin, zmax = c1[2], c2[2]
    
    xc = (xmin+xmax)/2
    yc = (ymin+ymax)/2
    # The issue still remains... since the code originates from xmax and xmin
    # It cannot be abs because xmin could be a positive: example Break scenario, xmin=-2 ,xmax=1
    # My guess is the tree doesn't even have sufficient points on the x direction. That would only mean 1 thing.
    # (1) My coordinate code has a mistake because i was using mean 
    # (2) There are insufficient points in the x-direction and y direction due to (1) 
    # To mitigate this on the short term, Lets change the tree.crop function.
    per_range = (xmax-xmin)/6 
    #per_range_y = (ymax-ymin)/6 
    # per_range = tree.get_extent()/6
    
    box_x = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-per_range/2,ymin,zmin), max_bound=(xc+per_range/2,ymax,zmax))
    box_y = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xmin,yc-per_range/2,zmin), max_bound=(xmax,yc+per_range/2,zmax))

    """This here is the short term fix"""
    slice_x = tree.crop(box_x) if box_x.get_max_bound()[0] != box_x.get_min_bound()[0] else open3d.geometry.PointCloud()
    slice_y = tree.crop(box_y) if box_y.get_max_bound()[1] != box_y.get_min_bound()[1] else open3d.geometry.PointCloud()
    """Short term fix End"""

    #open3d.visualization.draw_geometries([slice_x,slice_y])
    if len(slice_x.points) < min_no_points or len(slice_y.points) < min_no_points:
        return (0,0)
    img_x, img_y = pcd2img_np(slice_x,"x",stepsize), pcd2img_np(slice_y,"y",stepsize)
    height_lst = []
    img_lst = []
    
    # For each slice, generate an image
    for i, img in enumerate([img_x, img_y]):
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        preds = model.predict(img, convert_to_gray=False)
        
        xc , yc ,conf, pred = (preds[:,0]+preds[:,2])/2 ,(preds[:,1]+preds[:,3])/2, preds[:,4]*100,preds[:,-1]
        coords_with_pred =  np.stack([np.asarray(xc),np.asarray(yc), conf, pred], axis=1)
        
        # Calculate the Height and Scale it
        height = calculate_height(coords_with_pred, scale=1.0)
        height *= stepsize # Scaling it
        
        if i == 0:
            img_h, width = img_x.shape
        else:
            img_h, width = img_y.shape
            
        # Scaling the height
        #true_height = (height*(img_h/img_size[1]))*stepsize
        #true_height = (height* (img_size[1]/img_h) )*stepsize
        if height>0:
            height_lst.append(height)
        
        if img_with_h is True and height>0:
            img = draw_coord_on_img_with_pred(
                img, 
                coords_with_pred = coords_with_pred,
                height = height,
                circle_size = circle_size
            )
            img_lst.append(img)
        else:
            if height > 0:
                img_lst.append(img)
            img = img
        if gen_img:
           cv2.imwrite(f"{img_dir}_{i}.jpg", img)
    return (sum(height_lst)/len(height_lst), img_arr_to_b64(random.choice(img_lst))) if height_lst else (0, 0)
    #return max(height_lst)
