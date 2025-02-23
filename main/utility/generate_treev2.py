import open3d as o3d
import tqdm
# Under the assumption that the library works
def tree_from_coord(non_grd, coord:tuple, radius_expand:int=3):
    xc, yc = coord[0], -coord[1]
    ex = radius_expand/2
    min_bound = (xc-ex, yc-ex, -1)
    max_bound = (xc+ex, yc+ex, 15)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    tree = non_grd.crop(bbox)
    o3d.visualization.draw_geometries([tree],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

def get_tree_from_many(non_grd, coords:list):
    # t = tqdm(coords,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
    # for i, coord in enumerate(t):
    coord_1 = coords[0]
    tree_from_coord(non_grd, coord_1, radius_expand=3)
        