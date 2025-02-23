import open3d
# Under the assumption that the library works
def tree_from_coord(non_grd, coord:list, radius_expand:int=3):
    xc, yc = coord[0], -coord[1]
    
    