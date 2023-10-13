import numpy as np
import open3d as o3

def read_pointnet_colors(seg_labels):
    ''' map segementation labels to colors '''
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors

def PointsTo3DShape(points):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points)

    return o3.visualization.draw_plotly([pcd])