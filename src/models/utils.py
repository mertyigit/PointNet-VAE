import numpy as np
import open3d as o3

def PointsTo3DShape(points):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points)

    return o3.visualization.draw_plotly([pcd])
