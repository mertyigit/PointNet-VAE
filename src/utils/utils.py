import numpy as np
import open3d as o3


def PointsTo3DShape(points):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points)

    return o3.visualization.draw_plotly([pcd])


def VisualizeEmbedding(embedding):
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(8, 6))

    # Generate a 2D tensor
    tensor = embedding


    # Create a figure object with custom size

    # Plot the heatmap using Seaborn
    sns.heatmap(tensor, cmap='coolwarm', cbar=True)
    plt.title('Tensor Heatmap')

    plt.show()