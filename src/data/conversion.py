import sys
import os
import pandas as pd

import numpy as np

import open3d as o3d
import h5py

from src.utils import graph_construct

def PlyToOff(input_path, output_path):
    '''
    Converts .ply files to .off file
 
    input_path = Path to the directory that has .ply files.
    output_path = Path to the directory the the converted files should be saved.
    '''

    files = os.listdir(input_path)
    
    for file in files:
        if file[-3:] == 'ply':
            mesh = o3d.io.read_triangle_mesh(input_path + file)
            o3d.io.write_triangle_mesh(output_path + file[:-4]+".off", mesh, write_ascii=True)

def extract(path):
    f = open(path, 'r')
    lines = f.readlines()
    if lines[0] == 'OFF\n':
        num = int(float(lines[1].split(" ")[0]))
        pts = []
        for i in range(2, 2+num):
            temp = lines[i][:-1].split(' ')
            pts.append([float(temp[0]), float(temp[1]), float(temp[2])])
    else:
        num = int(float(lines[0].split(" ")[0][3:]))
        pts = []
        for i in range(1, 1+num):
            temp = lines[i][:-1].split(' ')
            pts.append([float(temp[0]), float(temp[1]), float(temp[2])])
    return pts


def create_point(input_path, output_path):
    '''
    Converts .off file to point cloud.
 
    input_path = Path to the directory that has .off files.
    output_path = Path to the directory the the converted files should be saved.

    There should be a slash at the end of each path.
    '''
    files = os.listdir(input_path)
    for file in files:
        if file[-3:] == 'off':
            h5f = h5py.File(output_path + file[:-4]+'.h5', 'w')
            temp = np.array(extract(input_path + file))
            h5f.create_dataset('object', data=temp)
            h5f.close()

def construct_graph_with_knn(input_path, output_path, k):
    files = os.listdir(input_path)
    for file in files:
        if file[-2:] == 'h5':        
            f = h5py.File(input_path + file, 'r')
            print(file)    
            for key in f.keys():
                pts_num = f[key][:].shape[0]
                #if f[key][:].shape[0] >= pts_num:
                pts = graph_construct.pts_norm(graph_construct.pts_sample(f[key][:], pts_num))
                if np.isnan(pts).any():
                    continue
                temp = graph_construct.graph_construct_kneigh(pts, k=k)
                filename = output_path + file[:-3] + '.h5'
                out = h5py.File(filename, 'w')
                print(filename)
                out.create_dataset('edges', data=temp[0])
                out.create_dataset('edge_weight', data=temp[1])
                out.create_dataset('nodes', data=pts)
                out.close()
                print(out)