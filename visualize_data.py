
#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
from data import ModelNet40
import numpy as np
import torch
import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vtctestregistrationdata import VTCTestRegistrationData
from generatedtrainingdata import GeneratedTrainingData


def draw_registration_result(source, target, transformation):
    SIZE_PERCENTAGE = 0.15
    v_size = round(max(source.get_max_bound() - source.get_min_bound()) * SIZE_PERCENTAGE, 4)

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)
    coord_syst = o3d.geometry.TriangleMesh.create_coordinate_frame(size=v_size, origin=[0, 0, 0])
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0 ])  # 1, 0.706, 0
    target_temp.paint_uniform_color([0, 0.651, 0.929]) #  # 0, 0.651, 0.929
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, coord_syst])

def initialize_pointcloud(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd



#########
generated_training_data_path = 'generated_training_dataset.hdf5'
trainset = GeneratedTrainingData(None, generated_training_data_path)

vtc_path = 'vtc_testing_dataset.hdf5'
testset = VTCTestRegistrationData(vtc_path, n_points=1024, meter_scaled=True)

#########


idx = 2
test_cts = testset[idx]
scaling = 1
t = initialize_pointcloud(test_cts[0].T.detach().cpu().numpy()*scaling)
s = initialize_pointcloud(test_cts[1].T.detach().cpu().numpy()*scaling)
transform = np.eye(4)
transform[:3, :3] = test_cts[2].detach().cpu().numpy()
transform[:3, 3] = test_cts[3].detach().cpu().numpy() * scaling
draw_registration_result(t,s,transform)


train_cts = trainset[idx]
t_train = initialize_pointcloud(train_cts[0].T)
s_train = initialize_pointcloud(train_cts[1].T)
transform = np.eye(4)
transform[:3, :3] = train_cts[2]
transform[:3, 3] = train_cts[3]

print(transform)
draw_registration_result(t_train,s_train,np.eye(4))
draw_registration_result(t_train,s_train,transform)


o3d.visualization.draw_geometries([t_train, s_train, t, s])