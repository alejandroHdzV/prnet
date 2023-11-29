import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py
from scipy.spatial.transform import Rotation as R
import random
from corruptedpointcloud import CorruptedPointCloud

class GeneratedTrainingData(Dataset):

    def __init__(self, args, dataset_path, n_points = 1024, meter_scaled=True,
                 Rz_max = 20, Rxy_max = 20, t_max = 0.5, severity = 3):
        """
            args:
                dataset_path: Path of dataset file, string
                n_points: Number of point cloud points to sample, int
                Rz_max: Max random initial rotation around z, float
                Rxy_max: Max random initial rotation around x and y, float
                t_max: Max random initial translation (x,y,z), float
                severity: Level of aggressiveness of point cloud distortions. Value in the range 1 to 5, int
        """
        self.dataset_path = dataset_path
        self.args = args
        self.n_points = np.abs(n_points)
        self.scale_factor = 1

        # Random initial misaligment
        self.Rz_max = Rz_max
        self.Rxy_max = Rxy_max
        self.t_max = t_max

        # Level of mimicking of common distortions on  real data
        self.severity = severity

        with h5py.File(self.dataset_path, "r") as f:
            self.point_cloud_data = np.asarray(f['data'])
            self.dataset_size = self.point_cloud_data.shape[0]

    def __len__(self):
        return self.dataset_size
    
    def get_transformations(self, igt):
        R_ba = igt[:, 0:3, 0:3]								# Ps = R_ba * Pt
        translation_ba = igt[:, 0:3, 3].unsqueeze(2)		# Ps = Pt + t_ba
        R_ab = R_ba.permute(0, 2, 1)						# Pt = R_ab * Ps
        translation_ab = -torch.bmm(R_ab, translation_ba)	# Pt = Ps + t_ab
        return R_ab[0], translation_ab[0], R_ba[0], translation_ba[0]

    def apply_transform(self, points: np.ndarray, transform: np.ndarray):
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points = np.matmul(points, rotation.T) + translation
        return points
    
    def get_random_transformation_matrix(self):
        z_max =  self.Rz_max  #args.Rz_max
        xy_max = self.Rxy_max #args.Rxy_max
        t_max =  self.t_max   #args.t_max
        
        t = [random.uniform(-t_max,t_max),
             random.uniform(-t_max,t_max),
             random.uniform(-t_max,t_max)]

        r = R.from_euler('xyz', [random.uniform(-z_max,z_max),
                                 random.uniform(-xy_max,xy_max),
                                 random.uniform(-xy_max,xy_max)],
                                 degrees=True).as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = t
        
        return transform

    def __getitem__(self, index):
        # Settings
        n_points = self.n_points
        severity = self.severity

        source = np.asarray(self.point_cloud_data[index])
        source, _ = CorruptedPointCloud(source).normalize_points()

        igt = self.get_random_transformation_matrix()
        template = self.apply_transform(source, igt)

        opt = random.choice(['cutout','crop_plane','camera_pov','None'])
        opt = 'crop_plane'
        if opt == 'cutout':
            template = CorruptedPointCloud(template).cutout(severity)
        elif opt == 'camera_pov':
            template = CorruptedPointCloud(template).camera_point_of_view(severity)
        elif opt == 'crop_plane':
            template = CorruptedPointCloud(template).random_crop_point_cloud_with_plane(severity)
        # elif opt == 'density_inc':
        #     template = CorruptedPointCloud(template).density_inc(severity)
        # elif opt == 'farthest_subsample_points':
        #     template = CorruptedPointCloud(template).farthest_subsample_points(severity)

        # Adjust severity for noise
        severity = 1
        opt = random.choice(['background_noise','gaussian_noise','uniform_noise', 'jitter_pointcloud', 'None']) # 
        if opt == 'gaussian_noise':
            template = CorruptedPointCloud(template).gaussian_noise(severity)
        # elif opt == 'background_noise':
        #     template = CorruptedPointCloud(template).background_noise(severity)
        elif opt == 'uniform_noise':
            template = CorruptedPointCloud(template).uniform_noise(severity)
        elif opt == 'jitter_pointcloud':
            template = CorruptedPointCloud(template).jitter_pointcloud(severity)
        

        ##### ENSURE EQUAL NUMBER OF POITNS #####
        template = CorruptedPointCloud(template).uniform_downsample_fixed_number_of_points(n_points)
        source = CorruptedPointCloud(source).uniform_downsample_fixed_number_of_points(n_points)

        template = CorruptedPointCloud(template).random_shuffle_points()
        source = CorruptedPointCloud(source).random_shuffle_points()

        ##############
        source = torch.from_numpy(source).float()
        template = torch.from_numpy(template).float()
        igt = torch.from_numpy(igt).float() # igt 4x4, source -> target

        R_ab, translation_ab, R_ba, translation_ba = self.get_transformations(igt.view(1,4,4))
        euler_ab = torch.from_numpy(R.from_matrix(R_ab.detach().cpu().numpy()).as_euler('zyx')).float()
        euler_ba = torch.from_numpy(R.from_matrix(R_ba.detach().cpu().numpy()).as_euler('zyx')).float()
        
        # torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3]) torch.Size([2, 3])
        
        # a, b, R_ab, t_at, R_ba, t_ba
        return template.T, source.T, R_ab, translation_ab.view(3), R_ba, translation_ba.view(3), euler_ab.view(3), euler_ba.view(3)
