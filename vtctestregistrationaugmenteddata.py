import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py
import random
from corruptedpointcloud import CorruptedPointCloud


class VTCTestRegistrationAugmentedData(Dataset):

    def __init__(self, dataset_path, n_points, meter_scaled=True, Rz_max = 20, Rxy_max = 20, t_max = 0.1):
        self.dataset_path = dataset_path
        self.n_points = n_points
        self.meter_scaled = meter_scaled
        self.scale_factor = 2.3
        self.severity = 2

        self.Rz_max = Rz_max
        self.Rxy_max = Rxy_max
        self.t_max = t_max

        with h5py.File(self.dataset_path, "r") as f:
            self.source = f['source']
            self.source = self.uniform_downsample_point_cloud(np.asarray(self.source), n_points)
            self.dataset_size = f['targets'].shape[0]
            
        # if not self.meter_scaled:
        self.source = self.source * self.scale_factor
            
        self.source = torch.from_numpy(self.source).float()

    def uniform_downsample_point_cloud(self, point_cloud, num_points: int) -> np.ndarray:
        """
        Uniformly downsamples a point cloud to a given number of points.

        Args:
            point_cloud (numpy.ndarray): The original point cloud of shape (N, 3).
            num_points (int): The desired number of points in the downsampled point cloud.

        Returns:
            numpy.ndarray: The downsampled point cloud of shape (num_points, 3).
        """
        np.random.seed(42)
        if not isinstance(point_cloud, np.ndarray):
            point_cloud = np.asarray(point_cloud.points)

        # Ensure num_points is within the range [1, len(point_cloud)]
        num_points = max(1, min(len(point_cloud), num_points))

        # Shuffle the indices of the original point cloud
        indices = np.arange(len(point_cloud))
        np.random.shuffle(indices)

        # Select the first num_points indices and create the downsampled point cloud
        downsampled_point_cloud = point_cloud[indices[:num_points]]

        return downsampled_point_cloud

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

        index = random.randint(0, 38)
        with h5py.File(self.dataset_path, "r") as f:
            template = np.asarray(f['targets'][index])
            igt = np.asarray(f['transformations'][index])
            self.source = self.uniform_downsample_point_cloud(np.asarray(f['source']), self.n_points) * self.scale_factor
            self.source = torch.from_numpy(self.source).float()

        template = self.apply_transform(template, np.linalg.inv(igt))
        igt = self.get_random_transformation_matrix()
        template = self.apply_transform(template, igt)        
        
        # if not self.meter_scaled:
        template = template * self.scale_factor
        igt[:3,3] = igt[:3,3] * self.scale_factor

        
        ###############
        template = CorruptedPointCloud(template).farthest_subsample_points(2)

        opt = random.choice(['background_noise','gaussian_noise','uniform_noise', 'jitter_pointcloud', 'None']) # 
        if opt == 'gaussian_noise':
            template = CorruptedPointCloud(template).gaussian_noise(self.severity)
        elif opt == 'background_noise':
            template = CorruptedPointCloud(template).background_noise(self.severity)
        elif opt == 'uniform_noise':
            template = CorruptedPointCloud(template).uniform_noise(self.severity)
        elif opt == 'uniform_noise':
            template = CorruptedPointCloud(template).jitter_pointcloud(self.severity)


        template = CorruptedPointCloud(template).random_shuffle_points()
        ################

        igt = torch.from_numpy(igt).float() # igt 4x4, source -> target
        template = self.uniform_downsample_point_cloud(template, self.n_points)
        template = torch.from_numpy(np.asarray(template)).float()

        R_ab, translation_ab, R_ba, translation_ba = self.get_transformations(igt.view(1,4,4))
        euler_ab = torch.from_numpy(R.from_matrix(R_ab.detach().cpu().numpy()).as_euler('zyx')).float()
        euler_ba = torch.from_numpy(R.from_matrix(R_ba.detach().cpu().numpy()).as_euler('zyx')).float()
        
        # torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3]) torch.Size([2, 3])
        
        # a, b, R_ab, t_at, R_ba, t_ba
        return template.T, self.source.T, R_ab, translation_ab.view(3), R_ba, translation_ba.view(3), euler_ab.view(3), euler_ba.view(3)
