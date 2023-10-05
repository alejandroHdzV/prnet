import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py

class VTCTestRegistrationData(Dataset):

    def __init__(self, dataset_path, n_points, meter_scaled=True):
        self.dataset_path = dataset_path
        self.n_points = n_points
        self.meter_scaled = meter_scaled
        self.scale_factor = 2
        
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

    def __getitem__(self, index):
        
        with h5py.File(self.dataset_path, "r") as f:
       
            template = np.asarray(f['targets'][index])
            igt = np.asarray(f['transformations'][index])
        
        template = self.uniform_downsample_point_cloud(template, self.n_points)
        template = torch.from_numpy(np.asarray(template)).float()
        
        # if not self.meter_scaled:
        template = template * self.scale_factor
        igt[:3,3] = igt[:3,3] * self.scale_factor

        igt = torch.from_numpy(igt).float() # igt 4x4, source -> target

        R_ab, translation_ab, R_ba, translation_ba = self.get_transformations(igt.view(1,4,4))
        euler_ab = torch.from_numpy(R.from_matrix(R_ab.detach().cpu().numpy()).as_euler('zyx')).float()
        euler_ba = torch.from_numpy(R.from_matrix(R_ba.detach().cpu().numpy()).as_euler('zyx')).float()
        
        # torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3, 3]) torch.Size([2, 3]) torch.Size([2, 3]) torch.Size([2, 3])
        
        # a, b, R_ab, t_at, R_ba, t_ba
        return template.T, self.source.T, R_ab, translation_ab.view(3), R_ba, translation_ba.view(3), euler_ab.view(3), euler_ba.view(3)
