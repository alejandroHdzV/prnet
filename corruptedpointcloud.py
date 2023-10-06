import numpy as np
import copy
import open3d as o3d
import torch
import random

class CorruptedPointCloud():
    """
        References:
        https://github.com/jiachens/ModelNet40-C
    """
    def __init__(self, pc):
        self.pc = copy.deepcopy(pc)
        self.ORIG_NUM = pc.shape[0]

    def get_original(self):
        return self.pc
    
    def uniform_sampling(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][severity-1]
        index = np.random.choice(self.ORIG_NUM, self.ORIG_NUM - c, replace=False)
        return pointcloud[index]
    
    def random_shuffle_points(self):
        """Randomly permute point cloud."""
        points = self.pc
        indices = np.random.permutation(points.shape[0])
        points = points[indices]
        return points

    def normalize_points_1(self):
        new_pc = self.pc
        new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
        new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
        new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
        leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
        if leng_x >= leng_y and leng_x >= leng_z:
            ratio = 2.0 / leng_x
        elif leng_y >= leng_x and leng_y >= leng_z:
            ratio = 2.0 / leng_y
        else:
            ratio = 2.0 / leng_z
        new_pc *= ratio
        return new_pc, ratio

    def normalize_points(self):
        """GEOTRANSFORMER: Normalize point cloud to a unit sphere at origin."""
        points = self.pc
        points = points - points.mean(axis=0)
        factor = 1 / np.max(np.linalg.norm(points, axis=1))
        points = points * factor
        return points, factor

    def uniform_downsample_fixed_number_of_points(self, num_points) -> np.ndarray:
        """
        Uniformly downsamples a point cloud to a given number of points.

        Args:
            point_cloud (numpy.ndarray): The original point cloud of shape (N, 3).
            num_points (int): The desired number of points in the downsampled point cloud.

        Returns:
            numpy.ndarray: The downsampled point cloud of shape (num_points, 3).
        """
        point_cloud = self.pc
        np.random.seed(42)

        # Ensure num_points is within the range [1, len(point_cloud)]
        num_points = max(1, min(len(point_cloud), num_points))

        # Shuffle the indices of the original point cloud
        indices = np.arange(len(point_cloud))
        np.random.shuffle(indices)

        # Select the first num_points indices and create the downsampled point cloud
        downsampled_point_cloud = point_cloud[indices[:num_points]]

        
        return downsampled_point_cloud

    # DensityCorruptionPatterns
    # {Occlusion, LiDAR, LocalDensityInc, LocalDensityDec, Cutout}
    def occlusion(self, severity):
        ## severity here does not stand for real severity ##
        original_data = self.pc
        new_pc = occlusion_1(original_data, 'occlusion', severity, n_points=1024)

        theta =  -np.pi / 2.
        gamma =  0
        beta = np.pi

        matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
        
        new_pc = np.matmul(new_pc,matrix_1)
        new_pc = np.matmul(new_pc,matrix_2)
        new_pc = normalize(np.matmul(new_pc,matrix_3).astype('float32'))

        pointcloud.append(new_pc)

        pointcloud = np.stack(pointcloud,axis=0)

        return pointcloud

    def cutout(self, severity):
        '''
            Cutout several part in the point cloud
        '''
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [(2,30), (3,30), (5,30), (7,30), (10,30)][severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # pointcloud[idx.squeeze()] = 0
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # print(pointcloud.shape)
        return pointcloud
 
    def density_inc(self, severity):
        '''
        Density-based up-sampling the point cloud
        '''
        # TODO: 1024 is the number of points it returns
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
        # idx = np.random.choice(N,c[0])
        temp = []
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            # idx = idx[idx_2]
            temp.append(pointcloud[idx.squeeze()])
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        
        idx = np.random.choice(pointcloud.shape[0], 1024 - c[0] * c[1])
        temp.append(pointcloud[idx.squeeze()])

        pointcloud = np.concatenate(temp)
        # print(pointcloud.shape)
        return pointcloud

    def density(self, severity):
        '''
        Density-based sampling the point cloud
        '''
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            idx = idx[idx_2]
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
            # pointcloud[idx.squeeze()] = 0
        # print(pointcloud.shape)
        return pointcloud

    def camera_point_of_view(self, severity) -> np.array:
        """
            Crop point cloud from a point of view using Open3D from +Z (diameter) camera position
        """
        def initialize_pointcloud(pts: np.array) -> o3d.geometry.PointCloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            return pcd

        def get_pcd_diameter(pcd: o3d.geometry.PointCloud) -> float:
            return np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

        def get_view_point(pcd: o3d.geometry.PointCloud, cam_loc, rad_factor) -> o3d.geometry.PointCloud:
            """
                args:
                    PCD: Open3D point cloud
                    cam_loc: 2 or 3 long list, if 2, computes the z dimension based on the diameter, else take it as is.
                    rad_factor: Factor multiplier for the diamter to compute the camera radius.
            """

            # pcd = copy.deepcopy(pcd)
            diameter = get_pcd_diameter(pcd)

            if len(cam_loc) == 2:
                camera = [cam_loc[0], cam_loc[1], diameter]
            else:
                camera = cam_loc

            radius = diameter * rad_factor
            # print(f"Camera: {camera}, Diameter is: {diameter}, Radius: {radius}")
            _, pt_map = pcd.hidden_point_removal(camera, radius)
            pcd_viewpoint = pcd.select_by_index(pt_map)

            # Camera point
            # cam_representation = o3d.geometry.TriangleMesh.create_sphere(radius=diameter/30)
            # cam_representation = cam_representation.translate(np.array(camera, np.double))

            return pcd_viewpoint#, cam_representation

        np_pc = self.pc
        o3d_pc = initialize_pointcloud(np_pc)
        diameter = get_pcd_diameter(o3d_pc) * 1
        # Camera location: [-,-,diameter]
        o3d_pc_VOF = get_view_point(o3d_pc, [random.uniform(-diameter,diameter), random.uniform(-diameter,diameter)], 300)

        return np.array(o3d_pc_VOF.points)

    def farthest_subsample_points(self, severity):
        from sklearn.neighbors import NearestNeighbors
        from scipy.spatial.distance import minkowski
        
        pointcloud1 = self.pc
        N, C = pointcloud1.shape
        num_subsampled_points = [int(N*0.95),int(N*0.85),int(N*0.75),int(N*0.65),int(N*0.55)][severity-1]

        num_points = pointcloud1.shape[0]
        nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                                metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
        idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
        gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
        # YOU MAY ALSO GET THE MASK , gt_mask
        return pointcloud1[idx1, :]

    def random_crop_point_cloud_with_plane(self, severity, p_normal=None):
        def random_sample_plane():
            """Random sample a plane passing the origin and return its normal."""
            phi = np.random.uniform(0.0, 2 * np.pi)  # longitude
            theta = np.random.uniform(0.0, np.pi)  # latitude

            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            normal = np.asarray([x, y, z])

            return normal

        """GEOTRANSFORMER: Random crop a point cloud with a plane and keep num_samples points."""
        keep_ratio = [0.95, 0.85, 0.75, 0.65, 0.60][severity-1]
        points = self.pc
        num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
        if p_normal is None:
            p_normal = random_sample_plane()  # (3,)
        distances = np.dot(points, p_normal)
        sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
        points = points[sel_indices]
        return points

    # NoiseCorruptionPatterns
    # {Uniform, Gaussian, Impulse, Upsampling, Background}
    
    def uniform_noise(self, severity=1):
        '''
            Add Uniform noise to point cloud 
        '''
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
        jitter = np.random.uniform(-c,c,(N, C))
        new_pc = (pointcloud + jitter).astype('float32')
        return new_pc

    def gaussian_noise(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
        jitter = np.random.normal(size=(N, C)) * c
        new_pc = (pointcloud + jitter).astype('float32')
        new_pc = np.clip(new_pc,-1,1)
        return new_pc
    
    def impulse_noise(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [N//30, N//25, N//20, N//15, N//10][severity-1]
        index = np.random.choice(self.ORIG_NUM, c, replace=False)
        pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.1
        return pointcloud

    def upsampling(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [N//5, N//4, N//3, N//2, N][severity-1]
        index = np.random.choice(self.ORIG_NUM, c, replace=False)
        add = pointcloud[index] + np.random.uniform(-0.05,0.05,(c, C))
        new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
        return new_pc
    
    def background_noise(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [N//45, N//40, N//35, N//30, N//20][severity-1]
        jitter = np.random.uniform(-1,1,(c, C))
        new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
        return new_pc

    def jitter_pointcloud(self, clip=0.05):
        ####### learining3D, PRNet, PRCNet
        # N, C = pointcloud.shape
        pointcloud = self.pc
        N, C = pointcloud.shape
        sigma = 0.005*np.random.random_sample()
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return pointcloud

    # TransformationCorruptionsPatterns
    # {Rotation, Shear, FFD, RBF, InvRBF}
    def shear(self, severity):
        pointcloud = self.pc
        N, C = pointcloud.shape
        c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
        a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

        matrix = np.array([[1,0,b],[d,1,e],[f,0,1]])
        new_pc = np.matmul(pointcloud,matrix).astype('float32')
        return new_pc

