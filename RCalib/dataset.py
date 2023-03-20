import torch
from torch.utils.data import Dataset
from torchvision import models
from glob import glob
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation, RotationSpline
import open3d as o3d
import numpy as np
import os


def FileLoader(preprocess=None):
    def innner(func):
        class Loader():
            def __init__(self, files):
                self.files = files
            def __getitem__(self, idx):
                if preprocess:
                    return preprocess(func(self.files[idx]))
                return func(self.files[idx])
            def __len__(self):
                return len(self.files)        
        def wrapper_func(glob_folder):
            files = sorted(glob(glob_folder), \
                           key = lambda x : int(os.path.split(x)[-1].split('.')[0]))
            return Loader(files)
        return wrapper_func
    return innner

@FileLoader()
def RGBImageLoader(x):
    return cv2.imread(x)


@FileLoader()
def BINLoader(x):
    bin = np.fromfile(x, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bin[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.c_[bin[:, 3], bin[:, 3], bin[:, 3]])
    return pcd

@FileLoader()
def PCDLoader(x):
    pcd = o3d.io.read_point_cloud(x)
    return pcd


# @FileLoader(preprocess = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
# def RGBImageLoader(x):
#     return cv2.imread(x)

def tensorifyIMG(img : np.ndarray):
    img = np.transpose(img, axes=[2,0,1]) / 255    
    return img

rng = np.random.default_rng(seed=42)
def randomTransformation(max_rot = 10, max_tra = 0.2):
    rotation_rnd = (rng.random(3) - 0.5) * 2 * max_rot * np.pi / 180
    translation_rnd = (rng.random(3) - 0.5) * 2 * max_tra
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = Rotation.from_rotvec(rotation_rnd).as_matrix()
    extrinsic[:3, 3] = translation_rnd
    extrinsic_inv = np.linalg.inv(extrinsic)
    rotation_inv = Rotation.from_matrix(extrinsic_inv[:3, :3]).as_rotvec()
    translation_inv = extrinsic_inv[:3, 3]
    return extrinsic, np.r_[rotation_inv, translation_inv]

class FusionDataset(Dataset):
    def __init__(self, camera_dataset, 
                lidar_dataset, 
                intrinsic,
                distortion,
                extrinsic,
                tensor_shape = (512, 512)                
                ):
        super().__init__()
        self.camera = camera_dataset
        self.lidar = lidar_dataset
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.extrinsic = extrinsic # Lidar To Camera        
        self.tensor_shape = tensor_shape
        assert len(self.camera) == len(self.lidar), "Dataset length are not matching."

    def __len__(self):
        return len(self.camera)

    def projectDepth(self, pointcloud, intrinsic, smooth = False):
        points_3d = np.asarray(pointcloud.points)
        zero_vec = np.array([0,0,0], dtype=np.float32)
        points_img, _ =  cv2.projectPoints(points_3d, zero_vec, zero_vec, 
                    intrinsic, self.distortion)
        
        points_img = points_img[:, 0, :]
        points_img_flag = (points_img[:, 1] > 0) & (points_img[:, 1] < self.tensor_shape[0]) & \
            (points_img[:, 0] > 0) & (points_img[:, 0] < self.tensor_shape[1]) & \
                (points_3d[:, 2] > 0)
        points_3d_flaged = points_3d[points_img_flag]
        points_img_flaged = points_img[points_img_flag]
        points_3d_norm = np.linalg.norm(points_3d_flaged, axis=1)
        
        ind = np.argsort(points_3d_norm)[::-1]
        points_img_sorted = points_img_flaged[ind]
        points_img_sorted = points_img_sorted.astype(int)
        points_3d_flaged_sorted = points_3d_flaged[ind]

        depth = np.zeros((1, *self.tensor_shape), dtype=np.float32)
        depth[0, points_img_sorted[:, 1], points_img_sorted[:, 0]] = points_3d_flaged_sorted[:, 2]
        if smooth:
            depth_smooth = cv2.GaussianBlur(depth, (15,15), 0)
            return depth_smooth
        return depth

    def __getitem__(self, index):
        img = self.camera[index]
        h, w = img.shape[:2]
        img_resize = cv2.resize(img, self.tensor_shape)
        if isinstance(self.intrinsic, list):
            intrinsic = self.intrinsic[index].copy()
        else:
            intrinsic = self.intrinsic.copy()
        # print(intrinsic)
        intrinsic[0] *= (self.tensor_shape[0]/w)
        intrinsic[1] *= (self.tensor_shape[1]/h)
        # print(intrinsic)
        
        pointcloud : o3d.geometry.PointCloud = self.lidar[index]
        pointcloud.transform(self.extrinsic)
        depth_gt = self.projectDepth(pointcloud, intrinsic, True)
        miss_calibration, miss_pose = randomTransformation()
        pointcloud.transform(miss_calibration)
        depth = self.projectDepth(pointcloud, intrinsic)
        return {'img' : tensorifyIMG(img_resize).astype(np.float32),
                'depth': depth.astype(np.float32),
                'depth_gt': depth_gt.astype(np.float32),
                'extrinsic': miss_calibration.astype(np.float32),
                'intrinsic': intrinsic.astype(np.float32),
                'pose' : miss_pose.astype(np.float32)
            }

class DatasetMerger:
    def __init__(self, *args):
        self.dataset_list = [*args]
        self.datset_length = [len(dataset) for dataset in self.dataset_list]
        self.dataset_edge = [0]
        for dataset in self.dataset_list:
            self.dataset_edge.append(len(dataset)+self.dataset_edge[-1])
        self.length = self.dataset_edge[-1]
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), "Out of range"
        for id, dataset in enumerate(self.dataset_list):
            if self.dataset_edge[id+1] > idx >= self.dataset_edge[id]:
                return dataset[idx-self.dataset_edge[id]]
        raise('Something went wrong!')

class DatasetSequencer:
    def __init__(self, dataset, sequence_length = 10):
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset) // self.sequence_length

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), "Out of range"
        index = idx * self.sequence_length
        batch_data = self.dataset[index]
        if isinstance(batch_data, np.ndarray):
            data_shape = batch_data.shape
            batch_data = batch_data[None, ...]
            for num in range(1, self.sequence_length):
                data =  self.dataset[index+num]
                assert data_shape == data.shape, "Data Shape not matching"
                batch_data = np.append(batch_data, data[None, ...], axis=0)
        elif isinstance(batch_data, dict):
            data_shape = {key : data.shape for key, data in batch_data.items()}
            batch_data = {key : data[None, ...] for key, data in batch_data.items()}
            for num in range(1, self.sequence_length):
                data =  self.dataset[index+num]
                for key in batch_data.keys():
                    assert data_shape[key] == data[key].shape, "Data Shape not matching"
                    batch_data[key] = np.append(batch_data[key], data[key][None, ...], axis=0)
        else:
            raise('Hmmm! we werent expecting this case =/.')


        return batch_data

class FusionDatasetV2(Dataset):
    def __init__(self, camera_dataset, 
                lidar_dataset, 
                intrinsic,
                distortion,
                extrinsic,
                tensor_shape = (512, 512)                
                ):
        super().__init__()
        self.camera = camera_dataset
        self.lidar = lidar_dataset
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.extrinsic = extrinsic # Lidar To Camera        
        self.tensor_shape = tensor_shape
        assert len(self.camera) == len(self.lidar), "Dataset length are not matching."

    def __len__(self):
        return len(self.camera)

    def projectDepth(self, pointcloud, intrinsic, smooth = False):
        points_3d = np.asarray(pointcloud.points)
        zero_vec = np.array([0,0,0], dtype=np.float32)
        points_img, _ =  cv2.projectPoints(points_3d, zero_vec, zero_vec, 
                    intrinsic, self.distortion)
        
        points_img = points_img[:, 0, :]
        points_img_flag = (points_img[:, 1] > 0) & (points_img[:, 1] < self.tensor_shape[0]) & \
            (points_img[:, 0] > 0) & (points_img[:, 0] < self.tensor_shape[1]) & \
                (points_3d[:, 2] > 0)
        points_3d_flaged = points_3d[points_img_flag]
        points_img_flaged = points_img[points_img_flag]
        points_3d_norm = np.linalg.norm(points_3d_flaged, axis=1)
        
        ind = np.argsort(points_3d_norm)[::-1]
        points_img_sorted = points_img_flaged[ind]
        points_img_sorted = points_img_sorted.astype(int)
        points_3d_flaged_sorted = points_3d_flaged[ind]

        depth = np.zeros((1, *self.tensor_shape), dtype=np.float32)
        depth[0, points_img_sorted[:, 1], points_img_sorted[:, 0]] = points_3d_flaged_sorted[:, 2]
        if smooth:
            depth_smooth = cv2.GaussianBlur(depth, (15,15), 0)
            return depth_smooth
        return depth

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, miss_calibration, miss_pose = index
        else:
            miss_calibration = np.eye(4); miss_pose = np.zeros(6)            
            
        img = self.camera[index]
        h, w = img.shape[:2]
        img_resize = cv2.resize(img, self.tensor_shape)        
        
        if isinstance(self.intrinsic, list):
            intrinsic = self.intrinsic[index].copy()
        else:
            intrinsic = self.intrinsic.copy()
        # intrinsic = self.intrinsic.copy()
        # print(intrinsic)
        intrinsic[0] *= (self.tensor_shape[0]/w)
        intrinsic[1] *= (self.tensor_shape[1]/h)
        # print(intrinsic)
        
        pointcloud : o3d.geometry.PointCloud = self.lidar[index]
        pointcloud.transform(self.extrinsic)
        depth_gt = self.projectDepth(pointcloud, intrinsic, True)
        # miss_calibration, miss_pose = randomTransformation()
        pointcloud.transform(miss_calibration)
        depth = self.projectDepth(pointcloud, intrinsic)
        return {'img' : tensorifyIMG(img_resize).astype(np.float32),
                'depth': depth.astype(np.float32),
                'depth_gt': depth_gt.astype(np.float32),
                'extrinsic': miss_calibration.astype(np.float32),
                'intrinsic': intrinsic.astype(np.float32),
                'pose' : miss_pose.astype(np.float32)
            }


class FusionDatasetV3(Dataset):
    def __init__(self, camera_dataset, 
                lidar_dataset, 
                intrinsic,
                distortion,
                extrinsic,
                tensor_shape = (512, 512),
                padding = 50
                ):
        super().__init__()
        self.camera = camera_dataset
        self.lidar = lidar_dataset
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.extrinsic = extrinsic # Lidar To Camera        
        self.tensor_shape = tensor_shape
        self.padding = padding
        assert len(self.camera) == len(self.lidar), "Dataset length are not matching."

    def __len__(self):
        return len(self.camera)

    def projectDepth(self, pointcloud, intrinsic, smooth = False):
        points_3d = np.asarray(pointcloud.points)
        zero_vec = np.array([0,0,0], dtype=np.float32)
        points_img, _ =  cv2.projectPoints(points_3d, zero_vec, zero_vec, 
                    intrinsic, self.distortion)
        
        paded_height = self.tensor_shape[0]+2*self.padding
        paded_width = self.tensor_shape[1]+2*self.padding

        points_img = points_img[:, 0, :]
        points_img_flag = (points_img[:, 1] > 0) & (points_img[:, 1] < paded_height) & \
            (points_img[:, 0] > 0) & (points_img[:, 0] < paded_width) & \
                (points_3d[:, 2] > 0)
        points_3d_flaged = points_3d[points_img_flag]
        points_img_flaged = points_img[points_img_flag]
        points_3d_norm = np.linalg.norm(points_3d_flaged, axis=1)
        
        ind = np.argsort(points_3d_norm)[::-1]
        points_img_sorted = points_img_flaged[ind]
        points_img_sorted = points_img_sorted.astype(int)
        points_3d_flaged_sorted = points_3d_flaged[ind]

        depth = np.zeros((1, paded_height, paded_width), dtype=np.float32)
        depth[0, points_img_sorted[:, 1], points_img_sorted[:, 0]] = points_3d_flaged_sorted[:, 2]
        if smooth:
            depth_smooth = cv2.GaussianBlur(depth, (15,15), 0)
            return depth_smooth
        return depth

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, miss_calibration, miss_pose = index
        else:
            miss_calibration = np.eye(4); miss_pose = np.zeros(6)            
            
        img = self.camera[index]
        h, w = img.shape[:2]
        img_resize = cv2.resize(img, self.tensor_shape)        
        
        if isinstance(self.intrinsic, list):
            intrinsic = self.intrinsic[index].copy()
        else:
            intrinsic = self.intrinsic.copy()
        # intrinsic = self.intrinsic.copy()
        # print(intrinsic)
        intrinsic[0] *= (self.tensor_shape[0]/w)
        intrinsic[1] *= (self.tensor_shape[1]/h)
        intrinsic[0,2] = intrinsic[0,2] + self.padding
        intrinsic[1,2] = intrinsic[1,2] + self.padding

        # print(intrinsic)
        
        pointcloud : o3d.geometry.PointCloud = self.lidar[index]
        pointcloud.transform(self.extrinsic)
        depth_gt = self.projectDepth(pointcloud, intrinsic, True)
        # miss_calibration, miss_pose = randomTransformation()
        pointcloud.transform(miss_calibration)
        depth = self.projectDepth(pointcloud, intrinsic)
        return {'img' : tensorifyIMG(img_resize).astype(np.float32),
                'depth': depth.astype(np.float32),
                'depth_gt': depth_gt.astype(np.float32),
                'extrinsic': miss_calibration.astype(np.float32),
                'intrinsic': intrinsic.astype(np.float32),
                'pose' : miss_pose.astype(np.float32)
            }




class DatasetBatchSequencer:
    def __init__(self, dataset, sequence_length = 4):
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset) // self.sequence_length

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), "Out of range"
        index = idx * self.sequence_length        
        batch_data = self.dataset[index]
        if isinstance(batch_data, np.ndarray):
            data_shape = batch_data.shape
            batch_data = batch_data[None, ...]
            for num in range(1, self.sequence_length):
                data =  self.dataset[index+num]
                assert data_shape == data.shape, "Data Shape not matching"
                batch_data = np.append(batch_data, data[None, ...], axis=0)
        elif isinstance(batch_data, dict):
            miss_calibration, miss_pose = randomTransformation()
            batch_data = self.dataset[index, miss_calibration, miss_pose]
            data_shape = {key : data.shape for key, data in batch_data.items()}
            batch_data = {key : data[None, ...] for key, data in batch_data.items()}            
            for num in range(1, self.sequence_length):                
                data =  self.dataset[index+num, miss_calibration, miss_pose]
                for key in batch_data.keys():
                    assert data_shape[key] == data[key].shape, "Data Shape not matching"
                    batch_data[key] = np.append(batch_data[key], data[key][None, ...], axis=0)
        else:
            raise('Hmmm! we werent expecting this case =/.')


        return batch_data


if __name__ == "__main__":
    cam2 = RGBImageLoader('/media/kbuyukburc/DATASET/kitti-odo/data_odometry_color/dataset/sequences/00/image_2/*')    
    lidar1 = BINLoader('/media/kbuyukburc/DATASET/kitti-odo/data_odometry_velodyne/dataset/sequences/00/velodyne/*')
    intrinsic = np.array([[718.856 ,   0.    , 607.1928],
       [  0.    , 718.856 , 185.2157],
       [  0.    ,   0.    ,   1.    ]])
    distortion = np.array([0,0,0,0,0.])
    rotvec_gt = np.asarray([ 1.20875812, -1.21797831, 1.19949], dtype="float32")  
    tvec_gt = np.asarray([0.05114661, -0.05403985, -0.29219686], dtype="float32")
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec(rotvec_gt).as_matrix()
    extrinsic[:3, 3] = tvec_gt
    dataset = FusionDataset(camera_dataset=cam2, lidar_dataset=lidar1,
                intrinsic=intrinsic, distortion=distortion,
                extrinsic=extrinsic
                )
    print(dataset[0].keys())
    sqset = DatasetSequencer(dataset=cam2, sequence_length=10)
    print(len(sqset))
    print(sqset[0].shape)
    print(sqset[453].shape)
    seqset = DatasetSequencer(dataset=dataset, sequence_length=10)
    print(seqset[0]['img'].shape)