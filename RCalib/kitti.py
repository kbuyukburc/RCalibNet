import numpy as np
if __name__ == "__main__":
    from dataset import *
else:
    from .dataset import *

from os import path
def load_kitti_calib(filename):
    filedata = {}
    data = {}
    with open(filename) as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                filedata[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data        


from os import path
root_folder = '/media/kbuyukburc/DATASET/kitti-odo/'
class KITTIDatset(DatasetMerger):
    def __init__(self, root_folder : str, sequence : str):        
        self.cam2 = RGBImageLoader(path.join(root_folder, f'data_odometry_color/dataset/sequences/{sequence}/image_2/*'))
        self.cam3 = RGBImageLoader(path.join(root_folder, f'data_odometry_color/dataset/sequences/{sequence}/image_3/*'))
        self.lidar1 = BINLoader(path.join(root_folder, f'data_odometry_velodyne/dataset/sequences/{sequence}/velodyne/*'))
        self.calib = load_kitti_calib(path.join(root_folder, f'data_odometry_calib/dataset/sequences/{sequence}/calib.txt'))
        distortion = np.array([0,0,0,0,0.])
        self.dataset_1 = DatasetSequencer(FusionDataset(camera_dataset=self.cam2, lidar_dataset=self.lidar1,
              intrinsic=self.calib['K_cam2'], distortion=distortion,
              extrinsic=self.calib['T_cam2_velo']
            ), sequence_length=10)
        self.dataset_2 = DatasetSequencer(FusionDataset(camera_dataset=self.cam3, lidar_dataset=self.lidar1,
              intrinsic=self.calib['K_cam3'], distortion=distortion,
              extrinsic=self.calib['T_cam3_velo']
            ), sequence_length=10)
        DatasetMerger.__init__(self, self.dataset_1, self.dataset_2)
      
      
from os import path
root_folder = '/home/kbuyukburc/Kitti'
# root_folder = '/media/kbuyukburc/DATASET/kitti-odo/'
class KITTIDatsetV2(DatasetMerger):
    def __init__(self, root_folder : str, sequence_list : list, seq_len = 6):
        dataset_list = []        
        for sequence in sequence_list:
            self.cam2 = RGBImageLoader(path.join(root_folder, f'data_odometry_color/dataset/sequences/{sequence}/image_2/*'))
            self.cam3 = RGBImageLoader(path.join(root_folder, f'data_odometry_color/dataset/sequences/{sequence}/image_3/*'))
            self.lidar1 = BINLoader(path.join(root_folder, f'data_odometry_velodyne/dataset/sequences/{sequence}/velodyne/*'))
            self.calib = load_kitti_calib(path.join(root_folder, f'data_odometry_calib/dataset/sequences/{sequence}/calib.txt'))
            distortion = np.array([0,0,0,0,0.])
            dataset_1 = DatasetBatchSequencer(FusionDatasetV2(camera_dataset=self.cam2, lidar_dataset=self.lidar1,
                intrinsic=self.calib['K_cam2'], distortion=distortion,
                extrinsic=self.calib['T_cam2_velo']
                ), sequence_length=seq_len)
            dataset_list.append(dataset_1)
            dataset_2 = DatasetBatchSequencer(FusionDatasetV2(camera_dataset=self.cam3, lidar_dataset=self.lidar1,
                intrinsic=self.calib['K_cam3'], distortion=distortion,
                extrinsic=self.calib['T_cam3_velo']
                ), sequence_length=seq_len)
            dataset_list.append(dataset_2)
        DatasetMerger.__init__(self, *dataset_list)
      

if __name__ == "__main__":
    kittiset = KITTIDatsetV2(root_folder, ['00', '01'])
    print(kittiset[0]['img'].shape)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(kittiset, batch_size=2, shuffle=True, num_workers=6)
    it = iter(train_dataloader)
    data_batch = next(it)
    print(data_batch.keys())