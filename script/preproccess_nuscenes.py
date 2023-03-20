from nuscenes.nuscenes import NuScenes
from os import path
import cv2
from tqdm import tqdm
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import os
import argparse
from nuscenes.utils.geometry_utils import view_points

parser = argparse.ArgumentParser("NuScenes")
parser.add_argument("--dataset", "-d", \
        type=str, help="Dataset folder of NuScenes", \
        default="~/repo/nuscenes/v1.0-mini")
parser.add_argument("--size", "-s", \
        type=int, help="Size of the image", \
        default=256)
parser.add_argument("--output", "-o", \
        type=str, help="Output folder",
        default="./dataset")
parser.add_argument("--mask", "-m", \
        help="Generate Masks",
        action='store_true', default=False)
args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval', dataroot='/media/kbuyukburc/DATASET/nuscenes/v1.0-trainval01_blobs', verbose=True)

def calibDictToMatrix(calib_dict : dict) -> np.ndarray:
    quat = Quaternion(calib_dict['rotation'])
    matrix = np.eye(4)
    matrix[:3,:3] = quat.rotation_matrix
    matrix[:3, 3] = calib_dict['translation']
    return matrix

scene_dict = nusc.scene[0]
first_sample_token = scene_dict['first_sample_token']
last_sample_token = scene_dict['last_sample_token']
first_sample = nusc.get('sample', first_sample_token)
last_sample = nusc.get('sample', last_sample_token)

dataset_folder = path.join(f"{args.output}_{args.size}")
mask_folder = path.join(f"{dataset_folder}", "MASK")
projected_folder = path.join(f"{dataset_folder}", "PROJECTED")
CAMERA_SENSORS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

try:
    os.mkdir(dataset_folder)
    os.mkdir(path.join(dataset_folder, 'LIDAR'))
except:
    pass
try:
    os.mkdir(projected_folder)    
except:
    pass
try:
    os.mkdir(mask_folder)
except:
    pass
for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(dataset_folder, cam_sensor))
        os.mkdir(path.join(dataset_folder, cam_sensor, 'intrinsic'))
    except:
        pass

for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(dataset_folder, 'LIDAR', cam_sensor))
    except:
        pass


for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(projected_folder, cam_sensor))
    except:
        pass    
    try:
        os.mkdir(path.join(mask_folder, cam_sensor))
    except:
        pass

cntr = 0
scenes_num = len(nusc.scene)

for scene_num in tqdm(range(scenes_num)):
    scene_dict = nusc.scene[scene_num]
    first_sample_token = scene_dict['first_sample_token']
    last_sample_token = scene_dict['last_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    last_sample = nusc.get('sample', last_sample_token)
    current_sample = first_sample

    while current_sample['token'] != last_sample['token']:
        for cam_sensor in CAMERA_SENSORS:
            cam = nusc.get('sample_data', current_sample['data'][cam_sensor])
            cam_img_path = path.join(nusc.dataroot, cam['filename'])
            cam_img = cv2.imread(cam_img_path)
            #cam = nusc.get('sample_data', current_sample['data'][cam_sensor])
            lidar_top = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            lidar_top_path = path.join(nusc.dataroot, lidar_top['filename'])
            lidar_top_bin = np.fromfile(lidar_top_path, dtype=np.float32).reshape(-1,5)[..., :4]    
            lidar_top_pointcloud = o3d.geometry.PointCloud()
            lidar_reflection = lidar_top_bin[:, 3]
            lidar_points = lidar_top_bin[..., :3]
            colors = np.c_[lidar_top_bin[..., 3], lidar_top_bin[..., 3], lidar_top_bin[..., 3]] / 255
            lidar_top_pointcloud.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_top_pointcloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Transformation Dict
            calib_lidar_to_vechile = nusc.get('calibrated_sensor', lidar_top['calibrated_sensor_token'])
            calib_vechile_to_global_lidar_time = nusc.get('ego_pose', lidar_top['ego_pose_token'])
            calib_vechile_to_global_camera_time = nusc.get('ego_pose', cam['ego_pose_token'])
            calib_camera_to_vechile =  nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            # Transformation Matrix
            calib_lidar_to_vechile_matrix = calibDictToMatrix(calib_lidar_to_vechile)
            calib_vechile_to_global_lidar_time_matrix = calibDictToMatrix(calib_vechile_to_global_lidar_time)
            calib_vechile_to_global_camera_time_matrix = calibDictToMatrix(calib_vechile_to_global_camera_time)
            calib_camera_to_vechile_matrix = calibDictToMatrix(calib_camera_to_vechile)
            calib_lidar_to_camera_matrix = np.linalg.inv(calib_camera_to_vechile_matrix) @\
                                            np.linalg.inv(calib_vechile_to_global_camera_time_matrix) @ \
                                                calib_vechile_to_global_lidar_time_matrix @ \
                                                calib_lidar_to_vechile_matrix
            camera_top_pointcloud = lidar_top_pointcloud.transform(calib_lidar_to_camera_matrix)
            camera_points = np.asarray(lidar_top_pointcloud.points)

            rvec = R.from_matrix(calib_lidar_to_camera_matrix[:3, :3]).as_rotvec()
            tvec = calib_lidar_to_camera_matrix[:3, 3]
            depth = camera_points[:, 2]
            img_points, joc = cv2.projectPoints(lidar_points.T, rvec, tvec, np.array(calib_camera_to_vechile['camera_intrinsic']), np.array([[0,0,0,0,0]], dtype=np.float32))
            flag = (img_points[:,0, 0] < cam_img.shape[1]) & (img_points[:,0, 0] > 0) & \
                (img_points[:,0, 1] < cam_img.shape[0]) & (img_points[:,0, 1] > 0) & (depth > 0)
            img_points_flag = img_points[:,0,:][flag].astype(int)
            img = deepcopy(cam_img)
            #img[img_points_flag[:,1], img_points_flag[:,0]] = [0, 0 , 255]
            # lidar_reflection_flag = lidar_reflection[flag]
            # depth_flag = depth[flag]
            # lidar_data = np.zeros((2,img.shape[0],img.shape[1]), dtype=np.float32)
            # lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            # lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = lidar_reflection_flag

            img_points_flag = img_points_flag.copy()
            img_points_flag[:, 0] = img_points_flag[:, 0] * args.size / img.shape[1]
            img_points_flag[:, 1] = img_points_flag[:, 1] * args.size / img.shape[0]
            # lidar_data = np.zeros((2, args.size, args.size), dtype=np.float32)
            # lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            # lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = lidar_reflection_flag
            img = cv2.resize(img, (args.size, args.size))
            img_projected = deepcopy(img)
            img_projected[img_points_flag[:,1], img_points_flag[:,0]] = [0, 0 , 255]
            
            cv2.imwrite(path.join(dataset_folder, f'{cam_sensor}/{cntr}.jpg'), cam_img)
            cv2.imwrite(path.join(projected_folder, f'{cam_sensor}/{cntr}.jpg'), img_projected)
            o3d.io.write_point_cloud(path.join(dataset_folder, f'LIDAR/{cam_sensor}/{cntr}.pcd'), camera_top_pointcloud)
            np.save(path.join(dataset_folder, f'{cam_sensor}/intrinsic/{cntr}.npy'), 
                    np.array(calib_camera_to_vechile['camera_intrinsic']))
            # np.savez_compressed(path.join(dataset_folder, f'LIDAR/{cam_sensor}/{cntr}.npy'), lidar_data)
            # Mask
            if args.mask:
                img_path, boxes, camera_intrinsic = nusc.get_sample_data(current_sample['data'][cam_sensor])
                mask = np.zeros(cam_img.shape[:2], dtype = np.uint8)
                for box in boxes:
                    img_corners = view_points(box.corners(), camera_intrinsic, True)[:2, :].T.astype(int)        
                    all_corners = np.array(np.meshgrid(range(8),range(8),range(8),range(8))).T.reshape(-1, 4)
                    for corners in all_corners:    
                        points = np.array([img_corners[corner] for corner in corners])
                        mask = cv2.fillPoly(mask, pts=[points], color=(255))
                mask = cv2.resize(mask, (args.size, args.size), interpolation = cv2.INTER_NEAREST_EXACT)
                cv2.imwrite(path.join(mask_folder, f'{cam_sensor}/{cntr}.jpg'), mask)
        cntr += 1
        current_sample = nusc.get('sample', current_sample['next'])
