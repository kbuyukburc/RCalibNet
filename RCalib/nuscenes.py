import numpy as np
if __name__ == "__main__":
    from dataset import *
else:
    from .dataset import *    
from os import path
root_folder = '/home/kbuyukburc/Kitti'
# root_folder = '/media/kbuyukburc/DATASET/kitti-odo/'
class NuscenesDatset(DatasetMerger):
    def __init__(self, root_folder : str, set_type : str = 'train', seq_len = 6, padding=120):
        self.padding = padding
        dataset_list = []
        if set_type == 'train':
            sensor_list = ['CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']
        elif set_type == 'test':
            sensor_list = ['CAM_BACK_LEFT', 'CAM_FRONT_RIGHT']
        else:
            assert 'you shall not pass!'
        for sensor in sensor_list:            
            cam = RGBImageLoader(path.join(root_folder, f'{sensor}/*.jpg'))
            lidar = PCDLoader(path.join(root_folder, f'LIDAR/{sensor}/*'))
            initrinc_files = sorted(glob(path.join(root_folder, f'{sensor}/intrinsic/*.npy')), \
                                    key = lambda x : int(os.path.split(x)[-1].split('.')[0])
                                    )
            intrinsic_list = [np.load(file) for file in initrinc_files]
            assert len(cam) == len(lidar) == len(intrinsic_list), 'Dataset are not matching!'
            distortion = np.array([0,0,0,0,0.])
            dataset_1 = DatasetBatchSequencer(FusionDatasetV3(camera_dataset=cam, lidar_dataset=lidar,
                intrinsic=intrinsic_list, distortion=distortion,
                extrinsic=np.eye(4), tensor_shape=(256,256), padding=self.padding
                ), sequence_length=seq_len)
            dataset_list.append(dataset_1)

        DatasetMerger.__init__(self, *dataset_list)

if __name__ == "__main__":
    nuset = NuscenesDatset(root_folder='./script/dataset_512/')
    print(nuset[0]['img'].shape)
    print(nuset[0]['depth'].shape)
    print(nuset[0]['depth_gt'].shape)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(nuset, batch_size=2, shuffle=False, num_workers=6)
    it = iter(train_dataloader)
    data_batch = next(it)    
    print(data_batch.keys())
    plt.figure(figsize=[5,5])
    for i in range(0,6):
        plt.subplot(6,3, (i*3)+1)
        plt.imshow(np.transpose(data_batch['img'][0,i].numpy(), axes=[1,2,0]))
        plt.subplot(6,3, (i*3)+2)
        plt.imshow(data_batch['depth'][0,i,0].numpy())
        plt.subplot(6,3, (i*3)+3)
        plt.imshow(data_batch['depth_gt'][0,i,0].numpy())
    cv2.imshow("w", data_batch['depth'][0,0,0].numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()