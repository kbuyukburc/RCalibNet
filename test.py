from RCalib import *
import torch
from torch.utils.data import DataLoader
from torch import optim
import argparse

# Arg parse
parser = argparse.ArgumentParser(prog='RCalibNet Train',)
parser.add_argument('-s', '--seq', help="Sequence length", default=12, type=int)
parser.add_argument('-b', '--batch', help="Batch length", default=16, type=int)
parser.add_argument('-d', '--dataset', help="Which dataset KITTI, NuScenes", default='kitti', type=str)

args = parser.parse_args()
# Load Model latest checkpoint
modelrnn = RCalibNet()
 
# Data loader 
if args.dataset.upper() == "KITTI":
    modelrnn.load_state_dict(torch.load('./rcalib-rot-tra-2.ckpt'))
    root_folder = '/home/kbuyukburc/Kitti'
    kittiset = KITTIDatsetV2(root_folder, ['06', '07'], seq_len = args.seq)
    train_dataloader = DataLoader(kittiset, batch_size=args.batch, shuffle=True, num_workers=8)
    padding = None
elif args.dataset.upper() == "NUSCENES":
    root_folder = './script/dataset_512'
    modelrnn.load_state_dict(torch.load('./rcalib-nuscenes-5.ckpt'))    
    nuset = NuscenesDatset(root_folder, "train", seq_len = args.seq)
    train_dataloader = DataLoader(nuset, batch_size=args.batch, shuffle=True, num_workers=8)
    padding = nuset.padding
else:
    assert "unknown dataset"

modelrnn.cuda()
modelrnn.eval()

with torch.no_grad():
    loss_list = []
    extrinsic_predict_np = np.empty([0,6])
    extrinsic_gt_np = np.empty([0,6])
    for data_batch in train_dataloader:
        
        for key in data_batch.keys():
            data_batch[key] = data_batch[key].cuda()
          
        B, S, C, H, W = data_batch['depth'].shape
        modelrnn.zero_grad()
        modelrnn.reset_hidden(B)
        def transform_depth(depth_map, extrinsic):            
            y = torch.arange(H)
            x = torch.arange(W)
            gridy, gridx = torch.meshgrid(x, y, indexing='ij')
            # mask_batch = (data_batch['depth'] < 0.1)
            gridx_batch = gridx.unsqueeze(0).unsqueeze(0).repeat([B,C,1,1]).cuda()
            gridy_batch = gridy.unsqueeze(0).unsqueeze(0).repeat([B,C,1,1]).cuda()

            # mask = (data_batch['depth'][0,0] == 0)[0]
            uvz = torch.zeros((B,3,H,W), dtype=torch.float32).cuda()
            uvz[:,0] = (gridx_batch * depth_map)[:,0]
            uvz[:,1] = (gridy_batch * depth_map)[:,0]
            uvz[:,2] = (depth_map)[:,0]
            points3d = (torch.inverse(data_batch['intrinsic'][:,frame_id]) @ uvz.reshape(B,3,-1))
            points3d_hom = torch.cat((points3d, torch.ones(B, 1, points3d.shape[-1]).cuda()),dim=1)
            projected = data_batch['intrinsic'][:,frame_id] @ (extrinsic @ points3d_hom)[:,:3]
            Z = projected[:,2,:]
            Y = projected[:,1,:] / Z
            X = projected[:,0,:] / Z
            X = X.to(torch.int64); Y = Y.to(torch.int64)
            depth_out = torch.zeros(B, C, H, W).cuda()
            flag = (Y < W) & (Y > 0) & (X > 0) & (X < W)
            flag_out_mean = torch.zeros(1).cuda()
            # for b in range(B):
            for b in range(B):
                flag_idx = flag[b]
                depth_out[b,:, Y[b, flag_idx],X[b, flag_idx]] = Z[b, flag_idx]
            return depth_out
        loss = torch.zeros(1).cuda()
        loss_squence_list = []
        for frame_id in range(S):
            # frame_id = 0 # 
            if frame_id == 0:        
                extrinsic_chained = torch.eye(4).unsqueeze(0).repeat(B,1,1).cuda()
                depth_ts = data_batch['depth'][:,frame_id]
                pose_gt = data_batch['pose'][:,frame_id].view(B,6)
                pose_org = data_batch['pose'][:,frame_id].clone().view(B,6)
            else:
                # print(frame_id)
                pose_gt = matrix_to_rtvec(rtvec_to_matrix(data_batch['pose'][:,frame_id].view(B,6)) @ torch.inverse(extrinsic_chained))
                depth_ts = data_batch['depth'][:,frame_id]
                depth_ts = transform_depth(depth_ts, extrinsic_chained)
                # depth_ts = depth_out

            if padding:
                calib_out = modelrnn(data_batch['img'][:,frame_id], depth_ts[:,:,padding:-padding,padding:-padding])
            else:
                calib_out = modelrnn(data_batch['img'][:,frame_id], depth_ts)
            calib_out_tanh = torch.tanh(calib_out)
            # extrinsic_ts = rtvec_to_matrix(calib_out_tanh * torch.pi).view(B,4,4)
            extrinsic_ts = rtvec_to_matrix(calib_out_tanh * torch.pi).view(B,4,4)        
            depth_out = transform_depth(depth_ts, extrinsic_ts)
            extrinsic_chained = extrinsic_ts.detach() @ extrinsic_chained

            # loss_linear = (data_batch['pose'][:,frame_id].view(B,6)/torch.pi - calib_out_tanh).abs().mean()
            linear_err = (pose_gt/torch.pi - calib_out_tanh)
            loss_rot_linear = torch.linalg.vector_norm(linear_err[:,:3], dim=1).mean()
            loss_tra_linear = torch.linalg.vector_norm(linear_err[:,3:], dim=1).mean()
            loss_depth = (data_batch['depth_gt'][:,frame_id] - depth_out).abs().mean()
            loss_squence_list.append(linear_err.abs().mean().detach().cpu().numpy())
            loss += loss_depth/64 + loss_rot_linear * 5 + loss_tra_linear * 5
            # if frame_id == 3:
            #     break

        extrinsic_predict_np = np.append(extrinsic_predict_np, matrix_to_rtvec(extrinsic_chained).detach().cpu().numpy(), axis=0)
        extrinsic_gt_np = np.append(extrinsic_gt_np, pose_org.detach().cpu().numpy(), axis=0)


        loss_list.append(loss.detach().cpu().numpy())
        print(f'Squence {np.array(loss_squence_list)} loss : {loss.item():.3f}, loss_rot : {loss_rot_linear:.3f}, loss_tra : {loss_tra_linear:.3f}, depth : {loss_depth:.3f} ', end='\r')    

    err_axis = np.abs(extrinsic_predict_np - extrinsic_gt_np).mean(0)
    print(f'\n loss : {np.array(loss_list).mean()} {err_axis}', end='\n')
    with open('./result.txt', 'a') as fw:
        abs_err = np.abs(extrinsic_predict_np - extrinsic_gt_np)* np.array([180/np.pi, 180/np.pi, 180/np.pi, 1,1,1])
        abs_rot_norm = np.linalg.norm(abs_err.mean(0)[:3])
        abs_tra_norm = np.linalg.norm(abs_err.mean(0)[3:])
        fw.write(f'{args.batch},{args.seq},{abs_rot_norm},{abs_tra_norm}\n')

    