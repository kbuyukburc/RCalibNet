import torch
import kornia.geometry as K

def rtvec_to_matrix(rtvec):
    B, C = rtvec.shape
    extrinsic = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(rtvec.device)
    extrinsic[:, :3, :3] = K.conversions.angle_axis_to_rotation_matrix(rtvec[:, :3])
    extrinsic[:, :3, 3] = rtvec[:, 3:]
    return extrinsic

def matrix_to_rtvec(extrinsic):
    B, C1, C2 = extrinsic.shape
    rtvec = torch.zeros(B, 6).to(extrinsic.device)
    rtvec[:, :3] = K.conversions.rotation_matrix_to_angle_axis(extrinsic[:, :3, :3].contiguous())
    rtvec[:, 3:] = extrinsic[:, :3, 3]
    return rtvec

if __name__ == "__main__":
    rtvec_rnd = torch.rand(8,6, requires_grad=True)
    print(rtvec_rnd)
    print(matrix_to_rtvec(rtvec_to_matrix(rtvec_rnd)))
