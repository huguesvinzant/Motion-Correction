import torch
import torch_dct as dct

from PoseCorrection.model import GCN_corr


def main_PC(poses_uniform):

    is_cuda = torch.cuda.is_available()

    model = GCN_corr()
    # model.load_state_dict(torch.load('PoseCorrection/Results/model.pt'))

    joints = list(range(15)) + [19, 21, 22, 24]
    poses_reshaped = poses_uniform[:, :, joints]
    poses_reshaped = poses_reshaped.reshape(-1, poses_reshaped.shape[1] * poses_reshaped.shape[2]).T

    frames = poses_reshaped.shape[1]
    dct_n = 25

    if frames >= dct_n:
        inputs = dct.dct_2d(poses_reshaped)[:, :dct_n]
    else:
        inputs = dct.dct_2d(torch.nn.ZeroPad2d((0, dct_n - frames, 0, 0))(poses_reshaped))

    if is_cuda:
        model.cuda()
        inputs = inputs.cuda()

    model.eval()
    with torch.no_grad():
        deltas, att = model(inputs)

        if frames > dct_n:
            m = torch.nn.ZeroPad2d((0, frames - dct_n, 0, 0))
            deltas = dct.idct_2d(m(deltas))
        else:
            deltas = dct.idct_2d(deltas[:, :frames])

        poses_corrected = poses_reshaped + deltas

    return poses_reshaped, poses_corrected, att