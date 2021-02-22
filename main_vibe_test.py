import pickle
import torch
import torch_dct as dct
import os

from PoseCorrection.model import GCN_class, GCN_corr
from VIBE.demo_hv import main_VIBE
from VIBE.lib.models.vibe import VIBE_Demo
from VIBE.lib.utils.demo_utils import download_ckpt
from utils.skeleton_uniform import centralize_normalize_rotate_poses


def main(video_file, pose_dict, model):

    is_cuda = torch.cuda.is_available()

    # ============== 3D pose estimation ============== #
    poses = main_VIBE(video_file, model)
    # with open('PoseCorrection/Results/poses_vibe.pickle', 'rb') as f:
    #     poses = pickle.load(f)

    # ============== Squeleton uniformization ============== #
    poses_uniform = centralize_normalize_rotate_poses(poses, pose_dict)
    joints = list(range(15)) + [19, 21, 22, 24]
    poses_reshaped = poses_uniform[:, :, joints]
    poses_reshaped = poses_reshaped.reshape(-1, poses_reshaped.shape[1] * poses_reshaped.shape[2]).T

    frames = poses_reshaped.shape[1]

    # ============== Input ============== #
    dct_n = 25
    if frames >= dct_n:
        inputs = dct.dct_2d(poses_reshaped)[:, :dct_n]
    else:
        inputs = dct.dct_2d(torch.nn.ZeroPad2d((0, dct_n - frames, 0, 0))(poses_reshaped))

    if is_cuda:
        inputs = inputs.cuda()

    # ============== Action recognition ============== #
    model_class = GCN_class()
    model_class.load_state_dict(torch.load('PoseCorrection/Data/model_class.pt'))

    if is_cuda:
        model_class.cuda()

    model_class.eval()
    with torch.no_grad():
        _, label = torch.max(model_class(inputs).data, 1)

    # ============== Motion correction ============== #
    model_corr = GCN_corr()
    model_corr.load_state_dict(torch.load('PoseCorrection/Data/model_corr.pt'))

    if is_cuda:
        model_corr.cuda()

    with torch.no_grad():
        model_corr.eval()
        deltas_dct, att = model_corr(inputs)

        if frames > dct_n:
            m = torch.nn.ZeroPad2d((0, frames - dct_n, 0, 0))
            deltas = dct.idct_2d(m(deltas_dct).transpose(1, 2))
        else:
            deltas = dct.idct_2d(deltas_dct[:, :frames].transpose(1, 2))

        poses_corrected = poses_reshaped + deltas.squeeze().squeeze().T

    # ============== Action recognition ============== #
    with torch.no_grad():
        _, label_corr = torch.max(model_class(inputs+deltas_dct).data, 1)

    return poses_reshaped, poses_corrected, label, label_corr

if __name__ == '__main__':

    with open('PoseCorrection/Data/pose_dict.pickle', 'rb') as f:
        pose_dict = pickle.load(f)

    video_folder = 'PoseCorrection/Data/Videos'

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(torch.device('cpu'))
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file, map_location=torch.device('cpu'))
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)

    results = {}
    files = [f for f in os.listdir(video_folder) if '.mp4' in f]
    for i, filename in enumerate(files):

        print(f'=============== {i+1}/{len(files)}: {filename} ================')
        results[filename[:-4]] = {}
        poses_reshaped, poses_corrected, label, label_corr = main(f'{video_folder}/{filename}', pose_dict, model)
        results[filename[:-4]]['poses_original'] = poses_reshaped
        results[filename[:-4]]['poses_corrected'] = poses_corrected
        results[filename[:-4]]['label_original'] = label
        results[filename[:-4]]['label_corrected'] = label_corr

    with open('PoseCorrection/Results/results.pickle', 'wb') as f:
        pickle.dump(results, f)