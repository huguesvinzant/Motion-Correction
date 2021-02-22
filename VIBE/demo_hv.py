# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import shutil
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from VIBE.lib.dataset.inference import Inference
from VIBE.lib.models.vibe import VIBE_Demo
from VIBE.lib.utils.demo_utils import video_to_images, download_ckpt

MIN_NUM_FRAMES = 25


def main_VIBE(video_dict_or_file, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_folder, num_frames, img_shape = video_to_images(video_dict_or_file, return_info=True)

    print(f'Input video number of frames {num_frames}')

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    # run multi object tracker
    mot = MPT(device=device, output_format='dict', yolo_img_size=256)
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    maximum = 0
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]
        elif tracking_results[person_id]['bbox'][:, 3].max() > maximum:
            bigbbox = person_id

    try:
        tracking_results = tracking_results[bigbbox]
    except UnboundLocalError:
        return None

    # ========= Define VIBE model ========= #
    # model = VIBE_Demo(
    #     seqlen=16,
    #     n_layers=2,
    #     hidden_size=1024,
    #     add_linear=True,
    #     use_residual=True,
    # ).to(device)

    # ========= Load pretrained weights ========= #
    # pretrained_file = download_ckpt(use_3dpw=False)
    # ckpt = torch.load(pretrained_file, map_location=device)
    # ckpt = ckpt['gen_state_dict']
    # model.load_state_dict(ckpt, strict=False)
    model.eval()
    # print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    joints2d = None

    bboxes = tracking_results['bbox']

    frames = tracking_results['frames']

    dataset = Inference(
        image_folder=image_folder,
        frames=frames,
        bboxes=bboxes,
        joints2d=joints2d,
        scale=bbox_scale,
    )

    dataloader = DataLoader(dataset)

    with torch.no_grad():

        pred_joints3d = []

        for batch in dataloader:
            batch = batch.unsqueeze(0)
            batch = batch.to(device)

            batch_size, seqlen = batch.shape[:2]
            output = model(batch)[-1]

            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

        pred_joints3d = torch.cat(pred_joints3d, dim=0)

        del batch

        vibe_results = pred_joints3d[:, :25, :].transpose(1, 2)

    del model

    shutil.rmtree(image_folder)

    return vibe_results

    # filename = os.path.abspath(os.path.join('tmp/', 'VIBE_result.pickle'))
    # if not os.path.exists(os.path.dirname(filename)):
    #     os.makedirs(os.path.dirname(filename))
    # with open(filename, 'wb') as f:
    #     pickle.dump(vibe_results, f)
    # print('================= END =================')
