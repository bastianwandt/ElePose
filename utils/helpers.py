import torch
import torch.nn as nn
import numpy as np


def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
    p2ds = poses_2d / scale.mean()

    p2ds = p2ds * (1 / 10)

    return p2ds


def normalize_head_test(poses_2d, scale=145.5329587164913):  # ground truth
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]

    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds


def perspective_projection(pose_3d):

    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))

