import numpy as np
import threading
from time import time

import torch
import torch.nn
import torch.optim
import scipy.io as sio
from torch.utils import data
from data_h36m_fetch_all import H36MDataset
from data_3dhp_all import Load3DHPDataset, Load3DHPTestDataset
from utils.plotCMU2d import plotCMU2d
from utils.plot17j_1f import plot17j_1f
from utils.plot17j_1f_scaled import plot17j_1f_scaled
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch import autograd
import torch.nn as nn
from models_def import ScalePredictor, DepthAngleEstimator
from utils.plot_real17j_scaled import plot_real17j_1f_scaled
from utils.plot_real17j_scaled import plot_real17j_1f_scaled as plot3d_scaled
from utils.plot_real17j_2d import plot_real17j_1f_scaled_2d as plot2d
from utils.print_losses import print_losses
from utils.validate_h36m import validate
from types import SimpleNamespace
from utils.eval_functions import err_3dpe
#from pytorch3d.transforms import so3_exponential_map as rodrigues
#from pytorch3d.transforms import euler_angles_to_matrix
from utils.rotation_conversions import euler_angles_to_matrix
import torch.nn as nn
import random
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb
from numpy.random import default_rng
from sklearn.decomposition import PCA
from utils.plotCMU2d_set import plotCMU2d_set
import copy
import pickle
from defs import *  # contains the data and code folders

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
parser.add_argument("-n", "--num_bases", help="number of PCA bases",
                    type=int, default=26) # 23
parser.add_argument("-b", "--bl", help="bone lengths",
                    type=float, default=50.0)  # 50.0
parser.add_argument("-t", "--translation", help="camera translation",
                    type=float, default=10.0)
parser.add_argument("-r", "--rep2d", help="2d reprojection",
                    type=float, default=1.0)
parser.add_argument("-o", "--rot3d", help="3d reconstruction",
                    type=float, default=1.0)
parser.add_argument("-e", "--elevation", help="elevation reconstruction",
                    type=float, default=0.0)  # 1.0
parser.add_argument("-v", "--velocity", help="velocity",
                    type=float, default=1.0)

args = parser.parse_args()
num_bases = args.num_bases

wandb.init(project="unsupervised_pred")
wandb.run.name = "h36m_inn_pred_" + str(num_bases) + "_" + wandb.run.name

config = wandb.config
config.learning_rate = 0.0002  # 0.0001
config.BATCH_SIZE = 256
config.N_epochs = 100

config.use_elevation = False

#config.weight_likeli = 1  # not used!
config.weight_bl = float(args.bl)
#config.weight_depth = float(args.depth)
config.weight_2d = float(args.rep2d)
config.weight_3d = float(args.rot3d)
config.weight_velocity = float(args.velocity)
config.weight_elevation = float(args.elevation)
config.depth = float(args.translation)
config.use_gt = True

config.num_joints = 17
config.num_bases = num_bases

# config.datafile = '/home/wandt/research/self_supervision/data/h36m_train_mpi_skeleton_pred.pickle'
#config.datafile = data_folder + 'H36M/h36m_all.pickle'
config.datafile = data_folder + 'H36M/h36m_all_with_detections.pkl'
#config.datafile = data_folder + '3DHP/3dhp_train_all.pkl'

def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], 
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def symmetry_loss(poses):

    bl = get_bone_lengths_all(poses)

    bone_pairs = [[0, 3], [1, 4], [2, 5], [10, 13], [11, 14], [12, 15]]

    bp = bl[:, bone_pairs]

    loss = (bp[:, :, 0] - bp[:, :, 1]).square().sum(dim=1).mean()

    return loss


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


def perspective_projection(pose_3d, c=10.0):

    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))


# a simple chain of operations is collected by ReversibleSequential
inn_2d_1 = Ff.SequenceINN(num_bases)
for k in range(8):
    inn_2d_1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
inn_2d_1.cuda()

# ground truth
#inn_2d_1.load_state_dict(torch.load(
#    project_folder + 'models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))

#inn_2d_1.load_state_dict(torch.load(
#    project_folder + 'models/model_inn_h36m_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))

inn_2d_1.load_state_dict(torch.load(
    project_folder + 'models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))

for param in inn_2d_1.parameters():
    param.requires_grad = False

class LitLifter(pl.LightningModule):
    def __init__(self, pca, inn_2d):
        super(LitLifter, self).__init__()

        self.inn_2d_1 = inn_2d

        self.depth_estimator = DepthAngleEstimator(use_batchnorm=False, num_joints=17).cuda()

        self.bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                            0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()
        # 3DHP
        #self.bone_relations_mean = torch.Tensor([0.4806, 1.8415, 1.4971, 0.4806, 1.8379, 1.4971, 0.9078, 0.9938, 0.3468,
        #                                        0.6938, 0.5722, 1.2108, 0.9254, 0.5784, 1.2108, 0.9254]).cuda()

        self.pca = pca

        self.automatic_optimization = False

        self.metrics = Metrics()

        self.losses = SimpleNamespace()
        self.losses_mean = SimpleNamespace()

    #def forward(self, x):
    #    output = self.depth_estimator(x)
    #    return output

    def configure_optimizers(self):
        #params = list(self.depth_estimator.parameters())
        #optimizer = optim.Adam(params)

        optimizer = torch.optim.Adam(self.depth_estimator.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()

        poses_2d = train_batch['p2d_gt']

        # normalize each pose with its std and translate to root joint
        poses_3d = train_batch['poses_3d']
        poses_3d_normalized = poses_3d.reshape(-1, 3, 17) - poses_3d.reshape(-1, 3, 17)[:, :, [0]]
        poses_3d_normalized = poses_3d_normalized.reshape(-1, 51)

        inp_poses = poses_2d #+ torch.normal(torch.zeros_like(poses_2d), 0.001*torch.ones_like(poses_2d))

        pred, props, _ = self.depth_estimator(inp_poses)
        pred[:, 0] = 0.0

        x_ang_comp = torch.ones((inp_poses.shape[0], 1), device='cuda') * props[:, [0]]
        y_ang_comp = torch.zeros((inp_poses.shape[0], 1), device='cuda')
        z_ang_comp = torch.zeros((inp_poses.shape[0], 1), device='cuda')

        euler_angles_comp = torch.cat((x_ang_comp, y_ang_comp, z_ang_comp), dim=1)
        R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')

        if config.use_elevation:
            # sample from learned distribution
            elevation = torch.cat((props[:, 0].mean().reshape(1), props[:, 0].std().reshape(1)))
            x_ang = (-elevation[0]) + elevation[1] * torch.normal(torch.zeros((inp_poses.shape[0], 1), device='cuda'),
                                                         torch.ones((inp_poses.shape[0], 1), device='cuda'))
            #x_ang = (torch.rand((inp_poses.shape[0], 1), device='cuda') - 0.5) * (2*np.pi/9.0)
        else:
            # predefined distribution
            x_ang = (torch.rand((inp_poses.shape[0], 1), device='cuda') - 0.5) * 2.0 * (np.pi / 9.0)

        y_ang = (torch.rand((inp_poses.shape[0], 1), device='cuda') - 0.5) * 2.0 * np.pi
        z_ang = torch.zeros((inp_poses.shape[0], 1), device='cuda')
        #z_ang = angles[:, [3]] + angles[:, [4]] * torch.normal(torch.zeros((inp_poses.shape[0], 1), device='cuda'),
        #                                             torch.ones((inp_poses.shape[0], 1), device='cuda'))
        Rx = euler_angles_to_matrix(torch.cat((x_ang, z_ang, z_ang), dim=1), 'XYZ')
        Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')
        if config.use_elevation:
            R = Rx @ (Ry @ R_comp)
        else:
            R = Rx @ Ry

        depth = pred + config.depth
        depth[depth < 1.0] = 1.0
        pred_3d = torch.cat(((inp_poses.reshape(-1, 2, 17) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17)

        pred_3d = pred_3d.reshape(-1, 3, 17) - pred_3d.reshape(-1, 3, 17)[:, :, [0]]
        rot_poses = (R.matmul(pred_3d)).reshape(-1, 51)

        ## lift from augmented camera and normalize
        global_pose = torch.cat((rot_poses[:, 0:34], rot_poses[:, 34:51] + config.depth), dim=1)
        rot_2d = perspective_projection(global_pose)
        norm_poses = rot_2d

        norm_poses_mean = norm_poses[:, 0:34] - torch.Tensor(pca.mean_.reshape(1, 34)).cuda()
        latent = norm_poses_mean @ torch.Tensor(pca.components_.T).cuda()

        z, log_jac_det = self.inn_2d_1(latent[:, 0:num_bases])
        likelis = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

        self.losses.likeli = likelis.mean() #+ 0.1*likelis_kcs.mean()

        ## reprojection error
        pred_rot, props_rot, _ = self.depth_estimator(norm_poses[:, 0:34])
        pred_rot[:, 0] = 0.0

        pred_rot_depth = pred_rot + config.depth
        pred_rot_depth[pred_rot_depth < 1.0] = 1.0
        pred_3d_rot = torch.cat(
            ((norm_poses[:, 0:34].reshape(-1, 2, 17) * pred_rot_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth),
            dim=1)
        pred_3d_rot = pred_3d_rot.reshape(-1, 3, 17) - pred_3d_rot.reshape(-1, 3, 17)[:, :, [0]]

        self.losses.L3d = (rot_poses - pred_3d_rot.reshape(-1, 51)).norm(dim=1).mean()

        re_rot_3d = (R.permute(0, 2, 1) @ pred_3d_rot).reshape(-1, 51)
        pred_rot_global_pose = torch.cat((re_rot_3d[:, 0:34], re_rot_3d[:, 34:51] + config.depth), dim=1)
        re_rot_2d = perspective_projection(pred_rot_global_pose)
        norm_re_rot_2d = re_rot_2d

        self.losses.rep_rot = (norm_re_rot_2d - inp_poses).abs().sum(dim=1).mean()

        # TODO: use this again
        #self.losses.re_rot_3d = (pred_3d.reshape(-1, 51) - re_rot_3d).square().sum(dim=1).mean()

        num_pairs = int(np.floor(pred_3d.shape[0] / 2))
        pose_pairs = pred_3d[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_3d = re_rot_3d[0:(2*num_pairs)].reshape(-1, 2, 51)
        self.losses.re_rot_3d = ((pose_pairs[:, 0] - pose_pairs[:, 1]) - (pose_pairs_re_rot_3d[:, 0] - pose_pairs_re_rot_3d[:, 1])).norm(dim=1).mean()

        ## bone lengths prior
        bl = get_bone_lengths_all(pred_3d.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        self.losses.bl_prior = (self.bone_relations_mean - rel_bl).square().sum(dim=1).mean()
        #self.losses.bl_prior = rel_bl.std(dim=0).sum()/3.0

        self.losses.elevation_rot = (props_rot[:, [0]] + x_ang).square().mean()
        #bl_pairs = rel_bl.reshape(-1, 2, 16)
        #self.losses.bl_prior = (bl_pairs[:, 0] - bl_pairs[:, 1]).square().sum(dim=1).mean()
        #self.losses.bl_prior = (pred_bl - rel_bl).square().sum(dim=1).mean()

        #self.losses.bl_prior = symmetry_loss(pred_3d.reshape(-1, 51))

        #self.losses.min_bl = bl.square().sum(dim=1).mean()

        #self.losses.l2_reg = pred.square().sum(dim=1).mean()

        self.losses.loss = self.losses.likeli + \
                           config.weight_2d*self.losses.rep_rot + \
                           config.weight_3d * self.losses.L3d + \
                           config.weight_velocity*self.losses.re_rot_3d
                           # + \

        if config.use_elevation:
            self.losses.loss = self.losses.loss + config.weight_elevation*self.losses.elevation_rot

        #self.losses.loss = self.losses.loss + config.weight_bl * self.losses.bl_prior
        self.losses.loss = self.losses.loss + config.weight_bl*self.losses.bl_prior

        for key, value in self.losses.__dict__.items():
            if key not in self.losses_mean.__dict__.keys():
                self.losses_mean.__dict__[key] = []

            self.losses_mean.__dict__[key].append(value.item())

        self.manual_backward(self.losses.loss)
        opt.step()

        return self.losses.loss

    def validation_step(self, val_batch, batch_idx):

        if config.use_gt:
            test_poses_2dgt_normalized = val_batch['p2d_gt']
        else:
            test_poses_2dgt_normalized = val_batch['p2d_pred']

        test_3dgt_normalized = val_batch['poses_3d']

        inp_test_poses = test_poses_2dgt_normalized

        pred_test, props_test, _ = self.depth_estimator(inp_test_poses)
        pred_test[:, 0] = 0.0

        self.depth_estimator.train()

        pred_test_depth = pred_test + config.depth
        pred_test_poses = torch.cat(
            ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
             pred_test_depth), dim=1).detach().cpu().numpy()

        # rotate to camera coordinate system
        test_poses_cam_frame = pred_test_poses.reshape(-1, 3, 17)

        self.losses.pa = 0

        err_list = list()
        # err_mpjpe_list = list()
        for eval_cnt in range(int(test_3dgt_normalized.shape[0])):
            err = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                 pred_test_poses[eval_cnt].reshape(-1, 51),
                                 reflection='best')
            self.losses.pa += err
            err_list.append(err)

        self.losses.pa /= test_3dgt_normalized.shape[0]

        self.losses.mpjpe_scaled = mb().mpjpe(test_3dgt_normalized,
                                         torch.Tensor(test_poses_cam_frame).cuda(), num_joints=17,
                                         root_joint=0).mean().cpu().numpy()

        wandb.log({'epoch': self.current_epoch})

        for key, value in self.losses_mean.__dict__.items():
            wandb.log({key: np.mean(value)})

        self.losses_mean = SimpleNamespace()

        for key, value in self.losses.__dict__.items():
            self.log(key, value.item(), prog_bar=True)


    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()


# data
train_dataset = H36MDataset(config.datafile, normalize_2d=False, get_confidences=False, get_2dgt=True,
                            get_PCA=True, subjects=[1, 5, 6, 7, 8], normalize_func=normalize_head, only_one_camera=False)
if config.use_gt:
    test_dataset = H36MDataset(config.datafile, normalize_2d=False, get_confidences=False, get_2dgt=True,
                               get_PCA=False, subjects=[9, 11], normalize_func=normalize_head_test)
else:
    test_dataset = H36MDataset(config.datafile, normalize_2d=False, get_confidences=False, get_2dgt=False,
                               get_PCA=False, subjects=[9, 11], normalize_func=normalize_head_test)

#train_dataset = Load3DHPDataset(config.datafile, normalize_2d=False, get_confidences=False, get_2dgt=True,
#                         get_PCA=False, normalize_func=normalize_head)
#test_dataset = Load3DHPTestDataset('/scratch-ssd/data/3DHP/3dhp_test_all.pkl', normalize_2d=False, get_confidences=False, get_2dgt=True,
#                         get_PCA=False, normalize_func=normalize_head)

pca = train_dataset.pca

train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

test_loader = data.DataLoader(test_dataset, batch_size=10000, num_workers=0)

# model
model = LitLifter(pca, inn_2d_1)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0,
                     checkpoint_callback=False, logger=False, max_epochs=config.N_epochs)
trainer.fit(model, train_loader, test_loader)

