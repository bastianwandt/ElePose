import threading
from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np
import scipy.io as sio
# import data
from torch.utils import data
from data_h36m_fetch_unsupervised import H36MDataset
from utils.plotCMU2d import plotCMU2d
from utils.plot17j_1f import plot17j_1f
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch import autograd
import torch.nn as nn
from models_def import Lifter, DepthEstimator, Discriminator, DepthOnlyEstimator
from utils.plot17j_1f_procrustes import plot as plot
from utils.print_losses import print_losses
from utils.validate_h36m import validate
from types import SimpleNamespace
from utils.eval_functions import err_3dpe
from pytorch3d.transforms import so3_exponential_map as rodrigues
from pytorch3d.transforms import euler_angles_to_matrix
import torch.nn as nn
import random
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb
from numpy.random import default_rng
from sklearn.decomposition import PCA
from utils.plotCMU2d_set import plotCMU2d_set
import copy
from utils.moment_score import MMS as mms

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
parser.add_argument("-n", "--num_bases", help="number of PCA bases",
                    type=int, default=10)

args = parser.parse_args()
num_bases = args.num_bases

wandb.init(project="unsupervised")
wandb.run.name = "h36m_pretrain_inn_pca_bases_" + str(num_bases)

config = wandb.config
config.learning_rate = 0.0001  #0.0001
config.BATCH_SIZE = 4*64
config.N_epochs = 100

config.num_bases = num_bases

config.datafile = '/ubc/cs/home/w/wandt/data/H36M/h36m_train_mpi_skeleton_pred_3dgt_cam_frame.pickle'

def get_bone_lengths(poses):
    bone_map = [[0, 1], [1, 2], [3, 4], [4, 5], [10, 11], [11, 12], [13, 14], [14, 15], [0, 6], [3, 6], [6, 7]]

    poses = poses.reshape((-1, 3, 16))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def symmetry_loss(poses):

    p3d = poses.reshape((-1, 3, 16))
    r_up_leg = torch.norm(p3d[:, :, 0] - p3d[:, :, 1], p=2, dim=1)
    r_low_leg = torch.norm(p3d[:, :, 1] - p3d[:, :, 2], p=2, dim=1)
    l_up_leg = torch.norm(p3d[:, :, 3] - p3d[:, :, 4], p=2, dim=1)
    l_low_leg = torch.norm(p3d[:, :, 4] - p3d[:, :, 5], p=2, dim=1)

    r_up_arm = torch.norm(p3d[:, :, 10] - p3d[:, :, 11], p=2, dim=1)
    r_low_arm = torch.norm(p3d[:, :, 11] - p3d[:, :, 12], p=2, dim=1)
    l_up_arm = torch.norm(p3d[:, :, 13] - p3d[:, :, 14], p=2, dim=1)
    l_low_arm = torch.norm(p3d[:, :, 14] - p3d[:, :, 15], p=2, dim=1)

    r_hip = torch.norm(p3d[:, :, 0] - p3d[:, :, 6], p=2, dim=1)
    l_hip = torch.norm(p3d[:, :, 3] - p3d[:, :, 6], p=2, dim=1)

    r_shoulder = torch.norm(p3d[:, :, 7] - p3d[:, :, 10], p=2, dim=1)
    l_shoulder = torch.norm(p3d[:, :, 7] - p3d[:, :, 13], p=2, dim=1)

    leg_symmetry = mse_loss(r_up_leg, l_up_leg) + mse_loss(r_low_leg, l_low_leg)
    arm_symmetry = mse_loss(r_up_arm, l_up_arm) + mse_loss(r_low_arm, l_low_arm)
    hip_symmetry = mse_loss(r_hip, l_hip)
    shoulder_symmetry = mse_loss(r_shoulder, l_shoulder)

    loss = leg_symmetry + arm_symmetry + hip_symmetry + shoulder_symmetry

    return loss


def loss_weighted_rep_no_scale(p2d, p3d, confs):

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d

    loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss


my_dataset = H36MDataset(config.datafile, normalize_2d=True, get_3d_sfm=False, get_confidences=False, get_2dgt=True, subjects=[1, 5, 6, 7, 8])
train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

# quick and dirty test data loading
test_data = np.load('/ubc/cs/home/w/wandt/data/H36M/h36m_test_mpi_skeleton_pred_3dgt_cam_frame.npy', allow_pickle=True).item()

cams = ['54138969', '55011271', '58860488', '60457274']

for cam in ['54138969', '55011271', '58860488', '60457274']:
    test_data['poses_2d_pred'][cam] = \
        (test_data['poses_2d_pred'][cam].reshape(-1, 2, 16) -
         test_data['poses_2d_pred'][cam].reshape(-1, 2, 16).mean(axis=2, keepdims=True)).reshape(-1, 32)
    # poses_2d[cam] /= poses_2d[cam].std(dim=1, keepdim=True)
    norm = np.linalg.norm(test_data['poses_2d_pred'][cam], ord=2, axis=1, keepdims=True)

    test_data['poses_2d_pred'][cam] /= norm
    test_data['poses_2d_pred'][cam][:, 16:32] = -test_data['poses_2d_pred'][cam][:, 16:32]

    test_data['poses_3d'][cam] = (test_data['poses_3d'][cam].reshape(-1, 3, 16) - test_data['poses_3d'][cam].reshape(-1, 3, 16)[:, :, 6:7]).reshape(-1, 48)

test_3dgt = torch.zeros((test_data['poses_3d']['54138969'].shape[0] * 4, 48)).cuda()
test_poses = torch.zeros((test_data['poses_2d_pred']['54138969'].shape[0] * 4, 32)).cuda()
test_poses_2dgt = torch.zeros((test_data['poses_2d']['54138969'].shape[0] * 4, 32)).cuda()
test_confidences = torch.zeros((test_data['confidences']['54138969'].shape[0] * 4, 16)).cuda()

cnt = 0
for b in range(test_data['poses_2d_pred']['54138969'].shape[0]):
    for c_idx, cam in enumerate(test_data['poses_2d_pred']):
        test_3dgt[cnt] = torch.Tensor(test_data['poses_3d'][cam][b])
        test_poses[cnt] = torch.Tensor(test_data['poses_2d_pred'][cam][b])
        test_poses_2dgt[cnt] = torch.Tensor(test_data['poses_2d'][cam][b])
        test_confidences[cnt] = torch.Tensor(test_data['confidences'][cam][b])
        cnt += 1

test_3dgt = test_3dgt.cpu().numpy()

#test_poses_2dgt[:, 16:] = -test_poses_2dgt[:, 16:]
test_poses_2dgt_normalized = (test_poses_2dgt.reshape(-1, 2, 16) - test_poses_2dgt.reshape(-1, 2, 16).mean(axis=-1, keepdims=True)).reshape(-1, 32)
test_poses_2dgt_normalized /= test_poses_2dgt_normalized.norm(p=2, dim=1, keepdim=True)

test_poses_normalized = (test_poses.reshape(-1, 2, 16) - test_poses.reshape(-1, 2, 16).mean(axis=-1, keepdims=True)).reshape(-1, 32)
test_poses_normalized /= test_poses_normalized.norm(p=2, dim=1, keepdim=True)


# align test_3dgt to camera frame
test_3dgt_normalized = test_3dgt.reshape(-1, 3, 16) - test_3dgt.reshape(-1, 3, 16).mean(axis=2, keepdims=True)
test_3dgt_normalized = test_3dgt_normalized.reshape(-1, 48)
test_3dgt_normalized_pt = torch.Tensor(test_3dgt_normalized).reshape(-1, 3, 16).cuda()

#model_skel_morph = torch.load('/home/wandt/research/unsupervised_pose_estimation/models/tmp/model_morpher_h36m_pretrain_inn_autumn-oath-670.pt')
#model_skel_morph = model_skel_morph.cuda()
#model_skel_morph.eval()

mse_loss = nn.MSELoss()

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024,  dims_out))


# a simple chain of operations is collected by ReversibleSequential
inn_2d = Ff.SequenceINN(num_bases)
for k in range(8):
    inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
inn_2d.cuda()

#inn_3d = Ff.SequenceINN(48)
#for k in range(8):
#    inn_3d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
#inn_3d.cuda()

#depth_estimator = DepthOnlyEstimator().cuda()

bone_lengths_mean = torch.Tensor([449.36988144, 445.37088404, 449.36829191, 445.3704963 ,
       279.75453686, 249.47456061, 279.75133987, 249.47533542,
       134.10583498, 134.10507151, 485.2253686 ]).cuda()/1000.0
bone_relations = bone_lengths_mean / bone_lengths_mean[0]

import pickle
pca = pickle.load(open("models/pca_gt.pkl", "rb"))
data_statistics = pickle.load(open("models/data_statistics.pkl", "rb"))
data_mean = data_statistics['mean']
data_cov = data_statistics['cov']

N_epochs = 110

params = list(inn_2d.parameters())
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

torch.autograd.set_detect_anomaly(True)

print('start training with %d PCA bases' % num_bases)

metrics = Metrics()

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

for epoch in range(N_epochs):
    tic = time()
    for i, sample in enumerate(train_loader):

        #poses_2d = {key:sample[key] for key in all_cams}
        poses_2d = sample['p2d_gt']

        # normalize each pose with its std and translate to root joint

        poses_3d = sample['poses_3d']
        poses_3d_normalized = poses_3d.reshape(-1, 3, 16) - poses_3d.reshape(-1, 3, 16).mean(axis=2, keepdims=True)
        poses_3d_normalized = poses_3d_normalized.reshape(-1, 48)

        # perform PCA
        inp_poses = torch.Tensor(pca.transform(poses_2d.detach().cpu().numpy())[:, 0:num_bases]).cuda()

        z_2d, log_jac_det_2d = inn_2d(inp_poses)

        likeli = (0.5 * torch.sum(z_2d ** 2, 1) - log_jac_det_2d)
        losses.dist_2d = likeli.mean()

        losses.loss = 1 * losses.dist_2d

        optimizer.zero_grad()
        losses.loss.backward()
        optimizer.step()

        for key, value in losses.__dict__.items():
            if key not in losses_mean.__dict__.keys():
                losses_mean.__dict__[key] = []

            losses_mean.__dict__[key].append(value.item())

        if not i % 100:
            with torch.no_grad():
                eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                inp_test_poses = torch.Tensor(
                    pca.transform(test_poses_2dgt_normalized.detach().cpu().numpy())[:, 0:num_bases]).cuda()

                pred_latent, log_jac_det_test = inn_2d(inp_test_poses)

                likeli_test = (0.5 * torch.sum(pred_latent ** 2, 1) - log_jac_det_test)
                losses.test_likeli = likeli_test.mean()

                # randomly sample poses
                z_test = torch.randn((25, num_bases), device='cuda')
                #test_rec_poses, ljd = inn_2d(z_test, rev=True)
                pred_pca_space, _ = inn_2d(z_test, rev=True)
                pred_pca_space_32 = torch.cat((pred_pca_space, torch.zeros((25, 32-num_bases), device='cuda')), dim=1)

                with torch.no_grad():
                    test_rec_poses = pca.inverse_transform(pred_pca_space_32.cpu().numpy())

                losses.mms = mms(torch.Tensor(test_rec_poses), data_mean, data_cov)

                #plotCMU2d_set(test_rec_poses)

            print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(i % 1000))

            '''
            # these are debugging plot functions
            p_idx = 0
            with torch.no_grad():
                scale = poses_3d_normalized.reshape(-1, 48)[p_idx].norm(p=2) / pred_3d.reshape(-1, 48)[p_idx].norm(p=2)
                p_pred = pred_3d.reshape(-1, 3, 16)[p_idx] - pred_3d.reshape(-1, 3, 16)[p_idx][:, [6]]
                p3d_n = poses_3d_normalized.reshape(-1, 3, 16)[p_idx] - poses_3d_normalized.reshape(-1, 3, 16)[p_idx][:,
                                                                        [6]]
                plot17j_1f(p_pred.reshape(-1, 48).cpu().numpy() * scale.cpu().numpy(), p3d_n.reshape(48).cpu().numpy())
            
            with torch.no_grad():
                plotCMU2d_set(test_poses.cpu().numpy())
            
            with torch.no_grad():
                plotCMU2d_set(pred_test_poses)
            
            p_idx = 1
            with torch.no_grad():
                plot(pred_test_poses[p_idx:p_idx + 1],
                     test_3dgt_normalized[p_idx:p_idx + 1])
                    
            p_idx = 1
            with torch.no_grad():
                plot(pred_3d[[p_idx]].cpu().numpy(),
                     poses_3d_normalized[[p_idx]].cpu().numpy())
                     
            p_idx = 0
            with torch.no_grad():
                plotCMU2d((pred_test_poses[p_idx]),
                test_poses[p_idx].cpu().numpy())
            
            import pickle
            res = dict()
            res['gt'] = test_3dgt_normalized
            res['pred'] = pred_test.cpu().numpy()
            pickle.dump(res, open( "results/results2.pkl", "wb" ) )
            '''

            if not (epoch == 0 and i == 0):
                for key, value in losses_mean.__dict__.items():
                    wandb.log({key: np.mean(value)})

            losses_mean = SimpleNamespace()

            '''
            # these are debugging plot functions
            
            p_idx = 0
            with torch.no_grad():
               plot17j_1f(pred_poses[p_idx:p_idx + 1].cpu().numpy())

            p_idx = 0
            with torch.no_grad():
                plot(pred_poses[4*p_idx:4*p_idx + 1].cpu().numpy(), 
                    poses_3d[p_idx:p_idx + 1].cpu().numpy())
                    
            p_idx = 0
            with torch.no_grad():
                plotCMU2d((pred_3d_online[p_idx]).cpu().numpy(),
                inp_poses[p_idx].cpu().numpy())
            
            # visualize training data
            p_idx = 0  # only works for p_idx=0
            with torch.no_grad():
                scale = poses_3d_normalized.reshape(-1, 48)[p_idx].norm(p=2) / rot_poses.reshape(-1, 48)[p_idx].norm(p=2)
                p_pred = rot_poses.reshape(-1, 3, 16)[p_idx] - rot_poses.reshape(-1, 3, 16)[p_idx][:, [6]]
                p3d_n = poses_3d_normalized.reshape(-1, 3, 16)[p_idx] - poses_3d_normalized.reshape(-1, 3, 16)[p_idx][:, [6]]
                plot17j_1f(p_pred.reshape(-1, 48).cpu().numpy() * scale.cpu().numpy(), p3d_n.reshape(48).cpu().numpy())
            
            p_idx = 0
            with torch.no_grad():
                scale = inp_poses.reshape(-1, 32)[p_idx].norm(p=2) / rot_poses.reshape(-1, 48)[p_idx,0:32].norm(p=2)
                p_rep = rot_poses.reshape(-1, 3, 16)[p_idx] - rot_poses.reshape(-1, 3, 16)[p_idx,:,6:7]
                plotCMU2d(p_rep.reshape(-1, 48)[0:32].cpu().numpy() * scale.cpu().numpy(), inp_poses[p_idx].cpu().numpy())
            '''

    #torch.save(lifter, 'models/tmp/model_lifter_' + wandb.run.name + '.pt')
    #wandb.save('models/model_' + wandb.run.name + '.pt')
    torch.save(inn_2d.state_dict(), 'models/model_inn_' + wandb.run.name + '.pt')

    #scheduler.step()

print('done')
