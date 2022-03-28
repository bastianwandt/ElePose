import torch.nn
import torch.optim
from torch.utils import data
import pytorch_lightning as pl

from data_h36m_fetch_all import H36MDataset
from models_def import DepthAngleEstimator
from types import SimpleNamespace
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb

from defs import *  # contains the data and code folders
from utils.helpers import *

# https://github.com/VLL-HD/FrEIA
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse


parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
parser.add_argument("-n", "--num_bases", help="number of PCA bases",
                    type=int, default=26)
parser.add_argument("-b", "--bl", help="bone lengths",
                    type=float, default=50.0)  # 50.0
parser.add_argument("-t", "--translation", help="camera translation",
                    type=float, default=10.0)
parser.add_argument("-r", "--rep2d", help="2d reprojection",
                    type=float, default=1.0)
parser.add_argument("-o", "--rot3d", help="3d reconstruction",
                    type=float, default=1.0)
parser.add_argument("-v", "--velocity", help="velocity",
                    type=float, default=1.0)

args = parser.parse_args()
num_bases = args.num_bases

wandb.init(project="ElePose")
wandb.run.name = "h36m_inn_pred_" + str(num_bases) + "_" + wandb.run.name

config = wandb.config
config.learning_rate = 0.0002
config.BATCH_SIZE = 256
config.N_epochs = 100

config.use_elevation = True

config.weight_bl = float(args.bl)
config.weight_2d = float(args.rep2d)
config.weight_3d = float(args.rot3d)
config.weight_velocity = float(args.velocity)
config.depth = float(args.translation)
config.use_gt = True

config.num_joints = 17
config.num_bases = num_bases

config.datafile = data_folder + 'H36M/h36m_all_with_detections.pkl'

## load pretrained INN
# a simple chain of operations is collected by ReversibleSequential
inn_2d_1 = Ff.SequenceINN(num_bases)
for k in range(8):
    inn_2d_1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

inn_2d_1.load_state_dict(torch.load(
    project_folder + 'models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))
# freeze all weights in INN
for param in inn_2d_1.parameters():
    param.requires_grad = False

class LitLifter(pl.LightningModule):
    def __init__(self, pca, inn_2d):
        super(LitLifter, self).__init__()

        self.inn_2d = inn_2d.to(self.device)

        self.depth_estimator = DepthAngleEstimator(use_batchnorm=False, num_joints=17).cuda()

        self.bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                            0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()

        self.pca = pca

        self.automatic_optimization = False

        self.metrics = Metrics()

        self.losses = SimpleNamespace()
        self.losses_mean = SimpleNamespace()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.depth_estimator.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR  (optimizer=optimizer, gamma=0.95)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()

        inp_poses = train_batch['p2d_gt']

        pred, props = self.depth_estimator(inp_poses)
        pred[:, 0] = 0.0

        x_ang_comp = torch.ones((inp_poses.shape[0], 1), device=self.device) * props
        y_ang_comp = torch.zeros((inp_poses.shape[0], 1), device=self.device)
        z_ang_comp = torch.zeros((inp_poses.shape[0], 1), device=self.device)

        euler_angles_comp = torch.cat((x_ang_comp, y_ang_comp, z_ang_comp), dim=1)
        R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')

        if config.use_elevation:
            # sample from learned distribution
            elevation = torch.cat((props.mean().reshape(1), props.std().reshape(1)))
            x_ang = (-elevation[0]) + elevation[1] * torch.normal(torch.zeros((inp_poses.shape[0], 1), device=self.device),
                                                         torch.ones((inp_poses.shape[0], 1), device=self.device))
        else:
            # predefined distribution
            x_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 2.0 * (np.pi / 9.0)

        y_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 2.0 * np.pi
        z_ang = torch.zeros((inp_poses.shape[0], 1), device=self.device)
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

        z, log_jac_det = self.inn_2d(latent[:, 0:num_bases])
        likelis = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

        self.losses.likeli = likelis.mean()

        ## reprojection error
        pred_rot, _ = self.depth_estimator(norm_poses[:, 0:34])
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

        # pairwise deformation loss
        num_pairs = int(np.floor(pred_3d.shape[0] / 2))
        pose_pairs = pred_3d[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_3d = re_rot_3d[0:(2*num_pairs)].reshape(-1, 2, 51)
        self.losses.re_rot_3d = ((pose_pairs[:, 0] - pose_pairs[:, 1]) - (pose_pairs_re_rot_3d[:, 0] - pose_pairs_re_rot_3d[:, 1])).norm(dim=1).mean()

        ## bone lengths prior
        bl = get_bone_lengths_all(pred_3d.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        self.losses.bl_prior = (self.bone_relations_mean - rel_bl).square().sum(dim=1).mean()

        self.losses.loss = self.losses.likeli + \
                           config.weight_2d*self.losses.rep_rot + \
                           config.weight_3d * self.losses.L3d + \
                           config.weight_velocity*self.losses.re_rot_3d

        self.losses.loss = self.losses.loss + config.weight_bl*self.losses.bl_prior

        # logging
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

        pred_test, _ = self.depth_estimator(inp_test_poses)
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
        for eval_cnt in range(int(test_3dgt_normalized.shape[0])):
            err = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                 pred_test_poses[eval_cnt].reshape(-1, 51),
                                 reflection='best')
            self.losses.pa += err
            err_list.append(err)

        self.losses.pa /= test_3dgt_normalized.shape[0]

        self.losses.mpjpe_scaled = mb().mpjpe(test_3dgt_normalized,
                                         torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17,
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
train_dataset = H36MDataset(config.datafile, normalize_2d=False, get_2dgt=True,
                            get_PCA=True, subjects=[1, 5, 6, 7, 8], normalize_func=normalize_head)
if config.use_gt:
    test_dataset = H36MDataset(config.datafile, normalize_2d=False, get_2dgt=True,
                               get_PCA=False, subjects=[9, 11], normalize_func=normalize_head_test)
else:
    test_dataset = H36MDataset(config.datafile, normalize_2d=False, get_2dgt=False,
                               get_PCA=False, subjects=[9, 11], normalize_func=normalize_head_test)

pca = train_dataset.pca

train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

test_loader = data.DataLoader(test_dataset, batch_size=10000, num_workers=0)

# model
model = LitLifter(pca, inn_2d_1)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0,
                     checkpoint_callback=False, logger=False, max_epochs=config.N_epochs)
trainer.fit(model, train_loader, test_loader)

