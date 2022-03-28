import torch.nn as nn


class res_block(nn.Module):
    def __init__(self, num_neurons: int = 1024, use_batchnorm: bool = False):
        super(res_block, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.l1 = nn.Linear(num_neurons, num_neurons)
        self.bn1 = nn.BatchNorm1d(num_neurons)
        self.l2 = nn.Linear(num_neurons, num_neurons)
        self.bn2 = nn.BatchNorm1d(num_neurons)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = nn.LeakyReLU()(self.l2(x))
        if self.use_batchnorm:
            x = self.bn2(x)
        x += inp

        return x

class DepthAngleEstimator(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=16):
        super(DepthAngleEstimator, self).__init__()

        self.upscale = nn.Linear(2*num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm)
        self.depth = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)
        #self.angles.bias.data[1] = 10.0

    def forward(self, x):

        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = self.depth(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = self.angles(xa)

        return xd, xa

