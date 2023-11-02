import torch
import torch.nn as nn
from .head import SMPLHead

from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d


class DummyModel(nn.Module):
    def __init__(
            self,
            img_res=224,
    ):
        super(DummyModel, self).__init__()

        npose = 24 * 6

        # this is a quite dumb model
        # it does avg pooling on the input image
        # and apply seperate mlps to regress SMPL
        # pose, shape, and cam params
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.decpose = nn.Linear(192, npose)
        self.decshape = nn.Linear(192, 10)
        self.deccam = nn.Linear(192, 3)

        # SMPLHead takes estimated pose, shape, cam parameters as input
        # and returns the 3D mesh vertices, 3D/2D joints as output
        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )

    def forward(self, images):

        batch_size = images.shape[0]

        features = self.avg_pool(images).reshape(batch_size, -1)

        pred_pose = self.decpose(features)
        pred_shape = self.decshape(features)
        pred_cam = self.deccam(features)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        smpl_output = self.smpl(
            rotmat=pred_rotmat,
            shape=pred_shape,
            cam=pred_cam,
            normalize_joints2d=True,
        )

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
        }
        smpl_output.update(output)

        # smpl output keys and valus:
        #         "smpl_vertices": torch.Size([batch_size, 6890, 3]), -> 3D mesh vertices
        #         "smpl_joints3d": torch.Size([batch_size, 49, 3]), -> 3D joints
        #         "smpl_joints2d": torch.Size([batch_size, 49, 2]), -> 2D joints
        #         "pred_cam_t": torch.Size([batch_size, 3]), -> camera translation [x,y,z]
        #         "pred_pose": torch.Size([batch_size, 24, 3, 3]), -> SMPL pose params in rotation matrix form
        #         "pred_cam": torch.Size([batch_size, 3]), -> weak perspective camera [s,tx,ty]
        #         "pred_shape": torch.Size([batch_size, 10]), -> SMPL shape (betas) params
        #         "pred_pose_6d": torch.Size([batch_size, 144]), -> SMPL pose params in 6D rotation form

        return smpl_output