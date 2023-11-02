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

import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import PareHead, SMPLHead
from .backbone.utils import get_backbone_info
from ..utils.train_utils import load_pretrained_model


class PARE(nn.Module):
    def __init__(
            self,
            num_joints=24,
            softmax_temp=1.0,
            num_features_smpl=64,
            backbone='resnet50',
            focal_length=5000.,
            img_res=224,
            pretrained=None,
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_heatmaps='',
            num_coattention_iter=1,
            coattention_conv='simple',
            deconv_conv_kernel_size=4,
            num_branch_iteration=0,
            num_deconv_layers=3,
            num_deconv_filters=256
    ):
        super(PARE, self).__init__()

        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(
                pretrained=True,
                downsample=False,
                use_conv=(use_conv == 'conv')
            )
        else:
            self.backbone = eval(backbone)(pretrained=True)

        self.head = PareHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(
                backbone)['n_output_channels'],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_heatmaps=use_heatmaps,
            backbone=backbone,
            num_coattention_iter=num_coattention_iter,
            coattention_conv=coattention_conv,
            num_branch_iteration=num_branch_iteration
        )

        self.smpl = SMPLHead(focal_length=focal_length, img_res=img_res)

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(
            self,
            images,
            cam_rotmat=None,
            cam_intrinsics=None,
            bbox_scale=None,
            bbox_center=None,
            img_w=None,
            img_h=None,
            gt_segm=None,
    ):
        ########## Get backbone features ##########
        features = self.backbone(images)
        hmr_output = self.head(features, gt_segm=gt_segm)

        if isinstance(hmr_output['pred_pose'], list):
            # if we have multiple smpl params prediction
            # create a dictionary of lists per prediction
            smpl_output = {
                'smpl_vertices': [],
                'smpl_joints3d': [],
                'smpl_joints2d': [],
                'pred_cam_t': [],
            }
            for idx in range(len(hmr_output['pred_pose'])):
                smpl_out = self.smpl(
                    rotmat=hmr_output['pred_pose'][idx],
                    shape=hmr_output['pred_shape'][idx],
                    cam=hmr_output['pred_cam'][idx],
                    normalize_joints2d=True,
                )
                for k, v in smpl_out.items():
                    smpl_output[k].append(v)
        else:
            smpl_output = self.smpl(
                rotmat=hmr_output['pred_pose'],
                shape=hmr_output['pred_shape'],
                cam=hmr_output['pred_cam'],
                normalize_joints2d=True,
            )
            smpl_output.update(hmr_output)

        return smpl_output

    def load_pretrained(self, file):
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)
        self.backbone.load_state_dict(state_dict, strict=False)
        load_pretrained_model(self.head, state_dict=state_dict,
                              strict=False, overwrite_shape_mismatch=True)
