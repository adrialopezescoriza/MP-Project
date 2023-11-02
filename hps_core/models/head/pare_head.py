import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from ...core.config import SMPL_MEAN_PARAMS
from ...layers.attention_images import SelfAttentionImages
from ...utils.geometry import rot6d_to_rotmat, get_coord_maps
from ...utils.kp_utils import get_smpl_neighbor_triplets
from ...layers.softargmax import softargmax2d, get_heatmap_preds
from ...layers import LocallyConnected2d, KeypointAttention, interpolate
from ..backbone.resnet import conv3x3, conv1x1, BasicBlock

BN_MOMENTUM = 0.1


class PareHead(nn.Module):
    def __init__(
            self,
            num_joints,
            num_input_features,
            softmax_temp=1.0,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            num_camera_params=3,
            num_features_smpl=64,
            final_conv_kernel=1,
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_heatmaps='',
            backbone='resnet',
            num_coattention_iter=1,
            coattention_conv='simple', # 'double_1', 'double_3', 'single_1', 'single_3', 'simple'
            num_branch_iteration=0
    ):
        super(PareHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = False
        self.use_heatmaps = use_heatmaps
        self.num_coattention_iter = num_coattention_iter
        self.coattention_conv = coattention_conv
        self.num_branch_iteration = num_branch_iteration
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size

        self.num_input_features = num_input_features

        # Part Branch Estimating 2D Keypoints
        conv_fn = self._make_deconv_layer

        self.keypoint_deconv_layers = conv_fn(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )
        # reset inplanes to 2048 -> final resnet layer
        self.num_input_features = num_input_features
        self.smpl_deconv_layers = conv_fn(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        # Computes attention across part heatmaps
        self.cross_heatmap_attention = SelfAttentionImages(in_dim=num_joints)

        pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        shape_mlp_inp_dim = num_joints * smpl_final_dim

        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints+1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        self.smpl_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=smpl_final_dim,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        # temperature for softargmax function
        self.register_buffer('temperature', torch.tensor(softmax_temp))

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.pose_mlp_inp_dim = pose_mlp_inp_dim
        self.shape_mlp_inp_dim = shape_mlp_inp_dim

        self.keypoint_attention = KeypointAttention(
            use_conv=False,
            in_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            out_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            act='softmax',
            use_scale=False,
        )

        # here we use 2 different MLPs to estimate shape and camera
        # They take a channelwise downsampled version of smpl features
        self.shape_mlp = self._get_shape_mlp(output_size=10)
        self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)

        # for pose each joint has a separate MLP
        # weights for these MLPs are not shared
        # hence we use Locally Connected layers
        self.pose_mlp = self._get_pose_mlp(num_joints=num_joints, output_size=6)


    def _get_shape_mlp(self, output_size):
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)

        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(
                    nn.Linear(self.shape_mlp_inp_dim, self.shape_mlp_hidden_size)
                )
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, output_size)
                )
            else:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, self.shape_mlp_hidden_size)
                )
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        if self.pose_mlp_num_layers == 1:
            return LocallyConnected2d(
                in_channels=self.pose_mlp_inp_dim,
                out_channels=output_size,
                output_size=[num_joints, 1],
                kernel_size=1,
                stride=1,
            )

        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_inp_dim,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=output_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            else:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
        return nn.Sequential(*module_list)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_res_conv_layers(self, input_channels, num_channels=64,
                              num_heads=1, num_basic_blocks=2):
        head_layers = []

        # kernel_sizes, strides, paddings = self._get_trans_cfg()
        # for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
        head_layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        )

        for i in range(num_heads):
            layers = []
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        return nn.Sequential(*head_layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_upsample_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_layers is different len(num_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_layers is different len(num_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(
                nn.Conv2d(in_channels=self.num_input_features, out_channels=planes,
                          kernel_size=kernel, stride=1, padding=padding, bias=self.deconv_with_bias)
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _prepare_pose_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[0], pred_pose.shape[2]

        joint_triplets = get_smpl_neighbor_triplets()

        inp_list = []

        for inp_type in self.pose_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'neighbor_pose_feats':
                # add the image features from neighboring joints
                n_pose_feat = []
                for jt in joint_triplets:
                    n_pose_feat.append(
                        feats[:, :, jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose_feat = torch.cat(n_pose_feat, 2)
                inp_list.append(n_pose_feat)

            if inp_type == 'self_pose':
                # add image features
                inp_list.append(pred_pose)

            if inp_type == 'all_pose':
                # append all of the joint angels
                all_pose = pred_pose.reshape(batch_size, -1, 1)[..., None].repeat(1, 1, num_joints, 1)
                inp_list.append(all_pose)

            if inp_type == 'neighbor_pose':
                # append only the joint angles of neighboring ones
                n_pose = []
                for jt in joint_triplets:
                    n_pose.append(
                        pred_pose[:,:,jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose = torch.cat(n_pose, 2)
                inp_list.append(n_pose)

            if inp_type == 'shape':
                # append shape predictions
                pred_shape = pred_shape[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                pred_cam = pred_cam[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        return torch.cat(inp_list, 1)

    def _prepare_shape_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[:2]

        inp_list = []

        for inp_type in self.shape_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'all_pose':
                # append all of the joint angels
                pred_pose = pred_pose.reshape(batch_size, -1)
                inp_list.append(pred_pose)

            if inp_type == 'shape':
                # append shape predictions
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        return torch.cat(inp_list, 1)

    def forward(self, features, gt_segm=None):
        batch_size = features.shape[0]

        init_pose = self.init_pose.expand(batch_size, -1)  # N, Jx6
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        output = {}

        ############## 2D PART BRANCH FEATURES ##############
        part_feats = self._get_2d_branch_feats(features)

        ############## GET PART ATTENTION MAP ##############
        part_attention, _ = self._get_part_attention_map(part_feats, output)

        ############## COMPUTE CROSS ATTENTION BETWEEN PARTS ##############
        part_attention,_ = self.cross_heatmap_attention(part_attention)

        ############## 3D SMPL BRANCH FEATURES ##############
        smpl_feats = self._get_3d_smpl_feats(features, part_feats)

        ############## GET LOCAL FEATURES ##############
        point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)

        ############## GET FINAL PREDICTIONS ##############
        pred_pose, pred_shape, pred_cam = self._get_final_preds(
            point_local_feat, cam_shape_feats, init_pose, init_shape, init_cam
        )

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)

        output.update({
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        })
        return output

    def _get_local_feats(self, smpl_feats, part_attention, output):
        cam_shape_feats = self.smpl_final_layer(smpl_feats)

        point_local_feat = self.keypoint_attention(smpl_feats, part_attention)
        cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)

        return point_local_feat, cam_shape_feats

    def _get_2d_branch_feats(self, features):
        part_feats = self.keypoint_deconv_layers(features)
        return part_feats

    def _get_3d_smpl_feats(self, features, part_feats):
        smpl_feats = self.smpl_deconv_layers(features)
        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):
        heatmaps = self.keypoint_final_layer(part_feats)

        # returns coords between [-1,1]
        pred_kp2d, normalized_heatmaps = softargmax2d(heatmaps, self.temperature)
        output['pred_kp2d'] = pred_kp2d
        output['pred_heatmaps_2d'] = heatmaps

        return heatmaps, normalized_heatmaps

    def _get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam):
        pose_feats = pose_feats.unsqueeze(-1)  #

        if init_pose.shape[-1] == 6:
            # This means init_pose comes from a previous iteration
            init_pose = init_pose.transpose(2,1).unsqueeze(-1)
        else:
            # This means init pose comes from mean pose
            init_pose = init_pose.reshape(init_pose.shape[0], 6, -1).unsqueeze(-1)

            shape_feats = cam_shape_feats
            shape_feats = cam_shape_feats

        shape_feats = cam_shape_feats

        shape_feats = torch.flatten(shape_feats, start_dim=1)

        pred_pose = self.pose_mlp(pose_feats)
        pred_cam = self.cam_mlp(shape_feats)
        pred_shape = self.shape_mlp(shape_feats)

        pred_pose = pred_pose.squeeze(-1).transpose(2, 1) # N, J, 6
        return pred_pose, pred_shape, pred_cam