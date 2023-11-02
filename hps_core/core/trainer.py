import os
import cv2
import json
import torch
import joblib
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import config
from . import constants
from ..losses import HMRLoss
from ..models import SMPL
from ..utils.renderer import Renderer
from ..models.dummy_model import DummyModel
from ..models.pare import PARE
from ..dataset import MixedDataset, BaseDataset
from ..utils.vis_utils import color_vertices_batch
from ..utils.train_utils import set_seed
from ..utils.image_utils import denormalize_images
from ..utils.eval_utils import reconstruction_error, compute_error_verts
from ..utils.geometry import estimate_translation, convert_weak_perspective_to_perspective


class HPSTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(HPSTrainer, self).__init__()

        self.hparams.update(hparams)

        # from ..models.hmr import HMR
        # self.model = HMR()
        # self.model = DummyModel(
        #     img_res=self.hparams.DATASET.IMG_RES,
        # )
        #self.model = PARE(img_res=self.hparams.DATASET.IMG_RES, backbone='resnet50')
        self.model = PARE(img_res=self.hparams.DATASET.IMG_RES, backbone='hrnet_w32-conv')

        # there are many hyperparameters for the loss function
        # but in my experience default ones are the optimal values
        # so you don't really need to spend time on them
        self.loss_fn = HMRLoss(
            shape_loss_weight=self.hparams.HMR.SHAPE_LOSS_WEIGHT,
            keypoint_loss_weight=self.hparams.HMR.KEYPOINT_LOSS_WEIGHT,
            pose_loss_weight=self.hparams.HMR.POSE_LOSS_WEIGHT,
            beta_loss_weight=self.hparams.HMR.BETA_LOSS_WEIGHT,
            openpose_train_weight=self.hparams.HMR.OPENPOSE_TRAIN_WEIGHT,
            gt_train_weight=self.hparams.HMR.GT_TRAIN_WEIGHT,
            loss_weight=self.hparams.HMR.LOSS_WEIGHT,
        )

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )
        self.add_module('smpl', self.smpl)

        # render resolution can be set to a higher res than the input images to inspect
        # the results better
        render_resolution = self.hparams.DATASET.RENDER_RES if self.hparams.RUN_TEST \
            else self.hparams.DATASET.IMG_RES

        # renderer is used to visualize the gt and predicted HPS
        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=render_resolution,
            faces=self.smpl.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

        # Initialize the training datasets only in training mode
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()

        self.example_input_array = torch.rand(1, 3,
                                              self.hparams.DATASET.IMG_RES,
                                              self.hparams.DATASET.IMG_RES)

        self.val_accuracy_results = []

        # Initialiatize variables required for evaluation
        self.init_evaluation_variables()

    def init_evaluation_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_v2v = [] # vertex2vertex

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        # use this to save the errors for each image
        if self.hparams.TESTING.SAVE_IMAGES:
            self.val_images_errors = []

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam'] = []
            # self.evaluation_results['vertices'] = []
            self.evaluation_results['joints'] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        '''
        A single training step over a single batch
        '''

        # Get data from the batch
        images = batch['img']  # input image
        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        batch_size = images.shape[0]

        # Get GT vertices and model joints
        gt_out = self.smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_size=self.hparams.DATASET.IMG_RES,
            use_all_joints=True if '3dpw' in self.hparams.DATASET.DATASETS_AND_RATIOS else False,
        )

        pred = self(images)

        batch['gt_cam_t'] = gt_cam_t
        batch['vertices'] = gt_vertices

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        tensorboard_logs = loss_dict

        self.log_dict(tensorboard_logs)

        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0:
            self.train_summaries(input_batch=batch, output=pred)

        return {'loss': loss, 'log': tensorboard_logs}

    def train_summaries(self, input_batch, output):
        '''
        Saves the rendered images to inspect the training progress
        '''

        images = input_batch['img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()
        opt_vertices = input_batch['vertices']

        pred_cam_t = output['pred_cam_t'].detach()
        opt_cam_t = input_batch['gt_cam_t']

        vertex_colors = None

        images_pred = self.renderer.visualize_tb(
            pred_vertices,
            pred_cam_t,
            images,
            sideview=self.hparams.TESTING.SIDEVIEW,
            vertex_colors=vertex_colors,
        )

        images_opt = self.renderer.visualize_tb(
            opt_vertices,
            opt_cam_t,
            images,
            sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.hparams.TRAINING.SAVE_IMAGES:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

            images_opt = images_opt.cpu().numpy().transpose(1, 2, 0) * 255
            images_opt = np.clip(images_opt, 0, 255).astype(np.uint8)

            save_dir = os.path.join(self.hparams.LOG_DIR, 'training_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_dir, f'result_{self.global_step:08d}.jpg'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )
            cv2.imwrite(
                os.path.join(save_dir, f'gt_{self.global_step:08d}.jpg'),
                cv2.cvtColor(images_opt, cv2.COLOR_BGR2RGB)
            )

    def validation_step(self, batch, batch_nb, dataloader_nb=0, save=True, mesh_save_dir=None):
        '''
        A single validation step over a single batch
        '''

        images = batch['img']
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred = self(images)
            pred_vertices = pred['smpl_vertices']

        gt_keypoints_3d = batch['pose_3d'].cuda()

        pred_keypoints_3d = pred['smpl_joints3d']
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        # Procrustes-aligned error (PA-MPJPE)
        r_error, r_error_per_joint = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None,
        )

        # Per-vertex error
        gt_vertices = batch['vertices'].cuda()

        v2v = compute_error_verts(
            pred_verts=pred_vertices.cpu().numpy(),
            target_verts=gt_vertices.cpu().numpy(),
        )

        self.val_v2v += v2v.tolist()
        self.val_mpjpe += error.tolist()
        self.val_pampjpe += r_error.tolist()

        error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()

        self.evaluation_results['mpjpe'] += error_per_joint.tolist()
        self.evaluation_results['pampjpe'] += r_error_per_joint.tolist()

        self.evaluation_results['v2v'] += v2v.tolist()

        self.evaluation_results['imgname'] += imgnames
        self.evaluation_results['dataset_name'] += dataset_names

        if self.hparams.TESTING.SAVE_RESULTS:
            tolist = lambda x: [i for i in x.cpu().numpy()]
            self.evaluation_results['pose'] += tolist(pred['pred_pose'])
            self.evaluation_results['shape'] += tolist(pred['pred_shape'])
            self.evaluation_results['cam'] += tolist(pred['pred_cam'])
            # self.evaluation_results['vertices'] += tolist(pred_vertices)

        if self.hparams.TESTING.SAVE_IMAGES:
            # this saves the rendered images
            self.validation_summaries(batch, pred, batch_nb, dataloader_nb)

        return {
            'mpjpe': error.mean(),
            'pampjpe': r_error.mean(),
            'per_mpjpe': error_per_joint,
            'per_pampjpe': r_error_per_joint
        }

    def validation_summaries(
            self, input_batch, output, batch_idx, dataloader_nb,
            save=True, mesh_save_dir=None
    ):
        '''
        Saves the validation step visualizations
        '''

        # images = input_batch['img']
        images = input_batch['disp_img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()

        # convert camera parameters to display image params
        pred_cam = output['pred_cam'].detach()
        pred_cam_t = convert_weak_perspective_to_perspective(
            pred_cam,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.RENDER_RES,
        )

        vertex_colors = None

        mesh_filename = None
        if self.hparams.TESTING.SAVE_MESHES:
            save_dir = mesh_save_dir if mesh_save_dir else os.path.join(self.hparams.LOG_DIR, 'output_meshes')
            os.makedirs(save_dir, exist_ok=True)
            mesh_filename = os.path.join(save_dir, f'result_{dataloader_nb:02d}_{batch_idx:05d}.obj')

            images_mesh = images[0].clone().cpu().numpy().transpose(1, 2, 0) * 255
            images_mesh = np.clip(images_mesh, 0, 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(save_dir, f'result_{dataloader_nb:02d}_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_mesh, cv2.COLOR_BGR2RGB)
            )

        images_pred = self.renderer.visualize_tb(
            pred_vertices,
            pred_cam_t,
            images,
            nb_max_img=1,
            sideview=self.hparams.TESTING.SIDEVIEW,
            vertex_colors=vertex_colors,
            multi_sideview=self.hparams.TESTING.MULTI_SIDEVIEW,
            mesh_filename=mesh_filename,
        )

        images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
        images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

        if save:
            save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_dir, f'result_{dataloader_nb:02d}_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )
        return cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)

    def validation_epoch_end(self, outputs) -> None:
        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_v2v = np.array(self.val_v2v)

        for k, v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        avg_mpjpe, avg_pampjpe = 1000 * self.val_mpjpe.mean(), 1000 * self.val_pampjpe.mean()
        avg_v2v = 1000 * self.val_v2v.mean()

        logger.info(f'***** Epoch {self.current_epoch} *****')
        logger.info('MPJPE: ' + str(avg_mpjpe))
        logger.info('PA-MPJPE: ' + str(avg_pampjpe))
        logger.info('V2V (mm): ' + str(avg_v2v))

        acc = {
            'val_mpjpe': avg_mpjpe.item(),
            'val_pampjpe': avg_pampjpe.item(),
            'val_v2v': avg_v2v.item(),
        }

        self.val_save_best_results(acc)

        # save the mpjpe and pa-mpjpe results per image
        if self.hparams.TESTING.SAVE_IMAGES and len(self.val_images_errors) > 0:
            save_path = os.path.join(self.hparams.LOG_DIR, 'val_images_error.npy')
            logger.info(f'Saving the errors of images {save_path}')
            np.save(save_path, np.asarray(self.val_images_errors))

        joblib.dump(
            self.evaluation_results,
            os.path.join(self.hparams.LOG_DIR, f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl')
        )

        avg_mpjpe, avg_pampjpe = torch.tensor(avg_mpjpe), torch.tensor(avg_pampjpe)
        tensorboard_logs = {
            'val/val_mpjpe': avg_mpjpe,
            'val/val_pampjpe': avg_pampjpe,
        }
        val_log = {
            'val_loss': avg_pampjpe,
            'val_mpjpe': avg_mpjpe,
            'val_pampjpe': avg_pampjpe,
            'log': tensorboard_logs
        }

        for k, v in val_log.items():
            if k == 'log':
                pass
            else:
                self.log(k, v)

        # reset evaluation variables
        self.init_evaluation_variables()

    def test_step(self, batch, batch_nb, dataloader_nb=0) -> None:
        '''
        A single test step over a single batch
        '''

        images = batch['img']
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        with torch.no_grad():
            pred = self(images)
            # pred_vertices = pred['smpl_vertices']
            pred_joints = pred['smpl_joints3d']

            self.evaluation_results['imgname'] += imgnames
            self.evaluation_results['dataset_name'] += dataset_names

            tolist = lambda x: [i for i in x.cpu().numpy()]
            self.evaluation_results['pose'] += tolist(pred['pred_pose'])
            self.evaluation_results['shape'] += tolist(pred['pred_shape'])
            self.evaluation_results['cam'] += tolist(pred['pred_cam'])
            self.evaluation_results['joints'] += tolist(pred_joints)

    def test_epoch_end(self, outputs) -> None:
        '''
        Saves the submission file
        '''
        for k, v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        pred_joints = np.concatenate(self.evaluation_results['joints']
                                     ).reshape(self.evaluation_results['joints'].shape[0], -1)
        np.savetxt(os.path.join(self.hparams.LOG_DIR, f'test_pred_joints.csv'), pred_joints, delimiter=",")

        # reset evaluation variables
        self.init_evaluation_variables()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataset(self):
        train_ds = MixedDataset(
            options=self.hparams.DATASET,
            ignore_3d=self.hparams.DATASET.IGNORE_3D,
            is_train=True
        )

        return train_ds

    def train_dataloader(self):
        set_seed(self.hparams.SEED_VALUE)

        self.train_ds = self.train_dataset()

        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataset(self):
        return BaseDataset(
            options=self.hparams.DATASET,
            dataset=self.hparams.DATASET.VAL_DS,
            is_train=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
        )

    def test_dataloader(self):
        test_ds = BaseDataset(
            options=self.hparams.DATASET,
            dataset='3dpw',
            is_train=False,
        )
        return DataLoader(
            dataset=test_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
        )

    def val_save_best_results(self, acc, ds_name=None):
        # log the running training metrics
        if ds_name:
            fname = f'val_accuracy_results_{ds_name}.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results[ds_name].append([self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results[ds_name], f, indent=4)
        else:
            fname = 'val_accuracy_results.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results.append([self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results, f, indent=4)
