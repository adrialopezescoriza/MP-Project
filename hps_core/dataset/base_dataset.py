import time
import torch
import numpy as np
from os.path import join
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..models import SMPL
from ..core import constants, config
from ..core.config import DATASET_FILES, DATASET_FOLDERS, EVAL_MESH_DATASETS
from ..utils.image_utils import crop, transform, rot_aa, read_img
from ..utils import kp_utils


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in hps_core/core/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False,
                 use_augmentation=True, is_train=True, num_images=0):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        self.dataset_dict = {dataset: 0}

        if num_images > 0:
            # set num_images > 0 if you don't want to use all images in the dataset
            if is_train:
                # select a random subset of the dataset
                rand = np.random.randint(0, len(self.imgname), size=(num_images))
                logger.info(f'{rand.shape[0]} images are randomly sampled from {self.dataset}')
                self.imgname = self.imgname[rand]
                self.data_subset = {}
                for f in self.data.files:
                    self.data_subset[f] = self.data[f][rand]
                self.data = self.data_subset
            else:
                interval = len(self.imgname) // num_images
                logger.info(f'{len(self.imgname[::interval])} images are selected from {self.dataset}')
                self.imgname = self.imgname[::interval]
                self.data_subset = {}
                for f in self.data.files:
                    self.data_subset[f] = self.data[f][::interval]
                self.data = self.data_subset

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale'] # scale = bbox_height/200.
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        self.occluders = None

        # evaluation variables
        if not self.is_train or (self.is_train and dataset == '3dpw'):

            self.smpl = SMPL(
                config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False
            )

            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False)

        self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def rgb_processing(self, rgb_img, center, scale, rot, pn, img_res):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [img_res, img_res], rot=rot)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2] + 1, center, scale,
                                  [self.options.IMG_RES, self.options.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2. * kp[:,:-1] / self.options.IMG_RES - 1.

        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 

        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)

        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()

        # We disabled the data augmentation here, but feel free to use it
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling

        load_start = time.perf_counter()
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except Exception as e:
            logger.info(e)
            logger.info(imgname)

        orig_shape = np.array(cv_img.shape)[:2]
        load_time = time.perf_counter() - load_start
        # print(f'loading: {time.perf_counter() - loading_start}s.')

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale, rot)

        proc_start = time.perf_counter()

        # Process image
        img = self.rgb_processing(cv_img, center, sc*scale, rot, pn,
                                  img_res=self.options.IMG_RES)

        img = torch.from_numpy(img).float()
        proc_time = time.perf_counter() - proc_start

        if not self.is_train and not self.options.RENDER_RES == self.options.IMG_RES:
            disp_img = self.rgb_processing(cv_img, center, sc * scale, rot, pn,
                                           img_res=self.options.RENDER_RES)
            disp_img = torch.from_numpy(disp_img).float()
            item['disp_img'] = self.normalize_img(disp_img)

        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # prepare pose_3d for evaluation
        # For 3DPW get the 14 common joints from the rendered shape
        if self.is_train and self.dataset == '3dpw':
            if self.gender[index] == 1:
                j3d = self.smpl_female(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),
                ).joints[0, 25:].cpu().numpy()
            else:
                j3d = self.smpl_male(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),
                ).joints[0, 25:].cpu().numpy()
            j3d -= np.array(j3d[2] + j3d[3]) / 2. # root center
            S = np.hstack([j3d, np.ones([24, 1])])
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot)).float()

        if not self.is_train:
            if self.dataset in EVAL_MESH_DATASETS:

                if self.gender[index] == 1:
                     smpl_out = self.smpl_female(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    )

                else:
                    smpl_out = self.smpl_male(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    )

                gt_vertices = smpl_out.vertices
                gt_joints_3d = smpl_out.joints

                pelvis = gt_joints_3d[:, [0], :].clone()
                gt_joints_3d = gt_joints_3d - pelvis
                item['pose_3d'] = gt_joints_3d[0].float()
                item['vertices'] = gt_vertices[0].float()
            else:
                item['pose_3d'] = item['pose_3d'][self.joint_mapper_gt, :-1].float()

        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        item['load_time'] = load_time
        item['proc_time'] = proc_time
        return item

    def __len__(self):
        return len(self.imgname)
