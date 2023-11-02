import torch
import joblib
import numpy as np
from smplx import SMPL
import sys

np2th = lambda x: torch.from_numpy(x).float()


def eval_old():
    gt_data = np.load('data_old/dataset_extras/3dpw_test_subset.npz')
    pred_data = joblib.load(sys.argv[1])

    smpl = SMPL(
        'data/body_models/smpl/',
        create_transl=False,
    )

    dataset_len = gt_data['imgname'].shape[0]

    # for k, v in pred_data.items():
    #     print(k, v.shape)
    #
    # for f in gt_data.files:
    #     print(f, gt_data[f].shape)

    error_list = []
    for indices in np.array_split(np.arange(0, dataset_len), 64):
        # gt_imgnames = gt_data['imgname'][indices]
        # pred_imgnames = pred_data['imgname'][indices]

        gt_pose = np2th(gt_data['pose'][indices])
        gt_shape = np2th(gt_data['shape'][indices])

        pred_pose = np2th(pred_data['pose'][indices])
        pred_shape = np2th(pred_data['shape'][indices])

        # import ipdb; ipdb.set_trace()
        gt_smpl_out = smpl(
            global_orient=gt_pose[:, :3],
            body_pose=gt_pose[:, 3:],
            betas=gt_shape,

        )
        pred_smpl_out = smpl(
            global_orient=pred_pose[:, 0:1],
            body_pose=pred_pose[:, 1:],
            betas=pred_shape,
            pose2rot=False,
        )

        error = torch.sqrt(((gt_smpl_out.joints - pred_smpl_out.joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        error_list += error.tolist()

    errors = np.array(error_list)
    # mean_error = errors.mean() * 1000
    print(f'MPJPE: {errors.mean()*1000} mm')

    # print(f'MPJPE: {mean_error} mm')
    # import ipdb; ipdb.set_trace()


def eval():
    pred_joints = np.loadtxt(sys.argv[1], delimiter=",")
    gt_joints = np.loadtxt("private/3dpw_test_joints3d.csv", delimiter=",")
    pred_joints = pred_joints.reshape((pred_joints.shape[0], -1, 3))[:, 25:].reshape(pred_joints.shape[0], -1)
    gt_joints = gt_joints.reshape((gt_joints.shape[0], -1, 3))[:, 25:].reshape(gt_joints.shape[0], -1)


    error = np.sqrt(((gt_joints - pred_joints) ** 2)).mean(axis=-1)
    print(f'MPJPE: {error.mean()*1000} mm')


def save_joints3d():
    from hps_core.models.head.smpl_head import SMPL

    gt_data = np.load('data_old/dataset_extras/3dpw_test_subset.npz')

    smpl = SMPL(
        'data/body_models/smpl/',
        create_transl=False,
    )

    dataset_len = gt_data['imgname'].shape[0]

    # for k, v in pred_data.items():
    #     print(k, v.shape)
    #
    # for f in gt_data.files:
    #     print(f, gt_data[f].shape)

    joints3d_list = []
    for indices in np.array_split(np.arange(0, dataset_len), 64):
        gt_pose = np2th(gt_data['pose'][indices])
        gt_shape = np2th(gt_data['shape'][indices])

        # import ipdb; ipdb.set_trace()
        gt_smpl_out = smpl(
            global_orient=gt_pose[:, :3],
            body_pose=gt_pose[:, 3:],
            betas=gt_shape,
        )

        gt_joints = gt_smpl_out.joints.detach().cpu().numpy()
        joints3d_list.append(gt_joints)

    # import ipdb; ipdb.set_trace();
    gt_joints = np.concatenate(joints3d_list).reshape(dataset_len, -1)
    np.savetxt("3dpw_test_joints3d.csv", gt_joints, delimiter=",")


if __name__ == '__main__':
    # eval_old()
    eval()