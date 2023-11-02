import torch
import sys

sys.path.append('.')
from hps_core.models.dummy_model import DummyModel
from hps_core.models.pare import PARE


def test():
    batch_size = 1
    img_res = 224

    model = PARE(img_res=img_res)

    dummy_inp = torch.rand(batch_size, 3, img_res, img_res)
    output = model(dummy_inp)

    desired_sizes = {
        "smpl_vertices": torch.Size([batch_size, 6890, 3]),
        "smpl_joints3d": torch.Size([batch_size, 49, 3]),
        "smpl_joints2d": torch.Size([batch_size, 49, 2]),
        "pred_cam_t": torch.Size([batch_size, 3]),
        "pred_pose": torch.Size([batch_size, 24, 3, 3]),
        "pred_cam": torch.Size([batch_size, 3]),
        "pred_shape": torch.Size([batch_size, 10]),
        "pred_pose_6d": torch.Size([batch_size, 144]),
    }

    for k, v in output.items():
        print(k, v.shape)
        assert v.shape == desired_sizes[k], f"Size of output \"{k}\" is not correct!"



if __name__ == '__main__':
    test()