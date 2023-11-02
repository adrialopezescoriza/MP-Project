from os.path import join
from yacs.config import CfgNode as CN

##### CONSTANTS #####
DATASET_NPZ_PATH = 'data/dataset_extras'


MPII_ROOT = 'data/dataset_folders/mpii'
PW3D_ROOT = 'data/dataset_folders/3dpw'

JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/body_models/smpl'

DATASET_FOLDERS = {
    'mpii': MPII_ROOT,
    '3dpw': PW3D_ROOT,
    '3dpw-val': PW3D_ROOT,
    '3dpw-all': PW3D_ROOT,
}

DATASET_FILES = [
    # Testing
    {
        '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_subset_cleaned.npz'),
        '3dpw-val': join(DATASET_NPZ_PATH, '3dpw_validation_subset.npz'),
    },
    # Training
    {
        'mpii': join(DATASET_NPZ_PATH, 'mpii_train_eft.npz'),
        '3dpw': join(DATASET_NPZ_PATH, '3dpw_train_subset.npz'),
    }
]

EVAL_MESH_DATASETS = ['3dpw', '3dpw-val']

hparams = CN()

# General settings
hparams.LOG_DIR = 'logs/experiments'
hparams.METHOD = 'pare'
hparams.EXP_NAME = 'default'
hparams.RUN_TEST = False
hparams.PROJECT_NAME = 'pare'
hparams.SEED_VALUE = -1

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.LOAD_TYPE = 'Base'
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.TRAIN_DS = 'all'
hparams.DATASET.VAL_DS = '3dpw_3doh'
hparams.DATASET.NUM_IMAGES = -1
hparams.DATASET.TRAIN_NUM_IMAGES = -1
hparams.DATASET.TEST_NUM_IMAGES = -1
hparams.DATASET.IMG_RES = 224
hparams.DATASET.RENDER_RES = 480
hparams.DATASET.MESH_COLOR = 'pinkish'
hparams.DATASET.FOCAL_LENGTH = 5000.
hparams.DATASET.IGNORE_3D = False
hparams.DATASET.DATASETS_AND_RATIOS = 'mpii_3dpw_0.5_0.5'

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 0.0001 # 0.00003
hparams.OPTIMIZER.WD = 0.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 50
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.TEST_BEFORE_TRAINING = False
hparams.TRAINING.SAVE_IMAGES = False
hparams.TRAINING.USE_AMP = False

# Training process hparams
hparams.TESTING = CN()
hparams.TESTING.SAVE_IMAGES = True
hparams.TESTING.SAVE_FREQ = 1
hparams.TESTING.SAVE_RESULTS = True
hparams.TESTING.SAVE_MESHES = True
hparams.TESTING.SIDEVIEW = True
hparams.TESTING.TEST_ON_TRAIN_END = True
hparams.TESTING.MULTI_SIDEVIEW = False
hparams.TESTING.USE_GT_CAM = False

hparams.HMR = CN()
hparams.HMR.SHAPE_LOSS_WEIGHT = 0
hparams.HMR.KEYPOINT_LOSS_WEIGHT = 5.
hparams.HMR.KEYPOINT_NATIVE_LOSS_WEIGHT = 5.
hparams.HMR.SMPL_PART_LOSS_WEIGHT = 1.
hparams.HMR.POSE_LOSS_WEIGHT = 1.
hparams.HMR.BETA_LOSS_WEIGHT = 0.001
hparams.HMR.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.HMR.GT_TRAIN_WEIGHT = 1.
hparams.HMR.LOSS_WEIGHT = 60.


def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()
