import os
import sys
import torch
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
sys.path.append('.')

from hps_core.core.trainer import HPSTrainer
from hps_core.core.config import update_hparams
from hps_core.utils.train_utils import load_pretrained_model, set_seed, add_init_smpl_params_to_dict


def main(hparams, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(torch.cuda.get_device_properties(device))

    logger.info(f'Hyperparameters: \n {hparams}')

    experiment_loggers = []

    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tb_logs',
        log_graph=False,
    )

    # This is where we initialize the model specific training routines
    # check HPSTrainer to see training, validation, testing loops
    model = HPSTrainer(hparams=hparams).to(device)

    # initialize wandb logger (parameters can be added inside the config file)
    wandb_logger = WandbLogger(
        name=hparams.EXP_NAME, project=hparams.PROJECT_NAME, config=hparams, log_model=False)
    
    # log gradients and model topology
    wandb_logger.watch(model)

    experiment_loggers.append(tb_logger)
    experiment_loggers.append(wandb_logger)

    # TRAINING.PRETRAINED_LIT points to the checkpoint files trained using this repo
    # This has a separate cfg value since in some cases we use checkpoint files from different repos
    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(
            f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(
            model, ckpt, overwrite_shape_mismatch=True, strict=True)

    # When training gets interrupted you can resume it from a checkpoint by setting
    # the hparams.TRAINING.RESUME param
    if hparams.TRAINING.RESUME is not None:
        resume_ckpt = torch.load(hparams.TRAINING.RESUME)
        if not 'model.head.init_pose' in resume_ckpt['state_dict'].keys():
            logger.info(
                'Adding init SMPL parameters to the resume checkpoint...')
            resume_ckpt = torch.load(hparams.TRAINING.RESUME)
            resume_ckpt['state_dict'] = add_init_smpl_params_to_dict(
                resume_ckpt['state_dict'])
            torch.save(resume_ckpt, hparams.TRAINING.RESUME)

    # this callback saves best 10 checkpoint based on the validation loss
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=3,  # reduce this if you don't have enough storage
        mode='min',
        period=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
    )

    # Optionally you can use Automatic Mixed Precision to reduce the training time
    # and gpu memory usage which helps with increasing the batch size or image size
    amp_params = {}
    if hparams.TRAINING.USE_AMP:
        logger.info(
            f'Using automatic mixed precision: ampl_level 02, precision 16...')
        amp_params = {
            'amp_level': 'O2',
            # 'amp_backend': 'apex',
            'precision': 16,
        }

    trainer = pl.Trainer(
        gpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,  # total number of epochs
        callbacks=[ckpt_callback],
        log_every_n_steps=50,
        terminate_on_nan=True,
        default_root_dir=log_dir,
        progress_bar_refresh_rate=50,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        reload_dataloaders_every_epoch=hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        **amp_params,
    )

    if hparams.RUN_TEST:
        logger.info('*** Started testing ***')
        trainer.test(model)
    else:
        logger.info('*** Started training ***')
        trainer.fit(model)
        logger.info('*** Started testing ***')
        trainer.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/pare.yaml',
                        help='path to the configuration yaml file')
    parser.add_argument('--fdr', action='store_true',
                        help='fast_dev_run mode: will run a full train, val and test loop using 1 batch(es)')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = update_hparams(args.cfg)
    main(hparams, fast_dev_run=args.fdr)
