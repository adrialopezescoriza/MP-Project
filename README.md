# Machine Perception - SS2023
# Project 1: 3D Human Pose and Shape Estimation from RGB Images
The goal of this project is to gain experience on implementing a human pose and shape (HPS in short) estimation model
from RGB images. This is still a challenging problem due to complex human poses and shapes, lack of 3D labels etc. We
expect you to come up with clever and innovative ways to improve the existing methods.

![alt text](sample.png) <br />
*Sample training output accessible through the logs folder of a training run.*

## How to run our code
To run our code you can stick to the procedure described below. That means, once having installed the requirements and copied the data,  for training you can simply execute: 

```shell
python scripts/main.py --cfg configs/baseline.yaml
```

For evaluation run: 

```shell
python scripts/main.py --cfg configs/baseline_test.yaml
```
where hyperparameters are set in the respective config files.

## Getting Started
This code has been implemented and tested with python >= 3.7.

Copy the project code via

```shell
cp -r /cluster/project/infk/hilliges/lectures/mp23/project1/mp2023_hps_project $YOUR_WORKSPACE
```

We will refer to the project folder as `$PROJECT_ROOT` in this README. When you execute the scripts in this README,
you should be inside the `$PROJECT_ROOT`. Run `cd $PROJECT_ROOT` to be able to do it.

We suggest to use conda as the package manager. You can install it via following the instructions on their 
[website](https://docs.conda.io/en/latest/miniconda.html).

Here we provide two ways to install the requirements both using pip-virtualenv or conda:

```shell
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

## Data

We have provided all the required data (datasets, body models etc.) in `$PROJECT_ROOT/data`

Here is a short description of what each subfolder contains:

- `data/body_models` folder includes the SMPL body model files under . We have provided the necessary functions
on how to use SMPL body model in your model in the `hps_core/models/dummy_model.py`. You can use this model as a 
starting point when you are implementing your own HPS model.
- `data/dataset_folders` includes the RGB images from [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW) 
and [MPII](http://human-pose.mpi-inf.mpg.de/) datasets. 3DPW is a dataset with in-the-wild accurate 3D poses. Here we
will use this dataset both for training and testing. MPII is a 2D human keypoint dataset. 
Since obtaining 3D pose labels is
quite costly and limited to a few subjects, HPS estimation methods use a mixture of 2D and 3D pose datasets. Similarly,
we use MPII dataset for training only. A sample dataloader to show how to use mixed data is provided in 
`hps_core/dataset/mixed_dataset.py`
- `data/dataset_extras` includes the preprocessed annotations for 3DPW and MPII datasets. The annotations are stored
as [npz](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) files. Each files store the labels required
for training or testing on the respective dataset. A sample dataloader is provided in `hps_core/dataset/base_dataset.py`.
This dataloader demonstrates all the details about loading and prepocessing the images and labels.
When you load the annotations, you will encounter the fields below 
(N means the number of samples):
```
imgname : image file names of shape (N,)
center  : center of the person instance in pixels of shape (N, 2)
scale   : scale (height/200.) of the person instances of shape (N,)
pose    : SMPL pose parameters of shape (N, 72)
shape   : SMPL shape (betas) parameters of shape (N, 10)
gender  : Gender annotations for the subjects of shape (N,) - only in 3DPW
part    : 2D keypoint labels and their confidences of shape (14667, 24, 3) - only in MPII
S       : 3D keypoint labels and their confidences of shape (14667, 24, 4) - only in MPII
```


## Training
```shell
python scripts/main.py --cfg configs/baseline.yaml
```

This command starts training a dummy model using the hyperparameters defined in `configs/baseline.yaml` YAML file. 
You can find all the configurable hyperparameters in `hps_core/core/config.py`. Feel free to add more parameters if you
feel the need to. If you want to perform a quick sanity check before starting full training, you can run:

```shell
python scripts/main.py --cfg configs/baseline.yaml --fdr
```

Here `--fdr` means fast dev run. This will run one training and validation loop over a single batch to ensure everything
is correct. This is a suggested way to test after you modify the model, dataset, or some training routines.

During training everything will be saved under `logs` folder. Log files include tensorboard files, checkpoints,
qualitative results of your model predictions. You can inspect these logs to ensure your training runs or converges 
as expected.

Training, validation, and testing loops are implemented in `hps_core/core/trainer.py` using 
[pytorch-lightning](https://www.pytorchlightning.ai/) package.

A submission file called `test_pred_joints.csv` will be created at the end of the training. You should submit this file
to our evaluation server to check your public score.

## Evaluation

Even though a submission file is saved after the training run, you can also create other submission files using different
checkpoints from your runs. For that you need to update the `TRAINING.PRETRAINED_LIT` field of the 
`configs/baseline_test.yaml` file with the checkpoint you want to evaluate and run:

```shell
python scripts/main.py --cfg configs/baseline_test.yaml
```

## Evaluation Metric

We use Mean Per Joint Position Error (MPJPE) between predicted and ground-truth joints. 
This is a fairly standard evaluation metric used in HPS research. 
It is simply the euclidean distance between predicted and ground-truth joints averaged
across all testing samples.
