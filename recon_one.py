import argparse
from importlib.machinery import SourceFileLoader
from utils.brainweb_download import download_brainweb_dataset
from pathlib import Path

# download_brainweb_dataset(
#     base_dir=Path('./content/data/Brainweb'),
#     name="",
#     institution="",
#     email=""
# )

import json
import os
import tensorflow as tf
from datetime import datetime
from utils.default_config_setup import get_config, get_options, get_datasets
from trainers.AE import AE
from trainers.VAE import VAE
from trainers.CE import CE
from trainers.ceVAE import ceVAE
from trainers.VAE_You import VAE_You
from trainers.GMVAE import GMVAE
from trainers.GMVAE_spatial import GMVAE_spatial

# from trainers.fAnoGAN import fAnoGAN
from trainers.ConstrainedAAE import ConstrainedAAE
from trainers.ConstrainedAE import ConstrainedAE

from models import (
    autoencoder,
    variational_autoencoder,
    context_encoder_variational_autoencoder,
    variational_autoencoder_Zimmerer,
    context_encoder_variational_autoencoder_Zimmerer,
    gaussian_mixture_variational_autoencoder_You,
    gaussian_mixture_variational_autoencoder_spatial,
    gaussian_mixture_variational_autoencoder,
    fanogan,
    constrained_adversarial_autoencoder,
    anovaegan,
)
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset


import numpy as np
import scipy
import cv2
from imageio import imwrite


def getconf():
    return {
        "config": "config.json",
        "batchsize": 8,
        "lr": 0.0001,
        "numEpochs": 1000,
        "zDim": 128,
        "outputWidth": 128,
        "outputHeight": 128,
        "optimizer": "ADAM",
        "intermediateResolutions": (8, 8),
        # "slices_start": 20,
        "slices_start": 80,
        # "slices_end": 130,
        "slices_end": 90,
        "trainer": "AE",
        "model": "autoencoder",
        "threshold": None,
        "ds": None,
        "numMonteCarloSamples": 0,
        "use_gradient_based_restoration": False,
        "kappa": 1.0,
        "scale": 10.0,
        "rho": 1.0,
        "dim_c": 9,
        "dim_z": 128,
        "dim_w": 1,
        "c_lambda": 1,
        "restore_lr": 0.001,
        "restore_steps": 150,
        "tv_lambda": -1.0,
    }

args = getconf()

args = argparse.Namespace(**getconf())

base_path = os.path.dirname(os.path.abspath(__file__))

base_path_trainer = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainers', f'{args.trainer}.py')
base_path_network = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'{args.model}.py')
trainer = getattr(SourceFileLoader(args.trainer, base_path_trainer).load_module(), args.trainer)
network = getattr(SourceFileLoader(args.model, base_path_network).load_module(), args.model)

with open(os.path.join(base_path, args.config), 'r') as f:
    json_config = json.load(f)

dataset = Dataset.BRAINWEB
options = get_options(batchsize=args.batchsize, learningrate=args.lr, numEpochs=args.numEpochs, zDim=args.zDim, outputWidth=args.outputWidth,
                        outputHeight=args.outputHeight, slices_start=args.slices_start, slices_end=args.slices_end,
                        numMonteCarloSamples=args.numMonteCarloSamples, config=json_config)
options['data']['dir'] = options["globals"][dataset.value]
dataset_hc, dataset_pc = get_datasets(options, dataset=dataset)
config = get_config(
    trainer=trainer,
    options=options,
    optimizer=args.optimizer,
    intermediateResolutions=args.intermediateResolutions,
    dropout_rate=0.2,
    dataset=dataset_hc
)

def normalize_and_squeeze(x):
    return np.squeeze(cv2.normalize(x, None, 0, 255, norm_type=cv2.NORM_MINMAX)).astype('uint8')

modelObj = AE(tf.compat.v1.Session(), config, network=autoencoder.autoencoder)
modelObj.load_checkpoint()

output_dir = Path("./manual_run")

patient = dataset_hc.patients[1]

nii_filename = patient['filtered_files']
nii_name = Path(nii_filename).name

nii, nii_seg, nii_skullmap = dataset_hc.load_volume_and_groundtruth(nii_filename, patient)

if dataset_hc.options.sliceStart:
    slice_start = dataset_hc.options.sliceStart
if dataset_hc.options.sliceEnd:
    slice_end = min(dataset_hc.options.sliceEnd, nii.num_slices_along_axis(dataset_hc.options.axis))
for s in range(slice_start, slice_end):
    slice_data = nii.get_slice(s, dataset_hc.options.axis)
    slice_seg = nii_seg.get_slice(s, dataset_hc.options.axis).astype(int)
    slice_skullmap = nii_skullmap.get_slice(s, dataset_hc.options.axis).astype(int)

    if dataset_hc.options.sliceResolution is not None:
        zoom_factor = tuple([i / j for (i, j) in zip(dataset_hc.options.sliceResolution, slice_data.shape)])
        slice_data = scipy.ndimage.zoom(slice_data, zoom_factor)
        slice_seg = scipy.ndimage.zoom(slice_seg, zoom_factor, mode="nearest")
        slice_skullmap = scipy.ndimage.zoom(slice_skullmap, zoom_factor, mode="nearest")

    x = np.expand_dims(slice_data, 2)
    labelmaps = np.expand_dims(slice_seg, 2)
    results = modelObj.reconstruct(x)
    x_rec = results['reconstruction']

    op = Path(output_dir) / nii_name
    os.makedirs(op, exist_ok=True)

    imwrite(os.path.join(op, '{}.png'.format(s)), normalize_and_squeeze(x_rec))

