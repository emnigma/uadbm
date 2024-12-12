import argparse

# download_brainweb_dataset(
#     base_dir=Path('./content/data/Brainweb'),
#     name="",
#     institution="",
#     email=""
# )
import json
import os
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from imageio import imwrite

from dataloaders.BRAINWEB import BRAINWEB
from models import (
    anovaegan,
    autoencoder,
    constrained_adversarial_autoencoder,
    context_encoder_variational_autoencoder,
    context_encoder_variational_autoencoder_Zimmerer,
    fanogan,
    gaussian_mixture_variational_autoencoder,
    gaussian_mixture_variational_autoencoder_spatial,
    gaussian_mixture_variational_autoencoder_You,
    variational_autoencoder,
    variational_autoencoder_Zimmerer,
)
from trainers import Metrics
from trainers.AE import AE
from trainers.CE import CE
from trainers.ceVAE import ceVAE

# from trainers.fAnoGAN import fAnoGAN
from trainers.ConstrainedAAE import ConstrainedAAE
from trainers.ConstrainedAE import ConstrainedAE
from trainers.GMVAE import GMVAE
from trainers.GMVAE_spatial import GMVAE_spatial
from trainers.VAE import VAE
from trainers.VAE_You import VAE_You
from utils import Evaluation
from utils.brainweb_download import download_brainweb_dataset
from utils.default_config_setup import Dataset, get_config, get_datasets, get_options
from utils.Evaluation import (
    add_colorbar,
    apply_3d_median_filter,
    apply_brainmask,
    get_eval_dictionary,
    is_float,
    normalize_and_squeeze,
    should,
    squash_intensities,
)
from utils.utils import apply_colormap


def recon(args):
    base_path = os.path.dirname(os.path.abspath(__file__))

    base_path_trainer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "trainers", f"{args.trainer}.py"
    )
    trainer = getattr(
        SourceFileLoader(args.trainer, base_path_trainer).load_module(), args.trainer
    )

    with open(os.path.join(base_path, args.config), "r") as f:
        json_config = json.load(f)

    dataset = Dataset.BRAINWEB
    options = get_options(
        batchsize=args.batchsize,
        learningrate=args.lr,
        numEpochs=args.numEpochs,
        zDim=args.zDim,
        outputWidth=args.outputWidth,
        outputHeight=args.outputHeight,
        slices_start=args.slices_start,
        slices_end=args.slices_end,
        numMonteCarloSamples=args.numMonteCarloSamples,
        config=json_config,
    )
    options["data"]["dir"] = options["globals"][dataset.value]
    dataset_hc, dataset_pc = get_datasets(options, dataset=dataset)
    config = get_config(
        trainer=trainer,
        options=options,
        optimizer=args.optimizer,
        intermediateResolutions=args.intermediateResolutions,
        dropout_rate=0.2,
        dataset=dataset_hc,
    )

    modelObj = AE(tf.compat.v1.Session(), config, network=autoencoder.autoencoder)
    modelObj.load_checkpoint()
    patient = dataset_pc.patients[1]

    

    patient = {
        'name': os.path.basename(patient["name"]),
        'type': "NORMAL",
        'fullpath': patient["fullpath"]
    }
    patient['filtered_files'] = patient['fullpath']

    patient["groundtruth_filename"] = "content/data/Brainweb/groundtruth/normal.nii.gz"

    nii_filename = patient["filtered_files"]
    output_dir = Path(f"./{Path(nii_filename).name}")
    Path.mkdir(output_dir, exist_ok=True)

    recon_eval(nii_filename, patient, datasetObj = dataset_hc, options=options, modelObj=modelObj, output_dir = output_dir)

def recon_eval(nii_filename, patient, datasetObj, options, modelObj, output_dir):
    nii, nii_seg, nii_skullmap = datasetObj.load_volume_and_groundtruth(nii_filename, patient)
    p = 0
    sampleDir = output_dir
    prior_quantile = np.quantile(nii.data, 0.9)

    # Sanity checks - if coregistration went wrong and shapes are bad, we skip this sample
    if min(nii.shape()) < (datasetObj.options.sliceEnd - datasetObj.options.sliceStart):
        raise RuntimeError("Coregistration went wrong")

    # Iterate over all slices and collect them
    subvolume = np.zeros(
        [datasetObj.options.sliceEnd - datasetObj.options.sliceStart, options['train']['outputHeight'],
            options['train']['outputWidth']])
    subvolume_idx = 0
    slice_start = 0
    slice_end = nii.num_slices_along_axis(datasetObj.options.axis)
    zoom_factor = 1.0
    if datasetObj.options.sliceStart:
        slice_start = datasetObj.options.sliceStart
    if datasetObj.options.sliceEnd:
        slice_end = min(datasetObj.options.sliceEnd, nii.num_slices_along_axis(datasetObj.options.axis))

    _eval_dict = get_eval_dictionary()

    for s in range(slice_start, slice_end):
        slice_data = nii.get_slice(s, datasetObj.options.axis)
        slice_seg = nii_seg.get_slice(s, datasetObj.options.axis).astype(int)
        slice_skullmap = nii_skullmap.get_slice(s, datasetObj.options.axis).astype(int)

        if datasetObj.options.sliceResolution is not None:
            zoom_factor = tuple([i / j for (i, j) in zip(datasetObj.options.sliceResolution, slice_data.shape)])
            slice_data = scipy.ndimage.zoom(slice_data, zoom_factor)
            slice_seg = scipy.ndimage.zoom(slice_seg, zoom_factor, mode="nearest")
            slice_skullmap = scipy.ndimage.zoom(slice_skullmap, zoom_factor, mode="nearest")

        x = np.expand_dims(slice_data, 2)
        labelmaps = np.expand_dims(slice_seg, 2)
        _tmp = time.time()

        # Monte Carlo Uncertainty Estimation
        num_samples = 1
        if should(options, "numMonteCarloSamples"):
            num_samples = options["numMonteCarloSamples"]
        x_recs = []
        x_diffs = []
        x_log_vars = []
        results = None
        for i in range(num_samples):
            if num_samples > 1:
                results = modelObj.reconstruct(x, dropout=True)
            else:
                results = modelObj.reconstruct(x)
            x_rec_tmp = results['reconstruction']
            if "log_var" in results:
                x_log_vars += [results["log_var"]]

            x_recs += [np.reshape(Evaluation.apply_brainmask(x_rec_tmp, slice_skullmap, erode=should(options, "erodeBrainmask")),
                                    [1, *datasetObj.options.sliceResolution, 1])]
            x_diffs += [
                np.reshape(apply_brainmask(np.maximum(x - x_rec_tmp, 0), slice_skullmap, erode=should(options, "erodeBrainmask")),
                            [1, *datasetObj.options.sliceResolution, 1])]
        x_recs = np.array(x_recs)
        x_diffs = np.array(x_diffs)
        x_log_vars = np.array(x_log_vars)
        if x_log_vars.size == 0:
            x_log_vars = np.zeros(x_diffs.shape)
        x_recs_var = Metrics.combined_predictive_uncertainty(x_recs, x_log_vars, axis=0, log_var=False)
        x_recs_var_epistemic = Metrics.combined_predictive_uncertainty(x_recs, np.zeros(x_recs.shape), axis=0, log_var=False)
        x_recs_mean = np.mean(x_recs, axis=0)

        x_recs_var = apply_brainmask(x_recs_var, slice_skullmap, erode=should(options, "erodeBrainmask"))

        x_recs_var_epistemic * (2 * np.expand_dims(np.expand_dims(slice_skullmap, axis=0), axis=-1) - 1)
        # values outside the brain are getting negative, while values on the brain stay the same

        _eval_dict['reconstructionTimes'] += [time.time() - _tmp]

        # Get a sample without dropout
        x_rec = results['reconstruction']
        l1err = results['l1err']
        l2err = results['l2err']
        if num_samples > 1:
            x_rec = x_recs_mean
        if should(options, "keepOnlyPositiveResiduals"):
            x_diff = np.maximum(x - x_rec, 0)
        else:
            x_diff = np.abs(x - x_rec)
        x_diff = np.reshape(apply_brainmask(x_diff, slice_skullmap, erode=should(options, "erodeBrainmask")),
                            [1, *datasetObj.options.sliceResolution, 1])
        if should(options, "applyHyperIntensityPrior"):
            x_diff[np.reshape(x, [1, *datasetObj.options.sliceResolution, 1]) < prior_quantile] = 0

        subvolume[subvolume_idx, :, :] = np.squeeze(x_diff)
        subvolume_idx += 1

        # Fill eval array
        _eval_dict['x'] += [x]
        if num_samples > 1:
            _eval_dict['epistemic_variance'] += [x_recs_var_epistemic]
        _eval_dict['reconstructions'] += [x_rec]
        _eval_dict['labelmaps'] += [np.squeeze(labelmaps)]
        _eval_dict['l1reconstructionErrors'] += [l1err]
        _eval_dict['l2reconstructionErrors'] += [l2err]
        imwrite(os.path.join(sampleDir, '{}_{}.png'.format(p, s)), normalize_and_squeeze(x))
        imwrite(os.path.join(sampleDir, '{}_{}_rec.png'.format(p, s)), normalize_and_squeeze(x_rec))
        imwrite(os.path.join(sampleDir, '{}_{}_gt.png'.format(p, s)), normalize_and_squeeze(labelmaps))  # check if normalization is useful
        imwrite(os.path.join(sampleDir, '{}_{}_diff.png'.format(p, s)), normalize_and_squeeze(x_diff))
        imwrite(os.path.join(sampleDir, '{}_{}_rec_variance_combined.png'.format(p, s)),
                np.squeeze(apply_colormap(x_recs_var, plt.cm.jet)))
        if x_log_vars.size > 0:
            imwrite(os.path.join(sampleDir, '{}_{}_logvar.png'.format(p, s)), normalize_and_squeeze(np.mean(x_log_vars, axis=0)))

    if should(options, "medianFiltering"):
        subvolume = apply_3d_median_filter(subvolume)

    _eval_dict['diffs'] += [subvolume]

    for s in range(datasetObj.options.sliceStart, min(datasetObj.options.sliceEnd, nii.num_slices_along_axis(datasetObj.options.axis))):
        imwrite(os.path.join(sampleDir, '{}_{}_diff_filtered.png'.format(p, s)),
                normalize_and_squeeze(subvolume[s - datasetObj.options.sliceStart]))
        squashed = squash_intensities(np.squeeze(subvolume[s - datasetObj.options.sliceStart]))
        squashed = add_colorbar(squashed)
        imwrite(os.path.join(sampleDir, '{}_{}_heatmap.png'.format(p, s)), np.squeeze(apply_colormap(squashed, plt.cm.jet)))

    if should(options, "exportVolumes"):
        dezoom_factor = tuple([1]) + tuple(1 / np.asarray(zoom_factor))
        subvolume_deprocessed = scipy.ndimage.interpolation.zoom(subvolume, dezoom_factor)
        nii_seg.set_to_zero()
        nii_seg.cast_to_float()
        nii_seg.set_subvolume(datasetObj.options.sliceStart, datasetObj.options.sliceEnd, subvolume_deprocessed,
                                axis=datasetObj.options.axis)
        nii_seg.save(os.path.join(sampleDir, '{}.nii.gz'.format(patient['name'])))
        if options['threshold'] and is_float(options['threshold']):
            nii_seg.data = np.asarray((nii_seg.data > options['threshold'])).astype(np.float32)
            nii_seg.update_sitk()
            nii_seg.save(os.path.join(sampleDir, '{}.binary.nii.gz'.format(patient['name'])))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Framework')
    args.add_argument("--nii", type=Path, help="Patient path", dest="nii_filename", default=Path("content/data/Brainweb/normal/t2_icbm_normal_1mm_pn0_rf0.nii.gz"))
    args.add_argument('-c', '--config', default='config.json', type=str, help='config-path')
    args.add_argument('-b', '--batchsize', default=8, type=int, help='batchsize')
    args.add_argument('-l', '--lr', default=0.0001, type=float, help='learning rate')
    args.add_argument('-E', '--numEpochs', default=1000, type=int, help='how many epochs to train')
    args.add_argument('-z', '--zDim', default=128, type=int, help='Latent dimension')
    args.add_argument('-w', '--outputWidth', default=128, type=int, help='Output width')
    args.add_argument('-g', '--outputHeight', default=128, type=int, help='Output height')
    args.add_argument('-o', '--optimizer', default='ADAM', type=str, help='Can be either ADAM, SGD or RMSProp')
    args.add_argument('-i', '--intermediateResolutions', default=(8, 8), type=tuple[int], help='Spatial Bottleneck resolution')
    args.add_argument('-s', '--slices_start', default=20, type=int, help='slices start')
    args.add_argument('-e', '--slices_end', default=130, type=int, help='slices end')
    args.add_argument('-t', '--trainer', default='AE', type=str, help='Can be every class from trainers directory')
    args.add_argument('-m', '--model', default='autoencoder', type=str, help='Can be every class from models directory')
    args.add_argument('-O', '--threshold', default=None, type=float, help='Use predefined ThreshOld')
    args.add_argument('-d', '--ds', default=None, type=Dataset, help='Only evaluate on given dataset')

    # following arguments are only relevant for specific architectures
    args.add_argument('-n', '--numMonteCarloSamples', default=0, type=int, help='Amount of Monte Carlos Samples during restoration')
    args.add_argument('-G', '--use_gradient_based_restoration', default=False, type=bool, help='only for ceVAE')
    args.add_argument('-K', '--kappa', default=1.0, type=float, help='only for GANs')
    args.add_argument('-M', '--scale', default=10.0, type=float, help='only for GANs')
    args.add_argument('-R', '--rho', default=1.0, type=float, help='only for ConstrainedAAE')
    args.add_argument('-C', '--dim_c', default=9, type=int, help='only for GMVAE')
    args.add_argument('-Z', '--dim_z', default=128, type=int, help='only for GMVAE')
    args.add_argument('-W', '--dim_w', default=1, type=int, help='only for GMVAE')
    args.add_argument('-A', '--c_lambda', default=1, type=int, help='only for GMVAE')
    args.add_argument('-L', '--restore_lr', default=1e-3, type=float, help='only for GMVAE')
    args.add_argument('-S', '--restore_steps', default=150, type=int, help='only for GMVAE')
    args.add_argument('-T', '--tv_lambda', default=-1.0, type=float, help='only for GMVAE')

    recon(args.parse_args())
