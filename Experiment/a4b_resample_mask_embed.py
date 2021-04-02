# coding: utf-8

# ===================================================================
# Mask fMRIs
# Romuald Menuet - June 2018
# ===================================================================
# Summary: This script masks fMRIs whose path is stored as a column
#          of a dataframe to a common affine
# ===================================================================

# 3rd party modules
import argparse
import os
import pickle
import json

import pandas as pd
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing


# ============================
# === RESAMPLING FUNCTIONS ===
# ============================
def resample_mini_batch(fmris, ref_map, cache_folder,
                        overwrite=False,
                        file_field="absolute_path",
                        suffix="_resamp",
                        interpolation="linear",
                        verbose=False):
    """
    Batch resampling of fMRIs whose filenames are stored in a Pandas DataFrame
    column, and that are stored as gzip ('*.gz') compressed Nifti  files
    (useful to resample files downloaded from Neurovault).

    :param fmris: pandas.DataFrame
        The dataframe where the filenames of fMRIs are stored.
    :param cache_folder: string
        The folder where to store the new files.
    :param ref_map: string
        The absolute path of a reference Nifti file whose affine will be used to
        resample others.
    :param overwrite: boolean
        If True, existing - allready resampled - files will be overwritten by
        the new ones.
    :param file_field: string
        The name of the columnn where the absolute path of the files to resample
        is stored.
    :param suffix: string
        The suffix to apply to both filenames and the column where to store
        their path.
    :param interpolation: string
        The interpolation (as named in nilearn's resample_to_img function) to
        apply.
    :param verbose: boolean
        Whether to print debugging messages.
    :return: pandas.DataFrame, int, set of indexes
        - The dataframe with the resampled absolute pathes
        - The number of encountered errors as well
        - The indexes of the fMRIs where they occured.
    """
    fmris_resamp = fmris

    errors = 0
    failed_MRIs = set()
    resamp_field = file_field + suffix
    if resamp_field not in fmris_resamp.columns:
        fmris_resamp[resamp_field] = ""

    resampled_fmris = []
    for idx, row in fmris.iterrows():
        file = row[file_field]
        if overwrite or not os.path.isfile(row[resamp_field]):
            try:
                # new file path by simply removing the ".gz" at the end
                new_file = (cache_folder
                            + "/fmri_"
                            + str(idx)
                            + suffix
                            + ".nii.gz")
                map_orig = nib.load(file)
                map_resampled = resample_to_img(map_orig,
                                                ref_map,
                                                interpolation=interpolation)
                # map_resampled.to_filename(new_file)
                resampled_fmris.append(map_resampled)
                fmris_resamp.at[idx, resamp_field] = new_file
                if verbose:
                    print(file, "resampled to", new_file)
            except Exception as e:
                print("Error resampling fMRI", idx)
                print(str(e))
                errors += 1
                failed_MRIs.update((idx,))
                # break

    return fmris_resamp, resampled_fmris, errors, failed_MRIs


def resample_batch(fmris_file,
                   ref_file,
                   cache_folder, path_file,
                   overwrite=False,
                   file_field="absolute_path",
                   interpolation="linear",
                   suffix="_resamp",
                   n_jobs=4,
                   verbose=False):

    fmris = pd.read_csv(fmris_file, low_memory=False, index_col=0)

    if verbose:
        print("> File read,",
              len(fmris[fmris["kept"]]),
              "fMRIs will be resampled with",
              interpolation,
              "interpolation")

    ref_map = nib.load(ref_file)

    fmris_split = np.array_split(fmris[fmris["kept"]], n_jobs)

    resamp = lambda x: resample_mini_batch(x, ref_map, cache_folder, overwrite,
                                           file_field=file_field,
                                           interpolation=interpolation,
                                           suffix=suffix,
                                           verbose=verbose)
    results = (Parallel(n_jobs=n_jobs, verbose=1, backend="threading")
               (delayed(resamp)(x) for x in fmris_split))

    fmris_resamp = pd.DataFrame()
    fmris_maps = []
    errors = 0
    failed = set()
    for result in results:
        fmris_resamp = pd.concat([fmris_resamp, result[0]])
        fmris_maps += result[1]
        errors += result[2]
        failed.update(result[3])

    pd.Series(list(failed)).to_csv(cache_folder + "/failed_resamples.csv",
                                   header=True)

    fmris_resamp[file_field + suffix].to_csv(path_file, header=True)

    return fmris_resamp, fmris_maps


def prepare_resample(global_config=None, n_jobs=1, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------
    config          = global_config["resample"]
    meta_path       = global_config["meta_path"]
    cache_path      = global_config["cache_path"]
    fmris_meta_file = meta_path + global_config["meta_file"]
    target_affine   = global_config["dict_file"]
    path_file       = cache_path + config["output_file"]


    # ------------------
    # --- RESAMPLING ---
    # ------------------
    if verbose:
        print("=" * 30)
        print(" > Resampling fMRIs using",
              target_affine,
              "as the target affine")

    fmris_df, fmris_maps = resample_batch(fmris_meta_file,
                   target_affine,
                   cache_path, path_file,
                   overwrite=config["overwrite"],
                   file_field=config["input_field"],
                   interpolation=config["interpolation"],
                   n_jobs=n_jobs,
                   verbose=verbose)

    print(">>> Resampling done")

    return fmris_maps


# =========================
# === MASKING FUNCTIONS ===
# =========================
def mask_mini_batch(fmris, masker, verbose=False):
    if verbose:
        print("  > masking started (one mini-batch, {} fMRIS)"
              .format(len(fmris)))

    X = np.zeros((len(fmris), masker.mask_img.get_data().sum()))
    i = 0
    for idx, fmri in enumerate(fmris):
        X[i] = masker.transform(fmri)
        i += 1

    if verbose:
        print("  > masking ended (one mini-batch)")

    return X


def mask_batch(fmris, masker, n_jobs=1, verbose=False):
    # fmris = pd.read_csv(fmris_file, index_col=0, header=0,
    #                     low_memory=False, squeeze=True)

    if verbose:
        print("> File read, {} fMRIs will be  masked".format(len(fmris)))

    fmri_split = np.array_split(fmris, n_jobs)
    ma = lambda x: mask_mini_batch(x, masker, verbose=verbose)
    results = (Parallel(n_jobs=n_jobs, verbose=1, backend="threading")
               (delayed(ma)(x) for x in fmri_split))

    X = np.vstack(results)

    return X


def prepare_mask(fmris_maps, global_config=None, n_jobs=1, verbose=False):
    if verbose:
        print("> Start masking...")

    # --------------
    # --- CONFIG ---
    # --------------
    config = global_config["mask"]

    # ---------------
    # --- MASKING ---
    # ---------------
    if verbose:
        print("=" * 30)
        print(" > Masking fMRIs using",
              global_config["mask_file"],
              "as the mask")

    mask = nib.load(global_config["mask_file"])

    if verbose:
        print("> Start fitting mask...")

    masker = NiftiMasker(mask_img=mask).fit()

    if verbose:
        print("> Applying fitted mask...")

    fmris_masked = mask_batch(
        fmris_maps,
        masker,
        n_jobs=n_jobs,
        verbose=verbose
    )

    return fmris_masked


# ===========================
# === EMBEDDING FUNCTIONS ===
# ===========================
def embed_from_atlas(fmris_masked, atlas_masked,
                     center=False, scale=False, absolute=False,
                     nan_max=1.0, tol=0.1,
                     projection=False,
                     verbose=False):
    """
    Embeds fMRI stat-maps using a dictionary of components,
    either projecting on it or regressing the data over it.

    :param fmris_masked_file:
    :param atlas_masked:
    :param center:
    :param scale:
    :param absolute:
    :param nan_max:
    :param projection:
    :param verbose:
    :return:
    """
    fmris_data = fmris_masked

    if absolute:
        if verbose:
            print("> Taking positive part...")
        fmris_data[fmris_data < 0] = 0

    if center or scale:
        if verbose:
            print("> Scaling...")
        fmris_data = preprocessing.scale(fmris_data,
                                         with_mean=center,
                                         with_std=scale,
                                         axis=1)

    if verbose:
        print("Calculating components...")

    if projection:
        result = fmris_data @ atlas_masked.T
    else:
        result = fmris_data @ np.linalg.pinv(atlas_masked)

    if nan_max < 1.0:
        if verbose:
            print("> Setting components with too many missing voxels to NaN...")
            print("  > Treshold =", nan_max)

        mask_missing_voxels = (np.abs(fmris_data) < tol)

        # if a voxel is missing and is part of a component,
        # this component is set to NaN:
        mask_result_nans = (mask_missing_voxels @ atlas_masked.T) > nan_max
        result[mask_result_nans] = np.nan

        if verbose:
            print("  > Total number of missing voxels:",
                  mask_missing_voxels.sum())
            print("  > Total number of missing components:",
                  mask_result_nans.sum())

    return result


def prepare_embed(fmris_masked, global_config=None, verbose=False):
    if verbose:
        print("> Start embedding...")

    config = global_config["embed"]

    mask = nib.load(global_config["mask_file"])
    atlas = nib.load(global_config["dict_file"])

    masker = NiftiMasker(mask_img=mask).fit()
    atlas_masked = masker.transform(atlas)

    fmris_embedded = embed_from_atlas(
        fmris_masked,
        atlas_masked,
        center=config["center"],
        scale=config["scale"],
        nan_max=config["nan_max"],
        verbose=verbose
    )

    if verbose:
        print("> Embedding finished, saving to file...")

    with open(config["output_file"], 'wb') as f:
        pickle.dump(fmris_embedded, f, protocol=4)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A masking scriptfor data fetched from Neurovault.",
        epilog='''Example: python a4_mask.py -C config.json -j 8 -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./preparation_config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        help="Number of jobs")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) debugging outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    fmris_maps = prepare_resample(global_config, args.jobs, args.verbose)
    fmris_masked = prepare_mask(fmris_maps, global_config, args.jobs, args.verbose)
    prepare_embed(fmris_masked, global_config, args.verbose)
