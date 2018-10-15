"""
This file is part of the Cobalt CNV detection tool.

Cobalt is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Cobalt is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Cobalt.  If not, see <https://www.gnu.org/licenses/>.
"""



import numpy as np
from cobaltcnv import util, model, transform
import logging

# The minimum size of a 'chunk' (list of regions over which the SVD is computed)
MIN_CHUNK_SIZE = 50


def _fit_sites(depth_matrix, depths_prepped, var_cutoff, mods, min_depth):
    """
    Transform the given data by fitting a PCA to the prepped depths, subtracting the given number of components, iteratively
    predicting the deviations produced by increasing or decreasing the raw depth values at each site, then returning the
    mean and standard deviation of the resulting distributions
    :param depth_matrix:  Raw, untransformed depth matrix
    :param depths_prepped: Depth matrix transposed and possibly modified by log-transforming, centering, etc. following 'prep_mode'
    :param colmeans: Values to subtract from each column of the raw depth matrix, typically from 'prep_mode'
    :param colsds: Column std. devs to divide each column
    :param rowmeans: Values to subtract from each target (row)
    :param var_cutoff: Remove components amounting to at least this amount of variance
    :param chunk_start: For logging only, chunk offset to add to reported site index
    :return: Tuple of (components, (params for each entry in mod), {dict of optimized sites -> components})
    """
    components = transform.fit_svd(depths_prepped, var_cutoff=var_cutoff)

    transformed = transform.transform_raw_iterative(depths_prepped, components, zscores=False)
    dmat = np.transpose(depth_matrix)
    all_params = [[] for _ in range(len(mods))]
    num_sites = depths_prepped.shape[1]

    for site in range(num_sites):
        fits = transform.fit_site2(dmat, depths_prepped, components, site, transformed, mods=mods, rmoutliers=False, min_depth=min_depth)
        for i,p in enumerate(fits):
            all_params[i].append(p)

    return components, all_params


def split_bychr(regions):
    """
    Return a list of (start, end) tuples that contain the first and last+1 indexes of each separate chromosome
    in the given regions list
    :param regions: List of tuples defining regions. Must be sorted by contig
    """

    bdys = []
    current_chr = None
    start_index = 0
    for i, r in enumerate(regions):
        if r[0] != current_chr:
            if current_chr is not None:
                bdys.append((start_index, i))
            current_chr = r[0]
            start_index = i
    bdys.append((start_index, i+1))
    return bdys


def gen_chunk_indices(regions, chunksize):
    """
    A new way to create chunks that just returns lists of array indices for chunks
    :param regions:
    :param chunksize:
    :return: A list containing arrays of array indices
    """
    if chunksize < MIN_CHUNK_SIZE:
        raise AttributeError('Minimum chunk size is {}'.format(MIN_CHUNK_SIZE))

    if chunksize > len(regions):
        chunksize = len(regions)
        logging.warning("Reducing chunk size to {} because there are only {} regions".format(chunksize, len(regions)))



    numchunks = len(regions) / chunksize
    numchunks = int(max(1, numchunks))
    indices = []
    for start in range(numchunks):
        indices.append(np.arange(start, len(regions), step=numchunks))
    return indices

def train(depths_path, model_save_path, use_depth_mask, var_cutoff, max_cv, chunk_size, min_depth, low_depth_trim_frac, high_depth_trim_frac, high_cv_trim_frac):
    """
    Train a new model by reading in a depths matrix, masking low quality sites, applying some transformation, removing PCA
    components in chunks, then estimating transformed depths of duplications and deletions and emitting them all in a
    model file.
    :param depths_path: Path to BED-formatted file containing matrix of depths x samples
    :param use_depth_mask: If true, remove 'poor' targets from analysis before creating model
    :param model_save_path: Path to which to save model
    :param num_components: Number of components / rank of reduced matrix to remove from depths
    :param max_cv: Maximum coefficient of variation of depths for samples,
    :param chunk_size: Approximate number of sites to include in each partition of the targets
    :param min_depth: Minimum depth of target for inclusion in model
    :param low_depth_trim_frac: Fraction of targets to remove due to low coverage
    :param high_depth_trim_frac: Fraction of targets to remove because of high coverage
    """

    logging.info("Starting new training run using depths from {}".format(depths_path))
    depth_matrix, sample_names = util.read_data_bed(depths_path)
    regions = util.read_regions(depths_path)

    if max_cv is not None:
        logging.info("Removing samples with depth CV > {}".format(max_cv))
        cvs = util.calc_depth_cv(depth_matrix)
        which = cvs > max_cv
        if all(which):
            logging.error("All samples have CV > {} !".format(max_cv))
            return

        if sum(which)==0:
            logging.info("No samples have CV < {}".format(max_cv))

        for i in range(len(cvs)):
            if which[i]:
                logging.info("Removing sample {} (CV={:.4f})".format(sample_names[i], cvs[i]))

        depth_matrix = util.remove_samples_max_cv(depth_matrix, max_cv=max_cv)

    logging.info("Beginning new training run removing {:.2f}% of variance and chunk size {}".format(100.0*var_cutoff, chunk_size))

    if use_depth_mask:
        logging.info("Creating target mask")
        mask = util.create_region_mask(depth_matrix,
                                       cvar_trim_frac=high_cv_trim_frac,
                                       low_depth_trim_frac=low_depth_trim_frac,
                                       high_depth_trim_frac=high_depth_trim_frac,
                                       min_depth=min_depth)
    else:
        logging.info("Skipping mask creation")
        mask = np.ones(shape=(depth_matrix.shape[0], )) == 1 # Convert 1 to True

    masked_regions = [r for r,m in zip(regions, mask) if m]
    masked_depths = depth_matrix[mask, :]


    depths_prepped = transform.prep_data(masked_depths)

    mods = [0.01, 0.5, 1.0, 1.5, 2.0]
    chunk_data = []
    all_params = [[-1 for a in range(masked_depths.shape[0])] for _ in range(len(mods))]

    chunk_indices = gen_chunk_indices(masked_regions, chunk_size)

    for i, indices in enumerate(chunk_indices):
        logging.info("Processing chunk {} of {}".format(i+1, len(chunk_indices)))
        depths_prepped_chunk = depths_prepped[:, indices]
        raw_depths_chunk = masked_depths[indices, :]
        components, params = _fit_sites(raw_depths_chunk,
                                        depths_prepped_chunk,
                                        var_cutoff,
                                        mods=mods,
                                        min_depth=min_depth)

        chunk_data.append((indices, components))
        for i,par in enumerate(params):
            for j,p in zip(indices, par):
                all_params[i][j] = p

    logging.info("Training run complete, saving model to {}".format(model_save_path))


    cobaltmodel = model.CobaltModel(chunk_data, all_params, mods, regions, mask=mask, samplenames=sample_names)
    model.save_model(cobaltmodel, model_save_path)

