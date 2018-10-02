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
from cobaltcnv import util
import logging
from sklearn.utils.extmath import randomized_svd

def prep_data(depth_matrix, rowmeans=None):
    """
    Add 1 to all depths, then take the log and return the transposed matrix
    :param depth_matrix: np.matrix with samples as columns and targets as rows
    :return: A new np.matrix containing transposed, transformed values from depth matrix
    """

    depths_prepped = np.log(depth_matrix+1.0) #
    depths_prepped = np.transpose(depths_prepped)
    return depths_prepped

def fit_svd(prepped_depths, var_cutoff, max_components=25):
    """
    Compute and return the eigenvectors associated with the top k singular values such that
    the proportion of variance explained by the k singular values is at least var_cutoff
    :param prepped_depths: Matrix of depths, with rows as samples and columns as targets, after 'prep'
    :param var_cutoff: Proportion of variance to remove
    :param max_components: Dont try to remove more than this number of components
    :return: Matrix of components, with num_components columns and [target count] rows
    """
    colmeans = np.median(prepped_depths, axis=0)
    centered = prepped_depths - colmeans
    # centered = prepped_depths
    U, S, V = randomized_svd(centered, n_components=max_components, n_oversamples=10)
    prop_v = np.cumsum(S * S / (np.sum(S * S)))  # Cumulative proportion of variance explained
    eigs = np.nonzero(prop_v > var_cutoff)[0][0]  # Number of components (columns of V) to retain
    logging.debug("Retaining top {} vectors from SVD".format(eigs))
    return np.matrix(V[0:eigs,:]).T


# @util.timeit
def transform_raw(mat, components=None, comp_mat=None, zscores=True):
    """
    Subtract from the sample_depths matrix the comp_mat matrix, or the inner product of the components with itself
    Then, optionally return either that result or the sample-level z-scores of the result
    Must provide one of components (matrix with vectors as columns) or comp_mat, comp_mat should be
    pre-computed comp.dot( comp.T) matrix,

    :param sample_depths: Array of depths for a sinlge sample
    :param components: Optional, components matrix from PCA
    :param comp_mat: Optional, pre-computed components.dot( components.T ),
    :return: Depth matrix with mat.components.components removed
    """
    if components is None and comp_mat is None:
        raise ValueError("must provide either components or pre-multiplied components matrix")

    if components is None:
        compcomp = comp_mat
    else:
        compcomp = components.dot(components.T)
    transformed = mat - mat.dot(compcomp)

    if not zscores:
        return transformed
    else:
        zscores = (transformed - np.median(transformed, axis=1)) / transformed.std(axis=1)
        return zscores

def transform_raw_iterative(sample_depths, components, zscores=True):
    """
    Produces results identical to transform_raw, but doesn't rely on computing components.dot( components.T), which
    can generate a huge matrix. This is likely to be slower, but won't eat all of the RAM.
    :param sample_depths:Vector of raw (uncentered) sample depths
    :param components: Matrix of pca components
    :return:
    """
    mat_centered = sample_depths
    result = np.matrix(mat_centered.copy())
    for i in range(components.shape[0]):
        result[:,i] -= mat_centered.dot(components.dot(components[i, :].T))
    if not zscores:
        return result
    else:
        zscores = (result - np.median(result, axis=1)) / result.std(axis=1)
        return zscores
    return result

def transform_single_site(depths, components, site, orig_scores=None, zscores=True):

    a = depths.dot(components.dot(components[site, :].T))
    result = depths[:,site].reshape((depths.shape[0], 1)) - a

    if not zscores:
        return result
    else:
        if orig_scores is None:
            raise ValueError("Must supply original transformed but not z-scored data for z-scores")
        b = orig_scores.copy()
        b[:,site] = result.reshape(result.shape[0], 1)
        zscores = (b - np.median(b, axis=1)) / b.std(axis=1)
        return zscores


def fast_fit_skewnorm(raw_obs):
    """
     Assume skewness is zero, just return the sample mean and stdev of the values
    """
    return [0.0, np.mean(raw_obs), np.std(raw_obs, ddof=1.0)]


# @util.timeit
def fit_single(dmat, components, site, intermediate, id_outliers=True, sample_mask=None, dbg=False):
    """
    Identify pca-transformed / z-scored values for a single site, then estimate the distribution
    parameters for it and return those
    :param dmat:  Depth matrix
    :param components: PCA components matrix
    :param site: Site to compute values for
    :param intermediate: PCA-transformed, but not z-scored matrix of unadjusted values
    :return: Parameter estimates for distribution
    """
    which = None
    try:
        adj_zscores = transform_single_site(dmat, components, site=site, orig_scores=intermediate, zscores=True)

        transformed_obs = adj_zscores[:, site]
        if id_outliers and sample_mask:
            raise ValueError('Cant find outliers and use sample mask at the same time')

        if id_outliers:
            transformed_obs, which = util.remove_outliers(transformed_obs.getA1(), cutoff=3.25)

        if sample_mask:
            transformed_obs = transformed_obs[sample_mask]

        a, shape, loc = fast_fit_skewnorm(transformed_obs)
    except Exception as ex:
        logging.error("Error computing parameters for site {} : {}".format(site, str(ex)))
        a, shape, loc = None, None, None

    return a, shape, loc, which

def fit_site2(dmat_raw, dmat_prepped, components, site, intermediate, mods, rmoutliers=False, min_depth=20):
    """
    Estimate distribution parameters for 'normal' (diploid, unadjusted), deletion (normal * 0.5) and
    duplication (normal * 1.5) values independently and return them all as a list of tuples
    :param dmat: Depth matrix
    :param components: PCA components
    :param site: Site in question
    :param intermediate: PCA-transformed but not z-scored depth matrix data
    :return: List of tuples [(deletion params), (diploid params), (duplication params)]
    """
    orig_z = dmat_prepped[:, site].copy()
    orig_row = dmat_raw[:, site].copy()
    if orig_row.mean() < min_depth:
        logging.warning("Insufficient coverage at site {}, returning default params".format(site))
        return [(0, 0.0, 1.0) for _ in range(len(mods))]

    result = []
    for mod in mods:
        dmat_prepped[:, site] = np.log(mod * orig_row + 1.0)
        a, shape, loc, _ = fit_single(dmat_prepped, components, site, intermediate, id_outliers=rmoutliers, sample_mask=None, dbg=False) # site == 5 and (mod == 1.0 or mod == 0.5))
        result.append( (a, shape, loc) )

    dmat_prepped[:, site] = orig_z # Not sure if this is required, but it seems like a good idea

    for r in result:
        if any(np.isnan(r)) or any(np.isinf(r)):
            return [(0, 0.0, 1.0) for _ in range(len(mods))]

    return result


def transform_by_genchunks(depths, cmodel):
    """
    Transform raw (actually, prepped) depths
    :param depths:
    :param cmodel: PCAGenChunksModel
    :return: PCA transformed depths
    """
    transformed_depths = np.zeros(depths.shape[1])
    for i, chunk_info in enumerate(cmodel.chunk_data):
        indices, components = chunk_info
        transformed_depths[indices] = transform_raw(depths[:, indices], components=components, zscores=True)

    return transformed_depths

