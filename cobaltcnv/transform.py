

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

def fit_pca(prepped_depths, num_components=6):
    """
    Compute and return the eigenvectors associated with the top 'num_components' singular values
    :param depth_matrix: Matrix of depths, with rows as samples and columns as targets
    :return: Matrix of components, with num_components columns and [target count] rows
    """
    colmeans = np.median(prepped_depths, axis=0)
    centered = prepped_depths - colmeans
    U, S, V = randomized_svd(centered, n_components=num_components, n_oversamples=10)
    return np.matrix(V).T


# @util.timeit
def transform_raw(sample_depths, components=None, comp_mat=None, zscores=True):
    """
    Mean-center the sample_depths, then apply the PCA transformation to remove first several components
     Finally, convert resulting values to z-scores and return them
    Must provide one of components of comp_mat, comp_mat should be pre-computed comp.dot( comp.T) matrix,
    which can be huge and takes a long time to compute
    :param sample_depths: Array of depths for a sinlge sample
    :param components: Optional,Components matrix from PCA
    :param comp_mat: Optional, pre-computed components.dot( components.T ),
    :return: PCA transformed, z-scored depths
    """
    if components is None and comp_mat is None:
        raise ValueError("must provide either components or pre-multiplied components matrix")

    mat = sample_depths

    if components is None:
        compcomp = comp_mat
    else:
        compcomp = components.dot(components.T)
    transformed = mat - mat.dot( compcomp )
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

def fit_site2(dmat_raw, dmat_prepped, components, site, intermediate, mods, rmoutliers=False):
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
    if orig_row.mean() < 20:
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

