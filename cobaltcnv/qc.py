
import numpy as np
from cobaltcnv import util, transform
import logging
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist

def find_diploid_means_stds(params, dip_index=2):
    """
    Return the means and standard deviations of the of the diploid distributions
    :param params: Parameter arrays, ith entry should correspond to ith mod value, typically there are 5 and 2 is diploid
    :param dip_index: Index of diploid entries in params
    :return: tuple of array of means, array of variances
    """
    means = np.array([p[1] for p in params[dip_index]])
    stds = np.array([p[2] for p in params[dip_index]])
    return means, stds


def compute_background_pca(params, transformed_depths):
    """
    Standardize the given transformed depths using the means and variances found from the params parameter,
    then perform an SVD to
    :param params: Parameters array
    :param transformed_depths: Depths list, prepped and transformed
    :return: Tuple of depths projected onto principle directions (aka principle components), top 2 directions
    """
    means, stds = find_diploid_means_stds(params, dip_index=2)
    standardized_depths = (transformed_depths - means) / stds
    U, S, Vt = randomized_svd(standardized_depths, n_components=2, n_oversamples=5)
    V = Vt.T
    comps = standardized_depths.dot(V)
    return comps, V



def project_sample(cmodel, sample_transformed_depths):
    """
    Using the 'directions' (eigenvectors from PCA) stored in the model along with the means
    and standard deviations from the model parameters, standardize the given sample depths
    and project the
    :param cmodel: Cobalt model
    :param sample_transformed_depths: Prepped and transformed depths from
    :return: Projection of depths onto first two principle directions, thus a 2x1 numpy array
    """
    if not hasattr(cmodel, 'directions') or cmodel.directions is None:
        raise ValueError('Sorry, this model file does not contain stored QC information.')
    means, stds = find_diploid_means_stds(cmodel.params, dip_index=2)
    standardized_depths = (sample_transformed_depths - means) / stds
    projection = standardized_depths.dot(cmodel.directions)
    return projection

# def distance_zscore(cmodel, point):
#     """
#     Compute the mean distance of the given point to each other 2-d point from cmodel.comps (the projection
#     of each background sample along the first and second components), then divide by
#     :param cmodel:
#     :param point:
#     :return:
#     """

def compute_mean_dist(cmodel, sample_transformed_depths):
    """
    Project the transformed depths onto the eigenvectors stored in the model, then compute the mean
    distance of the resulting point to each point stored in the model (which was computed from samples
    used to create the model). Do the same thing for ALL pairwise distances in the background
    and report the distance of the query sample, z-scored by the distances among background samples.

    The result quantifies how distant the query sample is from the background samples, relative to the
    distances of the background samples to oen another. Bigger values mean the sample is farther from
    the norm, values above 2 or so are pretty suspicious.

    :param cmodel: Cobalt model
    :param sample_transformed_depths: Prepped and transformed depths
    :return: Standardized distance to background samples
    """
    projection = project_sample(cmodel, sample_transformed_depths)
    d = np.power(cmodel.comps[:, 0:2] - projection, 2)
    mean_dist = np.sum(np.sqrt(d[:,0] + d[:,1])) / cmodel.comps.shape[0]
    background_pairwise_dists = pdist(cmodel.comps[:,0:2])
    mean_bknd_dist = np.mean(background_pairwise_dists)
    std_bknd_dist = np.std(background_pairwise_dists)
    return np.abs((mean_dist - mean_bknd_dist) / std_bknd_dist)

