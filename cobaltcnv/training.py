

import numpy as np
from cobaltcnv import util, model, transform
import logging


def _fit_sites(depth_matrix, depths_prepped, num_components, mods):
    """
    Transform the given data by fitting a PCA to the prepped depths, subtracting the given number of components, iteratively
    predicting the deviations produced by increasing or decreasing the raw depth values at each site, then returning the
    mean and standard deviation of the resulting distributions
    :param depth_matrix:  Raw, untransformed depth matrix
    :param depths_prepped: Depth matrix transposed and possibly modified by log-transforming, centering, etc. following 'prep_mode'
    :param colmeans: Values to subtract from each column of the raw depth matrix, typically from 'prep_mode'
    :param colsds: Column std. devs to divide each column
    :param rowmeans: Values to subtract from each target (row)
    :param num_components: Number of PCA components to subtract
    :param chunk_start: For logging only, chunk offset to add to reported site index
    :return: Tuple of (components, (params for each entry in mod), {dict of optimized sites -> components})
    """
    components = transform.fit_pca(depths_prepped, num_components=num_components)

    transformed = transform.transform_raw_iterative(depths_prepped, components, zscores=False)
    dmat = np.transpose(depth_matrix)
    all_params = [[] for _ in range(len(mods))]
    num_sites = depths_prepped.shape[1]

    for site in range(num_sites):

        fits = transform.fit_site2(dmat, depths_prepped, components, site, transformed, mods=mods, rmoutliers=False)
        for i,p in enumerate(fits):
            all_params[i].append(p)

    return components, all_params


def make_chunks_simple(chunk_size, site_max):
    """
    Generate ranges of size chunk_size, truncating the last to site_max
    WARNING: This could yield a very small chunk, for instance 1 site, if site_max = n*chunk_size+1
    :param chunk_size: Size of chunks to make (except last)
    :return: List of (start index, end index) tuples describing chunk boundaries
    """
    return [ (s, min(s+chunk_size, site_max)) for s in range(0, site_max, chunk_size)]

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

def region_dist(r0, r1):
    """
    Distance in bp between midpoints of two regions
    """
    if r0[0] != r1[0]:
        return float("inf")
    else:
        m0 = float(r0[1] + r0[2]) / 2.0
        m1 = float(r1[1] + r1[2]) / 2.0
        return abs(m0-m1)

def find_split(regions, start, end):
    """
    Find best split point for regions between the given indices
    :return: Index of best split point
    """
    best_index = start
    biggest_gap = float("-inf")
    for i in range(start, end-1):
        gap_size = region_dist(regions[i], regions[i+1])
        if gap_size > biggest_gap:
            best_index = i
            biggest_gap = gap_size
    return best_index

def split_chunk(regions, start, end, min_chunk_size):
    """
    For the current chunk defined by [start..end), attempt to divide it into two smaller regions, both of which
    must be greater than min_chunk_size
    :param regions:
    :param start: Current region start index
    :param end: Current region end index
    :param min_chunk_size: Minimum size of chunk
    :return:
    """
    if end-start < min_chunk_size:
        raise ValueError('Region is already smaller than min_chunk_size')
    if end-start < 2*min_chunk_size:
        raise ValueError('No divisions possible, region already smaller than 2*min_chunk_size')

    inner_start = start + min_chunk_size
    inner_end = end - min_chunk_size

    split_point = find_split(regions, inner_start, inner_end)
    return [(start, split_point+1), (split_point+1, end)]


def split_all_regions(regions, min_chunk_size, max_chunk_size):
    """
    Create a partitioning of the regions such that no partition contains more than max_chunk_size targets or less
    than min_chunk_size targets, and the boundaries between partitions are chosen so as to prefer splitting on
    larger distances, so that nearby regions (for instance, regions in genes) typically end up in the same partition
    :param regions: List of tuples defining all regions. Must be sorted by contig and start position.
    :param min_chunk_size: No partitions should include more than this number of regions
    :param max_chunk_size: No partitions should contain fewer than this number of regions
    :return: List of [start, end) tuples defining indexes of regions in each partition.
    """
    #First, split everything by chrom
    if max_chunk_size < 2*min_chunk_size:
        raise ValueError('Sorry, maximum chunk size must be greater than 2*minimum_chunk_size')
    chrom_splits = split_bychr(regions)
    final_chunks = []

    for start, end in chrom_splits:
        chr_chunks = [(start, end)]
        # Iterate over chunks in chr_chunks, if any are bigger than max_chunk_size, split them and add them to
        # a new list, then iterate over that.
        while any( ((e-s > max_chunk_size) for s,e in chr_chunks)):
            new_chunks = []
            for chunk in chr_chunks:
                if chunk[1] - chunk[0] > max_chunk_size:
                    subchunk0, subchunk1 = split_chunk(regions, chunk[0], chunk[1], min_chunk_size)
                    new_chunks.extend([subchunk0, subchunk1])
                else:
                    new_chunks.append(chunk)
            chr_chunks = new_chunks
        final_chunks.extend(chr_chunks)
    return final_chunks

def gen_chunk_indices(regions, chunksize):
    """
    A new way to create chunks that just returns lists of array indices for chunks
    :param regions:
    :param chunksize:
    :return: A list containing arrays of array indices
    """
    numchunks = len(regions) / chunksize
    numchunks = int(max(1, numchunks))
    indices = []
    for start in range(numchunks):
        indices.append(np.arange(start, len(regions), step=numchunks))
    return indices

def train(depths_path, model_save_path, use_depth_mask, num_components=6, max_cv=1.0, chunk_size=1000):
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

        if (len(sample_names)-sum(which))<num_components:
            logging.error("Need at least as many passing samples as components ({} samples passed CV filter, {} components)".format(sum(which), num_components))
            return

        if sum(which)==0:
            logging.info("No samples have CV < {}".format(max_cv))

        for i in range(len(cvs)):
            if which[i]:
                logging.info("Removing sample {} (CV={:.4f})".format(sample_names[i], cvs[i]))

        depth_matrix = util.remove_samples_max_cv(depth_matrix, max_cv=max_cv)

    logging.info("Beginning new training run using {} components and chunk size {}".format(num_components, chunk_size))

    if use_depth_mask:
        logging.info("Creating target mask")
        mask = util.create_region_mask(depth_matrix, max_coeff_var=0.6, min_depth=25.0)
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
                                        num_components,
                                        mods=mods)

        chunk_data.append((indices, components))
        for i,par in enumerate(params):
            for j,p in zip(indices, par):
                all_params[i][j] = p

    logging.info("Training run complete, saving model to {}".format(model_save_path))


    cobaltmodel = model.CobaltModel(chunk_data, all_params, mods, regions, mask=mask, samplecount=len(sample_names))
    model.save_model(cobaltmodel, model_save_path)

