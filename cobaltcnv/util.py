
import logging
import pandas as pd
import numpy as np
import time
from collections import defaultdict

X_CHROM=["X", "chrX"]
Y_CHROM=["Y", "chrY"]
SEX_CHROMS= X_CHROM + Y_CHROM


# Locations of pseudoautosomal regions on X chromosome, see https://www.ncbi.nlm.nih.gov/grc/human
X_PAR1_B37 = [60001, 2699520]
X_PAR2_B37 = [154931044, 155260560]

X_PAR1_B38 = [10001, 2781479]
X_PAR2_B38 = [155701383, 156030895]

class ReferenceGenomes(object):
    B37 = "b37"
    B38 = "b38"

class InsufficientXRegionException(BaseException):
    pass

def is_bed_header(line):
    """
    Returns true if this line looks like a header and not a BED content line
    :param line:
    :return:
    """
    if line.startswith("#") or line.startswith("track"):
        return True
    toks = line.split("\t")
    if len(toks)<3:
        return True
    try:
        int(toks[1])
        int(toks[2])
        return False
    except:
        pass

    return True

def read_data_bed(bedfile, rescale_func=None):
    """
    Read depths from a BED-formatted file, where each column after the first three represents one sample
    :param bedfile:
    :return: Numpy matrix with regions as rows and samples as columns
    """
    with open(bedfile, "r") as bed_fh:
        first_line = bed_fh.readline()
        has_header = is_bed_header(first_line)
        if has_header:
            second_line = bed_fh.readline()
            numcols = len(second_line.split('\t'))
        else:
            numcols = len(first_line.split('\t'))
    header = 0 if has_header else None
    data = pd.read_csv(bedfile, sep="\t", usecols=range(3, numcols), header=header)
    num_regions, num_samples = data.shape
    logging.info("Read depths for {} regions for {} samples".format(num_regions, num_samples))
    for sample in data:
        logging.info("Mean depth for sample {} : {:.2f} (sd: {:.2f})".format(sample, data[sample].mean(), data[sample].std()))

    sample_names = list(data)

    if rescale_func is not None:
        rescaled_data = pd.DataFrame()
        for sample in data:
            scaled_vals = rescale_func(data[sample])
            rescaled_data[sample] = scaled_vals
        data = rescaled_data

    return data.values, sample_names

def sample_means(depths):
    """
    Return a vector of column means, which, if the input is a data matrix read in by read_depths_bed, is
    the mean depth for each sample
    :param depths: Matrix with regions as rows and samples as columns
    :return: Vector with sample (column) means
    """
    return np.mean(depths, axis=0)

def target_means(depths, means):
    """
    Return a vector containing the means of the rows (targets) after normalizing each sample by the column means
    :param depths: Raw depth matrix
    :param means: Sample exposures / mean depths
    :return: Vector of normalized row means
    """
    return np.mean(depths/means, axis=1)


def calc_depth_cv(depths):
    """
    Given a depth matrix with samples as columns and targets as rows (as read in by read_data_bed,
    compute and return a list of coefficients of variation in target coverage for each sample
    :param depths: depth matrix
    :return: Coefficients of variation for each sample
    """
    return np.std(depths, axis=0) / np.mean(depths, axis=0)

def parse_region_str(s):
    """
    Parse a region string that looks like chr17:100-500 and return the chrom, start, and end (17, 100, 500) values
    :return: Tuple of chrom, start, end
    """
    chrom, vals = s.split(":")
    if "-" in vals:
        start, end = vals.split("-")
    else:
        start = int(vals)
        end = start + 1
    return chrom, int(start), int(end)

def find_site_index_for_region(chrom, start, end, regions):
    hits = []
    for index, region in enumerate(regions):
        if region[0] == chrom:
            if not (start >= region[2] or region[1] >= end):
                hits.append(index)

    return hits



def remove_samples_max_cv(depths, max_cv = 0.8):
    """
    Return a new depth matrix containing only samples whose CV (from calc_depth_cv) is below the given
     threshold
    :param depths: Depth matrix
    :param max_cv: Maximum coeff of variation for a sample to be included
    :return: New depth matrix with samples removed
    """
    which = np.where(calc_depth_cv(depths) > max_cv)
    m = np.delete(depths, which, axis=1)
    return m


def create_region_mask(depths, cvar_trim_frac, low_depth_trim_frac, high_depth_trim_frac, min_depth):
    """
    Create a list of booleans, one for each region in depths.shape[0], signifying if the region is to be
    masked from further analysis.
    :param depths: Matrix of normalized depths with regions in rows and samples in columns
    :param max_coeff_var: Maximum allowed coefficient of variation, regions above this will be flagged as failing
    :param min_depth: Minimum mean target depth allowed
    :return: array of True/False indicating of regions is OK (False = masked)
    """
    means = np.mean(depths, axis=1) # Target means
    coeff_var = np.std(depths, axis=1) / means
    coeff_var = np.nan_to_num(coeff_var) # Some sites will have 0 mean depth, which leads to NaNs in the coeff_var array

    cvar_sorted = np.sort(coeff_var)
    cvar_cutoff = cvar_sorted[ int(cvar_sorted.shape[0]*(1.0-cvar_trim_frac))]

    means_sorted = np.sort(means)
    low_depth_cutoff = means_sorted[int(means_sorted.shape[0] * low_depth_trim_frac)]
    high_depth_cutoff = means_sorted[int(means_sorted.shape[0] * (1.0-high_depth_trim_frac))]

    min_depth_mask = means < min_depth
    logging.info("Min depth ({}) mask flagged {} of {} ({:.2%}) regions as failing".format(min_depth, min_depth_mask.sum(), depths.shape[0],
                                                                                           min_depth_mask.sum() / float(depths.shape[0])))
    low_mean_mask = means <= low_depth_cutoff
    logging.info("Low mean ({:.2f}) mask flagged {} of {} ({:.2%}) regions as failing".format(low_depth_cutoff, low_mean_mask.sum(), depths.shape[0],
                                                                                           low_mean_mask.sum() / float(depths.shape[0])))
    high_mean_mask = means > high_depth_cutoff
    logging.info("High mean ({:.2f}) mask flagged {} of {} ({:.2%}) regions as failing".format(high_depth_cutoff, high_mean_mask.sum(), depths.shape[0],
                                                                                               high_mean_mask.sum() / float(depths.shape[0])))
    high_var_mask = coeff_var >= cvar_cutoff
    logging.info("High CV ({:.2f}) mask flagged {} of {} ({:.2%}) regions as failing".format(cvar_cutoff, high_var_mask.sum(), depths.shape[0],
                                                                                             high_var_mask.sum() / float(depths.shape[0])))
    mask = np.logical_or(low_mean_mask, high_mean_mask)
    mask = np.logical_or(min_depth_mask, mask)
    mask = np.logical_or(high_var_mask, mask)

    mask = np.logical_not(mask)
    failing = len(mask)-mask.sum()

    logging.info("Flagged {} of {} ({:.2%}) regions as failing".format(failing, depths.shape[0], failing / float(depths.shape[0])))
    return mask


def read_regions(bed):
    """
    Return a list of regions (chrom, start, end) from a BED-formatted file, discarding all other information
    :param bed: Path to a BED-formatted text file
    :return: List of region tuples (chrom, start, end)
    """
    regions = []
    with open(bed) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            toks = line.split("\t")
            regions.append( (toks[0], int(toks[1]), int(toks[2])) )
    return regions

def gen_transition_matrix(alpha, beta, dimension):
    """
    Create a transition matrix suitable for HMM use of the given dimension
    alpha describes probability of moving from away from the middle state - moving to a CN-altered state
    beta describes the probability of transitioning back to a more normal state
    :param alpha: probability of moving from away from the middle state - moving to a CN-altered state
    :param beta: probability of transitioning back to a more normal state
    :param dimension: Number of rows & columns of transition matrix
    :return: np.matrix suitable for HMM use
    """
    vals = []
    if dimension==1:
        return np.matrix([[1.0]])

    for i in range(dimension):
        row = np.zeros(dimension)
        if i==0:
            row[i] = 1.0 - beta
            row[i+1] = beta
        elif i==(dimension-1):
            row[i] = 1.0 - beta
            row[i-1] = beta
        else:


            if i==(dimension/2):
                row[i] = 1.0 - 2*alpha
                row[i-1] = alpha
                row[i+1] = alpha
            elif i<(dimension/2):
                row[i] = 1.0 - alpha - beta
                row[i - 1] = alpha
                row[i + 1] = beta
            else:
                row[i] = 1.0 - alpha - beta
                row[i-1] = beta
                row[i+1] = alpha

        vals.append(row)

    return np.matrix(vals)

def fmt(s):
    """
    Formatter that works for values that are sometimes strings
    """
    try:
        return "{:.3f}".format(s)
    except:
        return str(s)

def timeit(method):
    """
    Quick timing decorator for profiling purposes
    :param method:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.info("{} {:.3f} sec".format(method.__name__, te-ts))
        return result

    return timed

def remove_outliers(vals, cutoff=2.0):
    """
    Experimental method for removing 'outliers', which are those that are
    more than 'cutoff' std deviations from the mean
    :param vals: List of values
    :param cutoff: Max standard deviation for inclusion
    :return: New list with outliers removed
    """
    v = (vals-np.mean(vals))/np.std(vals)
    which = (v < cutoff) & (v > -1.0*cutoff)

    return vals[which], np.where(which)

def intervals_overlap(region_a, region_b):
    """
    Return true if the two regions contain any mutual locations
    :param region_a: region tuple (chrom, start, end)
    :param region_b:
    :return:
    """
    if region_a[0] != region_b[0]:
        return False
    return not (region_a[1] >= region_b[2] or region_b[1] >= region_a[2])

def x_depth_ratio(regions, depths, genome, min_num_x_regions=10):
    """
    Compute the ratio of depths on the X chromosome to the autosomes
    :param depths: Depths matrix
    :param regions: List of region objects
    :param min_num_x_regions: Minimum number of regions on X chromosome required for calc to succeed (otherwise raise InsufficientXRegionException)
    :param genome: Reference genome build to use, needed for locations of PAR regions
    :return: Ratio of depths of the X chromosome to those on the autosomes
    """
    autosomes = [str(x) for x in range(22)]

    if genome==ReferenceGenomes.B37:
        xpar1 = X_PAR1_B37
        xpar2 = X_PAR2_B37
    elif genome==ReferenceGenomes.B38:
        xpar1 = X_PAR1_B38
        xpar2 = X_PAR2_B38
    else:
        raise ValueError('Unrecognized reference genome: {}'.format(genome))

    # Dont include PAR regions
    x_regions = [i for i,r in enumerate(regions)
                 if r[0].replace('chr', '').upper() =='X'
                 and not (intervals_overlap(r, xpar1) or intervals_overlap(r, xpar2))]

    a_regions = [i for i, r in enumerate(regions) if r[0].replace('chr', '').upper() in autosomes]

    if len(x_regions) < min_num_x_regions:
        raise InsufficientXRegionException()

    xdepths = depths[x_regions]
    autosome_depths = depths[a_regions]


    return xdepths.mean() / autosome_depths.mean()
