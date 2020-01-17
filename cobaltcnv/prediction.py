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
import sys
import os
from cobaltcnv import util, model, hmm, transform, vcf, qc
from cobaltcnv import __version__
from cobaltcnv.distributions import PosDependentSkewNormal
import logging
import pysam

class CNVCall(object):
    """
    Represents a single CNV call spanning one or more targets. These are produced by a segmentation
     or decoding algorithm, as yet TBD
    """

    def __init__(self, chrom, start, end, copynumber, ref_ploidy, quality, targets):
        self.chrom = chrom
        self.start = start      # Minimal ('inner') genomic start of call
        self.end = end          # Minimal ('inner') genomic end coordinate of call
        self.outer_start = None # Outer / maximal genomic start coordinate of call
        self.outer_end = None   # Outer / maximal genomic end coordinate of call
        self.copynum = copynumber
        self.ref_ploidy = ref_ploidy
        self.quality = quality
        self.targets = targets
        self.cn_exp = None # mean of copynumber expectation across targets
        self.cn_std = None # std dev of copynumber means across targets
        self.cn_std_mean = None # Mean of copynumber stds for each target
        self.cn_lower_conf = None # Lower confidence boundary for CN estimate
        self.cn_upper_conf = None # upper conf boundary

    def __str__(self):
        return "CNVCall {}:{}-{} quality: {:.3f} targets: {}".format(self.chrom, self.start, self.end, self.quality, self.targets)

    def phred_quality(self):
        return -10.0 * np.log10(self.quality)


class ProtoCNV(object):
    """
    Created during segmentation procedure to store partial information about a CNV call.
    """

    def __init__(self, all_regions, copynumber, ref_ploidy):
        self.all_regions = all_regions
        self.copy_number = copynumber
        self.ref_ploidy = ref_ploidy
        self.region_indices = []
        self.quals = []
        self.means = []
        self.variances = []

    def add_region(self, index, qual, mean, variance):
        self.region_indices.append(index)
        self.quals.append(qual)
        self.means.append(mean)
        self.variances.append(variance)

    @property
    def inner_start(self):
        return min(self.all_regions[i][1] for i in self.region_indices)

    @property
    def inner_end(self):
        return max(self.all_regions[i][2] for i in self.region_indices)

    @property
    def chrom(self):
        """
        Return the chromosome on which the CNV lies (computed on the fly from the list of included regions
        """
        chroms = set( self.all_regions[i][0] for i in self.region_indices)
        if len(chroms) > 1:
            raise ValueError("Found multiple chromosomes in a single CNV call! : {}".format(chroms))

        return list(chroms)[0]

    def trim_lowqual_edge_regions(self):
        """
        Experimental feature to remove regions on the either edge of the CNV that are much lower quality than the
        remaining regions. This is to avoid the situation where one very high confidence region borders a much lower
        confidence region, leading to a combined quality is low enough to cause the entire call to be ignored. In such
        cases its better to retain the few good targets.
        """
        if len(self.region_indices) < 2:
            return

        max_to_trim = min(len(self.region_indices) // 2 + 1, 5)  # Dont remove more than this number of targets from either edge

        qmean = np.mean(self.quals)
        left_trim = 0
        for i in range(1, max_to_trim):
            newqual = np.mean(self.quals[0:i])
            r = qmean / newqual
            if qmean / newqual > 1.25:
                left_trim = i
                qmean = np.mean(self.quals[left_trim:])

        qmean = np.mean(self.quals)
        right_trim = len(self.region_indices)
        for j in range(1, max_to_trim):
            newqual = np.mean(self.quals[len(self.region_indices) - j:])
            if qmean / newqual  > 1.25:
                right_trim = len(self.region_indices) - j
                qmean = np.mean(self.quals[0:right_trim])

        if left_trim > 0 or right_trim < len(self.region_indices):
            logging.info("Trimming CNV with quals {} (prev qual {:.4f}) to [{}:{}] (new qual {:.4f})".format(",".join("{:.3f}".format(x) for x in self.quals), np.mean(self.quals), left_trim, right_trim, np.mean(self.quals[left_trim:right_trim])))

        logging.debug("Trimming {} left and {} right regions from CNV {}:{}-{} (previously had {} targets)".format(left_trim, right_trim, self.chrom, self.inner_start, self.inner_end, len(self.region_indices)))
        self.region_indices = self.region_indices[left_trim:right_trim]
        self.quals = self.quals[left_trim:right_trim]
        self.means = self.means[left_trim:right_trim]
        self.variances = self.variances[left_trim:right_trim]


    def build_call(self, trim_lowqual_edges=False):
        """
        Create and return a single CNVCall object, with attributes set appropriately using the data stored in
        region_indices, quals, means, etc.
        :param all_regions: List of region tuples [(chrom, start, end), ...]
        :return: CNVCall object
        """

        if trim_lowqual_edges:
            self.trim_lowqual_edge_regions()

        chrom = self.chrom
        in_start = self.inner_start
        in_end = self.inner_end

        try:
            outer_start = max(r[2] for r in self.all_regions if r[2] < in_start and r[0] == chrom)
        except:
            outer_start = in_start

        try:
            outer_end = min(r[1] for r in self.all_regions if r[1] > in_end and r[0] == chrom)
        except:
            outer_end = in_end

        call = CNVCall(self.chrom, in_start, in_end, self.copy_number, self.ref_ploidy, 0, len(self.region_indices))
        call.outer_start = outer_start
        call.outer_end = outer_end
        call.quality = np.mean(self.quals)
        call.cn_exp = np.mean(self.means)
        call.cn_std = np.std(self.means)
        call.cn_std_mean = np.mean(np.sqrt(self.variances))
        call.cn_lower_conf = np.mean([max(0.0, m - 2.0 * np.sqrt(v)) for m, v in zip(self.means, self.variances)])
        call.cn_upper_conf = np.mean([m + 2.0 * np.sqrt(v) for m, v in zip(self.means, self.variances)])
        return call



def _create_hmm(params, mods, alpha, beta, states_to_use=None, ref_ploidy=2):
    """
    Create an HMM object using emission distribution params from the pcamode
    :param pcamodel:
    :param alpha: Transition probability parameter
    :param states_to_use: List of state indices to use, other states will be ignored
    :return: HMM object ready for prediction
    """
    state_count = len(mods)
    if states_to_use is not None:
        state_count = len(states_to_use)

    transitions = util.gen_transition_matrix(alpha, beta, state_count)

    emissions = []
    for state_index, (em_params, mod) in enumerate(zip(params, mods)):

        # If states to use has been specified and this state_index is not in it, skip it
        if states_to_use is not None and state_index not in states_to_use:
            continue

        if mod < 0.1:
            desc = "Hom. deletion"
            copy_number = 0
        elif mod < 0.75:
            desc = "Het deletion"
            copy_number = 1 * ref_ploidy / 2.0
        elif mod < 1.25:
            desc = "Diploid"
            copy_number = 2 * ref_ploidy / 2.0
        elif mod < 1.75:
            desc = "Het duplication"
            copy_number = 3 * ref_ploidy / 2.0
        else:
            desc = "Hom dup / amplification"
            copy_number = 4 * ref_ploidy / 2.0


        dist = PosDependentSkewNormal(em_params, user_desc=desc, copy_number=copy_number)
        emissions.append(dist)

    if states_to_use is not None and len(emissions) != len(states_to_use):
        raise ValueError('Number of states found not equal to number of states provided, was a state_index specified that wasn\'t available?')

    inits = [1.0 / len(mods) for _ in range(state_count)]
    return hmm.HMM(transitions, emissions, inits)

def gaussian_kullback_leibler(mu1, sig1, mu2, sig2):
    """
    Return the kullback-leibler divergence between two univariate Gaussian distributions
    :param mu1: Mean of first distribution
    :param sig1: Std dev of first distribution
    :param mu2: Mean of second dist
    :param sig2: Std dev of second dist
    :return: KL Divergence metric
    """
    return np.log(sig2 / sig1) + (sig1*sig1 + np.power(mu1-mu2, 2.0))/(2.0*sig2*sig2) - 0.5


def emit_site_info(model_path, emit_bed=False, outfile=None):
    cmodel = model.load_model(model_path)

    if not emit_bed:
        cmodel.describe() # Emits to sys.stdout by default
        return

    if hasattr(cmodel, 'mask'):
        mask = cmodel.mask
    else:
        mask = None

    modelhmm = _create_hmm(cmodel.params, cmodel.mods, 0.05, 0.05)
    dip_index = next((i, em) for i, em in enumerate(modelhmm.em) if em.copy_number() == 2)[0]
    # dip_index = int(len(cmodel.params) / 2)
    outlines = []
    mask_index = 0
    for nomask_index, region in enumerate(cmodel.regions):
        if mask is not None and not mask[nomask_index]:
            mask = f"{region[0]}\t{region[1]}\t{region[2]}\tMASKED"
            if outfile:
                outlines.append(mask)
            else:
                print(mask)
        else:
            dip_mu = cmodel.params[dip_index][mask_index][1]
            dip_sigma = cmodel.params[dip_index][mask_index][2]
            del_mu = cmodel.params[dip_index-1][mask_index][1]
            del_sigma = cmodel.params[dip_index - 1][mask_index][2]
            divergence = gaussian_kullback_leibler(del_mu, del_sigma, dip_mu, dip_sigma)
            div_line = f"{region[0]}\t{region[1]}\t{region[2]}\t{divergence:.4}"
            if outfile is not None:
                outlines.append(div_line)
            else:
                print(div_line)
            mask_index += 1
    if outfile is not None:
        return util.write_file(outfile, outlines)


def segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy, trim_low_quality_edges):
    """
    A very simple segmentation algorithm that just groups calls based on whether or not they
    have the same most-likely copy number state. Quality is the geometric mean of those probabilities
    :param regions: List of regions (from util.read_regions)
    :param stateprobs: State probabilities (from hmm.forward_backward)
    :param modelhmm: HMM instance with emission distributions
    :param ref_ploidy: Reference ploidy, used to construct CNVCall object
    :param trim_low_quality_edges: If True, use special trimming algorithm to remove low quality edge targets from CNVs
    :return: List of CNVCall objects
    """

    cnvs = []
    diploid_state = next( (i,em) for i,em in enumerate(modelhmm.em) if em.copy_number()==ref_ploidy)[0]
    prev_state = diploid_state
    current_cnv = None

    for j, (region, probs) in enumerate(zip(regions, stateprobs)):

        if region[1] == 196743945:
            print("hi!")

        state = np.argmax(probs)
        end_existing_cnv = False
        make_new_cnv = False

        if state==prev_state:
            if state == diploid_state:
                continue
            elif region[0] == current_cnv.chrom:
                current_cnv.add_region(j,
                                       probs[state],
                                       copynumber_expectation(probs, modelhmm=modelhmm),
                                       copynumber_variance(probs, modelhmm=modelhmm))
            else:
                end_existing_cnv = True
                make_new_cnv = True

        elif state==diploid_state:
            end_existing_cnv = True

        elif current_cnv is not None and current_cnv.chrom != region[0]:
            end_existing_cnv = True
            make_new_cnv = True

        else:
            end_existing_cnv = True
            make_new_cnv = True

        prev_state = state
        if end_existing_cnv and current_cnv is not None:
            cnvs.append(current_cnv.build_call(trim_lowqual_edges=trim_low_quality_edges))
            current_cnv = None

        if make_new_cnv:
            if current_cnv is not None:
                raise ValueError('Cant make a new CNV while one already exists')
            current_cnv = ProtoCNV(regions, modelhmm.em[state].copy_number(), ref_ploidy)
            current_cnv.add_region(j,
                                   probs[state],
                                   copynumber_expectation(probs, modelhmm=modelhmm),
                                   copynumber_variance(probs, modelhmm=modelhmm))

    # Dont forget to add the last CNV if we end traversing regions still in CNV state
    if current_cnv is not None:
        cnvs.append(current_cnv.build_call(trim_lowqual_edges=trim_low_quality_edges))

    return cnvs


def _chrom_include_exclude(chrom, chroms_to_include=None, chroms_to_exclude=None):
    """
    Return true if chrom is in chroms_to_include and not in chroms_to_exclude, with the catch
    that if chroms_to_include or _exclude is None then don't filter on inclusion (or exclusion)
    If both are None than everything will be included
    :param chrom: Name of chromosome to test
    :param chroms_to_include: List of chromosomes to match, or None to match all
    :param chroms_to_exclude: List of chromosomes to exclude by, or None to never exclude
    :return: True if included and not excluded
    """
    included = chroms_to_include is None or chrom in chroms_to_include
    excluded = (not chroms_to_exclude is None) and chrom in chroms_to_exclude
    return included and not excluded


def _filter_regions_by_chroms(regions, depths, params, chroms_to_include, chroms_to_exclude):
    if len(regions) != len(depths):
        raise ValueError("Number of regions must be equal to number of depths")
    if len(regions) != len(params[0]):
        raise ValueError("Number of regions must be equal to number of parameters")
    flt_depths = [depth for region, depth in zip(regions, depths) if _chrom_include_exclude(region[0], chroms_to_include, chroms_to_exclude)]
    flt_regions = [region for region in regions if _chrom_include_exclude(region[0], chroms_to_include, chroms_to_exclude)]
    flt_params = []
    for ps in params:
        flt_params.append([param for region, param in zip(regions, ps) if _chrom_include_exclude(region[0], chroms_to_include, chroms_to_exclude)])
    return flt_regions, flt_depths, flt_params


def copynumber_expectation(probs, modelhmm):
    """
    Given probabilities for a single target (typically computed via forward-backward), take them to be a discrete
     probability distribution and compute the expectation of the copy number.

    :param probs: List of state probabilities
    :param modelhmm: HMM, needed to associate states with copy numbers
    :param ref_ploidy: Usually two, should be 1 for Y anc X on males
    :return: Expectation of copy number
    """
    return sum( modelhmm.em[i].copy_number()*p for i,p in enumerate(probs) )


def copynumber_variance(probs, modelhmm):
    """
    Return variance in copy number for a single target, computed similarly to
    :param probs: List of state probabilities
    :return: The variance in the copy number estimate
    """

    expectation = copynumber_expectation(probs, modelhmm)
    # Floating point issues can occasionally produce a negative variance, so we truncate at 0.0 here
    # Looks long, but basically we compute variance as E[X^2] - E[X]^2, where X is the copy number
    variance = max(0.0, sum( modelhmm.em[i].copy_number()*modelhmm.em[i].copy_number()*p for i,p in enumerate(probs) ) - expectation * expectation)
    return variance


def emit_target_info(region, probs, fh, modelhmm, std=None, ref_ploidy=2):
    """
    Compute the copy-number expectation, stdev and log2 of the expectation for the given probability list
     and write them to the given file handle
    :param region: Region tuple (chrom, start, end)
    :param probs: List of state probabilties
    :param fh: File-like object to write to
    """
    cn_exp = copynumber_expectation(probs, modelhmm)
    cn_var = copynumber_variance(probs, modelhmm)

    try:
        cn_stdf = np.sqrt(cn_var)
        cn_std = "{:.4f}".format(cn_stdf)
    except:
        cn_std = "NA"

    try:
        log2 = "{:.4f}".format(np.log2(cn_exp/ref_ploidy))
    except:
        log2 = "NA"

    try:
        cn_exp = "{:.4f}".format(cn_exp)
    except:
        cn_exp = "NA"

    if std is not None:
        fh.write("\t".join([region[0], str(region[1]), str(region[2]), cn_exp, cn_std, log2, "{:.4f}".format(std)]) + "\n")
    else:
        fh.write("\t".join([region[0], str(region[1]), str(region[2]), cn_exp, cn_std, log2]) + "\n")

def standardize_depths(transformed_depths, params):
    """
    Subtract from each transformed mean the expected mean of diploid samples at that target, then divide by the
    standard deviation for that target to produce a target-specific Z-score
    :param cmodel: CNV calling model
    :param transformed_depths:
    :param diploid_index: Index of diploid vals in parameter lists
    :return: Diploid target-standardized depths
    """

    diploid_means = np.array([p[1] for p in params])
    diploid_variances = np.array([p[2] for p in params])
    return (transformed_depths - diploid_means) / np.sqrt(diploid_variances)

def construct_hmms_call_states(cmodel, regions, transformed_depths, alpha, beta, use_male_chrcounts, trim_lowqual_edges, emit_each_target_fh=None):
    """
    Construct HMMs using information stored in model, compute state probabilities for all targets, and
    run segmentation algorithm on state probability matrices to generate CNV calls.
    :param cmodel: PCAFlexChunk model with emission distribution params
    :param regions: List of regions with mask applied (all masked regions must be removed so that this is equal in length to the parameters in pcamodel)
    :param transformed_depths: Sample depths transformed via prepping / PCA transform
    :param alpha: Transition matrix parameter a
    :param beta: Transition matrix parameter b
    :param use_male_chrcounts: If true, assume there's only one X chromosome and modify potential CN states accordingly
    :param trim_lowqual_edges: If true, trim off low quality edge targets from CNV calls
    :param emit_each_target_fh: If given, must be a file-like object to which target-specific information will be written
    :return: List of CNVCall objects
    """


    autosomal_regions, autosomal_depths, autosomal_params = _filter_regions_by_chroms(regions, transformed_depths, cmodel.params, chroms_to_include=None, chroms_to_exclude=util.SEX_CHROMS)
    x_regions, x_depths, x_params = _filter_regions_by_chroms(regions, transformed_depths, cmodel.params, chroms_to_include=util.X_CHROM, chroms_to_exclude=None)
    y_regions, y_depths, y_params = _filter_regions_by_chroms(regions, transformed_depths, cmodel.params, chroms_to_include=util.Y_CHROM, chroms_to_exclude=None)

    logging.info("Determining autosomal copy number states and qualities")
    autosomal_model = _create_hmm(autosomal_params, cmodel.mods, alpha, beta, ref_ploidy=2)

    # diploid state is the INDEX of the column in state probs that corresponds to diploid, typically 2
    diploid_state = next((i, em) for i, em in enumerate(autosomal_model.em) if em.copy_number() == 2)[0]
    autosomal_state_probs = autosomal_model.forward_backward(autosomal_depths)[1:]
    autosomal_std = standardize_depths(autosomal_depths, autosomal_params[diploid_state])
    autosomal_cnvs = segment_cnvs(autosomal_regions, autosomal_state_probs, autosomal_model, ref_ploidy=2, trim_low_quality_edges=trim_lowqual_edges)

    x_cnvs = []
    x_state_probs = []
    x_std = []
    if len(x_regions)>0:
        if use_male_chrcounts:
            states_to_use = [0, 2, 4]
            ref_ploidy = 1
        else:
            states_to_use = None
            ref_ploidy = 2

        logging.info("Determining X chromosome copy number states and qualities")
        x_model = _create_hmm(x_params, cmodel.mods, alpha, beta, states_to_use=states_to_use, ref_ploidy=ref_ploidy)
        # x_diploid_state = next((i, em) for i, em in enumerate(x_model.em) if em.copy_number() == 2)[0]
        x_state_probs = x_model.forward_backward(x_depths)[1:]
        x_std = standardize_depths(x_depths, x_params[diploid_state])
        x_cnvs = segment_cnvs(x_regions, x_state_probs, x_model, ref_ploidy=ref_ploidy, trim_low_quality_edges=trim_lowqual_edges)
    else:
        logging.info("No X chromosome regions found, not calling CNVs on X chromosome")


    y_cnvs = []
    y_state_probs = []
    y_std = []
    if len(y_regions)>0 and use_male_chrcounts:
        logging.info("Determining Y chromosome copy number states and qualities")
        y_model = _create_hmm(y_params, cmodel.mods, alpha, beta, states_to_use=[0, 2, 4], ref_ploidy=1)
        # y_diploid_state = next((i, em) for i, em in enumerate(y_model.em) if em.copy_number() == 2)[0]
        y_state_probs = y_model.forward_backward(y_depths)[1:]
        y_std = standardize_depths(y_depths, y_params[diploid_state])
        y_cnvs = segment_cnvs(y_regions, y_state_probs, y_model, ref_ploidy=1, trim_low_quality_edges=trim_lowqual_edges)
    else:
        if use_male_chrcounts:
            logging.info("No Y chromosome regions found, not calling CNVs on X chromosome")
        else:
            logging.info("Not calling CNVs on Y (sample is female)")


    if emit_each_target_fh:
        logging.info("Writing target specific information")
        emit_each_target_fh.write("#chrom start end mean_cn std_cn log2 dev\n".replace(" ", "\t"))
        for region, probs, std in zip(autosomal_regions, autosomal_state_probs, autosomal_std):
            emit_target_info(region, probs, emit_each_target_fh, modelhmm=autosomal_model, std=std, ref_ploidy=2)

        for region, probs, std in zip(x_regions, x_state_probs, x_std):
            if use_male_chrcounts:
                ref_ploidy = 1
            else:
                ref_ploidy = 2

            emit_target_info(region, probs, emit_each_target_fh, modelhmm=x_model, std=std, ref_ploidy=ref_ploidy)

        for region, probs, std in zip(y_regions, y_state_probs, y_std):
            ref_ploidy = 1
            emit_target_info(region, probs, emit_each_target_fh, modelhmm=y_model, std=std, ref_ploidy=ref_ploidy)

    return list(autosomal_cnvs) + list(x_cnvs) + list(y_cnvs)


def has_x_regions(regions):
    """
    Returns true if any region in the list has chrom == 'X' or 'chrX'
    :param regions:
    :return:
    """
    return any(r[0] == 'X' or r[0] == 'chrX' for r in regions)


def mask_prepare_transform_depths(cmodel, depths):
    """
    Apply a region mask in the cmodel if present to the given depths, then 'prep' them and transform
    :param cmodel: Cobalt model
    :param depths: Raw sample depths
    :return: Tuple of Regions with mask targets removed, transformed depths
    """
    if hasattr(cmodel, 'mask') and cmodel.mask is not None:
        logging.info("Applying region mask, removing {} targets".format( len(cmodel.regions)-sum(cmodel.mask)))
        depths = depths[cmodel.mask, :]
        regions = [r for r,m in zip(cmodel.regions, cmodel.mask) if m]

    prepped_depths = transform.prep_data(depths)
    transformed_depths = transform.transform_by_genchunks(prepped_depths, cmodel)
    return regions, transformed_depths



def call_cnvs(cmodel, depths, alpha, beta, assume_female, genome, trim_lowqual_edges, emit_each_target_fh=None):
    """
    Discover CNVs in the list of depths using a CobaltModel (cmodel)
    :param cmodel: CobaltModel object
    :param depths: np.Matrix of sample depths
    :param alpha: FIrst transition prob param
    :param beta: Second transition prob param
    :param assume_female: Use female X chrom model (if false, use male, if None, infer)
    :return: List of CNVCall objects representing all calls
    """
    regions, transformed_depths = mask_prepare_transform_depths(cmodel, depths)

    if hasattr(cmodel, 'comps') and cmodel.comps is not None:
        mean_dist, score = qc.compute_mean_dist(cmodel, transformed_depths)
        logging.info("Sample QC metric: {:.4f}".format(score))


    return construct_hmms_call_states(cmodel,
                                      regions,
                                      transformed_depths,
                                      alpha,
                                      beta,
                                      trim_lowqual_edges=trim_lowqual_edges,
                                      use_male_chrcounts=not assume_female,
                                      emit_each_target_fh=emit_each_target_fh)

def depths_seem_female(cmodel, depths, genome):
    """
    Return True if there are at least a few targets on the X and the ratio of depths on X
    targets to autosomal targets is > 0.75 (as we would expect if the sample had 2 X chromosomes)
    :param cmodel: CNV calling model
    :param depths: Raw read depths
    :param genome: Genome build, need this for PAR regions, should be one of "hg19" or "hg38"
    """
    if not has_x_regions(cmodel.regions):
        return False # No X regions, so we cant really tell, but return False in case there are Y targets
    xratio = util.x_depth_ratio(cmodel.regions, depths, genome)
    female = xratio > 0.75
    if female:
        logging.info("Inferred sample sex is female (X / A ratio: {:.3f})".format(xratio))
    else:
        logging.info("Inferred sample sex is male (X / A ratio: {:.3f})".format(xratio))
    return female

def emit_bed(cnv_calls, min_quality, output_fh):
    """
    Write the list of CNV calls in BED format to the given output handle
    :param cnv_calls: CNV call objects
    :param min_quality: Minimum quality for including call
    :param output_fh: File-like object to write output to
    """
    output_fh.write("#chrom\tstart\tend\tcopy_number\tquality\ttargets\n")
    for call in cnv_calls:
        quality = "{:.3f}".format(call.quality)
        if call.quality >= min_quality:
            output_fh.write("\t".join([str(s) for s in
                                   [call.chrom, call.start, call.end, call.copynum, quality,
                                    call.targets]]) + "\n")

def emit_vcf(cnv_calls, samplename, min_quality, ref_path, output_fh):
    """
    Write the list of CNV calls
    :param cnv_calls:
    :param min_quality: Min quality to write CNV
    :param ref_path: Path to reference fasta
    :param output_fh: File-like object to write results to
    """
    header = vcf.VCF_HEADER.format(ver=__version__, cmd=" ".join(sys.argv), sample=samplename)
    output_fh.write(header + "\n")
    ref = pysam.FastaFile(ref_path)
    for cnv in cnv_calls:
        if cnv.quality >= min_quality:
            output_fh.write(vcf.cnv_to_vcf(cnv, ref, min_quality) + "\n")

def run_qc(model_path, depths_path, outputfh):
    """
    Emit basic QC info to the given output filehandle in CSV format
    :param model_path: Path to a cobalt model with QC info (comps & directions) stored
    :param depths_path: Path to depths file
    :param outputfh: File handle to write results to (if None write to sys.stdout)
    """
    if outputfh is None:
        outputfh = sys.stdout
    else:
        outputfh = open(outputfh, "w")

    cmodel = model.load_model(model_path)

    if not hasattr(cmodel, 'comps') or cmodel.comps is None:
        raise ValueError('Sorry, looks like the model {} does not contain stored data needed to compute QC metrics, please re-generate the model with cobalt 0.7.1 or later')

    sample_depths, sample_names = util.read_data_bed(depths_path)
    logging.info("Computing QC metrics for {} sample{}".format(len(sample_names), 's' if len(sample_names) > 1 else ''))
    outputfh.write("sample,mean_distance,score\n")

    for i, name in enumerate(sample_names):
        depths = np.matrix(sample_depths[:, i]).reshape((sample_depths.shape[0], 1))
        _, transformed_depths = mask_prepare_transform_depths(cmodel, depths)
        mean_dist, score = qc.compute_mean_dist(cmodel, transformed_depths)
        outputfh.write("{},{:.3f},{:.4f}\n".format(name, mean_dist, score))

    try:
        outputfh.close()
    except Exception as ex:
        logging.debug("Error closing output stream for QC writing: {}".format(ex))


def predict(model_path,
            depths_path,
            alpha=0.05,
            beta=0.05,
            output_path=None,
            output_bed=None,
            min_quality=0.90,
            assume_female=None,
            genome=util.ReferenceGenomes.B37,
            outputvcf=False,
            ref_path=None,
            emit_target_path=None,
            samplename=None,
            trim_lowqual_edges=True):
    """
    Run the prediction algorithm to identify CNVs from a test sample, using eigenvectors and parameters stored in a model file
    :param model_path: Path to model file, generated via a call to trainpca.train(...)
    :param depths_path: Path to depths BED-formatted file,
    :param min_quality: Minimum required CNV quality to include in output
    :param alpha: Transition probability parameter
    :param beta: Transition probability parameter
    :param assume_female: If true, allow all states on X chrom, if false, only allow homozygous dels and dups on X, if None, try to infer sex
    :param emit_target_path: Path to which to write per-target info. If None don't write anything
    """

    cmodel = model.load_model(model_path)
    sample_depths, sample_names = util.read_data_bed(depths_path)
    if samplename is None:
        samplename = sample_names[0]
    # regions = util.read_regions(depths_path)  # Compare these to the model regions to ensure they're the same??

    if sample_depths.shape[1] > 1:
        logging.warning("Found multiple samples in depths bed file, only the first will be analyzed")

    depths = np.matrix(sample_depths[:, 0]).reshape((sample_depths.shape[0], 1))


    logging.info("Beginning prediction run with alpha = {}, beta = {} and min_output_quality = {}".format(alpha, beta,
                                                                                                          min_quality))
    emit_each_target_fh = None
    if emit_target_path is not None:
        logging.info("Will write target specific info to {}".format(emit_target_path))
        emit_each_target_fh = open(emit_target_path, "w")

    if assume_female is None:
        assume_female = depths_seem_female(cmodel, depths, genome)
    elif assume_female:
        logging.info("Assuming sample sex is female")
    else:
        logging.info("Assuming sample sex is male")

    cnv_calls = call_cnvs(cmodel, depths, alpha, beta,
                          assume_female,
                          genome=genome,
                          trim_lowqual_edges=trim_lowqual_edges,
                          emit_each_target_fh=emit_each_target_fh)


    output_fh = sys.stdout
    if output_path is not None:
        logging.info("Done computing probabilities, emitting output to {}".format(output_path))
        output_fh = open(output_path, "w")

    if outputvcf:
        emit_vcf(cnv_calls, samplename=samplename, ref_path=ref_path, min_quality=min_quality, output_fh=output_fh)
        if output_bed is not None:
            output_bedfh = open(output_bed, "w")
            emit_bed(cnv_calls, min_quality=min_quality, output_fh=output_bedfh)
    else:
        emit_bed(cnv_calls, min_quality=min_quality, output_fh=output_fh)



