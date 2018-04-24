
import numpy as np
import sys
import os
from cobaltcnv import util, model, hmm, transform, vcf
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
        self.start = start
        self.end = end
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



def _create_hmm(params, mods, alpha, beta, states_to_use=None):
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
            copy_number = 1
        elif mod < 1.25:
            desc = "Diploid"
            copy_number = 2
        elif mod < 1.75:
            desc = "Het duplication"
            copy_number = 3
        else:
            desc = "Hom dup / amplification"
            copy_number = 4


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

def emit_site_info(model_path, emit_bed=False):
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

    mask_index = 0
    for nomask_index, region in enumerate(cmodel.regions):
        if mask is not None and not mask[nomask_index]:
            print("{}\t{}\t{}\t{}".format(region[0], region[1], region[2], "MASKED"))
        else:
            dip_mu = cmodel.params[dip_index][mask_index][1]
            dip_sigma = cmodel.params[dip_index][mask_index][2]
            del_mu = cmodel.params[dip_index-1][mask_index][1]
            del_sigma = cmodel.params[dip_index - 1][mask_index][2]
            divergence = gaussian_kullback_leibler(del_mu, del_sigma, dip_mu, dip_sigma)
            print("{}\t{}\t{}\t{:.4}".format(region[0], region[1], region[2], divergence))
            mask_index += 1


def segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy):
    """
    A very simple segmentation algorithm that just groups calls based on whether or not they
    have the same most-likely copy number state. Quality is the geometric mean of those probabilities
    :param regions: List of regions (from util.read_regions)
    :param stateprobs: State probabilities (from hmm.forward_backward)
    :param modelhmm: HMM instance with emission distributions
    :param ref_ploidy: Reference ploidy, used to construct CNVCall object
    :return: List of CNVCall objects
    """

    cnvs = []
    diploid_state = next( (i,em) for i,em in enumerate(modelhmm.em) if em.copy_number()==2)[0]
    cnv_state = diploid_state
    current_cnv = None
    quals = [] # Running list of single-target qualities to use for building the final quality
    means = [] # Ditto for copy-number expectations
    variances = [] # Ditto for copy-number variances

    for j, (region, probs) in enumerate(zip(regions, stateprobs)):

        state = np.argmax(probs)
        end_existing_cnv = False
        make_new_cnv = False

        if state==cnv_state:
            if state == diploid_state:
                continue
            elif region[0] == current_cnv.chrom:
                current_cnv.targets += 1
                current_cnv.end = region[2]
                quals.append(probs[state])
                means.append(copynumber_expectation(probs))
                variances.append(copynumber_variance(probs))
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

        cnv_state = state
        if end_existing_cnv and current_cnv is not None:
            current_cnv.quality = np.mean(quals)
            current_cnv.cn_exp = np.mean(means)
            current_cnv.cn_std = np.std(means)
            current_cnv.cn_std_mean = np.mean(np.sqrt(variances))
            current_cnv.cn_lower_conf = np.mean([max(0.0, m - 2.0 * np.sqrt(v)) for m, v in zip(means, variances)])
            current_cnv.cn_upper_conf = np.mean([m + 2.0 * np.sqrt(v) for m, v in zip(means, variances)])
            cnvs.append(current_cnv)
            quals = []
            means = []
            variances = []
            current_cnv = None

        if make_new_cnv:
            if current_cnv is not None:
                raise ValueError('Cant make a new CNV while one already exists')
            current_cnv = CNVCall(region[0], region[1], region[2], modelhmm.em[state].copy_number(), ref_ploidy, 0, 1)
            quals.append(probs[state])
            means.append(copynumber_expectation(probs))
            variances.append(copynumber_variance(probs))

    if current_cnv is not None:
        current_cnv.quality = np.mean(quals)
        current_cnv.cn_exp = np.mean(means)
        current_cnv.cn_std = np.std(means)
        current_cnv.cn_std_mean = np.mean(np.sqrt(variances))
        current_cnv.cn_lower_conf = np.mean([max(0.0, m-2.0*np.std(v)) for m,v in zip(means, variances)])
        current_cnv.cn_upper_conf = np.mean([m+2.0*np.std(v) for m,v in zip(means, variances)])
        cnvs.append(current_cnv)

    return cnvs


def _filter_regions_by_chroms(regions, depths, params, chroms_to_include):
    flt_depths = [depth for region, depth in zip(regions, depths) if region[0] in chroms_to_include]
    flt_regions = [region for region in regions if region[0] in chroms_to_include]
    flt_params = []
    for ps in params:
        flt_params.append([param for region, param in zip(regions, ps) if region[0] in chroms_to_include])
    return flt_regions, flt_depths, flt_params

def copynumber_expectation(probs):
    """
    Given probabilities for a single target (typically computed via forward-backward), take them to be a discrete
     probability distribution and compute the expectation of the copy number.

    !! This assumes that the copy numbers associated with each state are the INDEX of the state, so that probs[0]
    describes the probability of having 0 copies, probs[2] is probability of diploid, etc. !!

    :param probs: List of state probabilities
    :return: Expectation of copy number
    """
    return sum( i*p for i,p in enumerate(probs) )


def copynumber_variance(probs):
    """
    Return variance in copy number for a single target, computed similarly to
    :param probs: List of state probabilities
    :return:
    """
    expectation = copynumber_expectation(probs)
    # Floating point issues can occasionally produce a negative variance, so we truncate at 0.0 here
    variance = max(0.0, sum( i*i*p for i,p in enumerate(probs) ) - expectation * expectation)
    return variance


def emit_target_info(region, probs, fh, std=None):
    """
    Compute the copy-number expectation, stdev and log2 of the expectation for the given probability list
     and write them to the given file handle
    :param region: Region tuple (chrom, start, end)
    :param probs: List of state probabilties
    :param fh: File-like object to write to
    """
    cn_exp = copynumber_expectation(probs)
    cn_var = copynumber_variance(probs)

    try:
        cn_stdf = np.sqrt(cn_var)
        cn_std = "{:.4f}".format(cn_stdf)
    except:
        cn_std = "NA"

    try:
        log2 = "{:.4f}".format(np.log2(cn_exp/2.0))
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

def construct_hmms_call_states(cmodel, regions, transformed_depths, alpha, beta, use_male_chrcounts, emit_each_target_fh=None):
    """
    Construct HMMs using information stored in model, compute state probabilities for all targets, and
    run segmentation algorithm on state probability matrices to generate CNV calls.
    :param cmodel: PCAFlexChunk model with emission distribution params
    :param regions: List of regions with mask applied (all masked regions must be removed so that this is equal in length to the parameters in pcamodel)
    :param transformed_depths: Sample depths transformed via prepping / PCA transform
    :param alpha: Transition matrix parameter a
    :param beta: Transition matrix parameter b
    :param use_male_chrcounts: If true, assume there's only one X chromosome and modify potential CN states accordingly
    :param emit_each_target_fh: If given, must be a file-like object to which target-specific information will be written
    :return: List of CNVCall objects
    """

    diploid_state = 2
    autosomal_regions, autosomal_depths, autosomal_params = _filter_regions_by_chroms(regions, transformed_depths, cmodel.params, util.AUTOSOMES)
    x_regions, x_depths, x_params = _filter_regions_by_chroms(regions, transformed_depths, cmodel.params, util.X_CHROM)
    y_regions, y_depths, y_params = _filter_regions_by_chroms(cmodel.regions, transformed_depths, cmodel.params, util.Y_CHROM)

    logging.info("Determining autosomal copy number states and qualities")
    autosomal_model = _create_hmm(autosomal_params, cmodel.mods, alpha, beta)
    autosomal_state_probs = autosomal_model.forward_backward(autosomal_depths)[1:]
    autosomal_std = standardize_depths(autosomal_depths, autosomal_params[diploid_state])
    autosomal_cnvs = segment_cnvs(autosomal_regions, autosomal_state_probs, autosomal_model, ref_ploidy=2)

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
        x_model = _create_hmm(x_params, cmodel.mods, alpha, beta, states_to_use=states_to_use)
        x_state_probs = x_model.forward_backward(x_depths)[1:]
        x_std = standardize_depths(x_depths, x_params[diploid_state])
        x_cnvs = segment_cnvs(x_regions, x_state_probs, x_model, ref_ploidy=ref_ploidy)
    else:
        logging.info("No X chromosome regions found, not calling CNVs on X chromosome")


    y_cnvs = []
    y_state_probs = []
    y_std = []
    if len(y_regions)>0 and use_male_chrcounts:
        logging.info("Determining Y chromosome copy number states and qualities")
        y_model = _create_hmm(y_params, cmodel.mods, alpha, beta, states_to_use=[0, 2, 4])
        y_state_probs = y_model.forward_backward(y_depths)[1:]
        y_std = standardize_depths(y_depths, y_params[diploid_state])
        y_cnvs = segment_cnvs(y_regions, y_state_probs, y_model, ref_ploidy=1)
    else:
        if use_male_chrcounts:
            logging.info("No Y chromosome regions found, not calling CNVs on X chromosome")
        else:
            logging.info("Not calling CNVs on Y (sample is female)")


    if emit_each_target_fh:
        logging.info("Writing target specific information")
        emit_each_target_fh.write("#chrom start end mean_cn std_cn log2 dev\n".replace(" ", "\t"))
        for region, probs, std in zip(autosomal_regions, autosomal_state_probs, autosomal_std):
            emit_target_info(region, probs, emit_each_target_fh, std)

        for region, probs, std in zip(x_regions, x_state_probs, x_std):
            emit_target_info(region, probs, emit_each_target_fh, std)

        for region, probs, std in zip(y_regions, y_state_probs, y_std):
            emit_target_info(region, probs, emit_each_target_fh, std)

    return list(autosomal_cnvs) + list(x_cnvs) + list(y_cnvs)


def has_x_regions(regions):
    """
    Returns true if any region in the list has chrom == 'X' or 'chrX'
    :param regions:
    :return:
    """
    return any(r[0] == 'X' or r[0] == 'chrX' for r in regions)


def call_cnvs(cmodel, depths, alpha, beta, assume_female, genome, emit_each_target_fh=None):
    """
    Discover CNVs in the list of depths using a CobaltModel (cmodel)
    :param cmodel: CobaltModel object
    :param depths: np.Matrix of sample depths
    :param alpha: FIrst transition prob param
    :param beta: Second transition prob param
    :param assume_female: Use female X chrom model (if false, use male, if None, infer)
    :return: List of CNVCall objects representing all calls
    """
    if hasattr(cmodel, 'mask') and cmodel.mask is not None:
        logging.info("Applying region mask, removing {} targets".format( len(cmodel.regions)-sum(cmodel.mask)))
        depths = depths[cmodel.mask, :]
        regions = [r for r,m in zip(cmodel.regions, cmodel.mask) if m]

    prepped_depths = transform.prep_data(depths)
    # colmeans = np.median(prepped_depths, axis=1)
    # centered = prepped_depths - colmeans # HOW IS THIS SOMEHOW NOT REQUIRED??
    transformed_depths = transform.transform_by_genchunks(prepped_depths, cmodel)

    return construct_hmms_call_states(cmodel,
                                      regions,
                                      transformed_depths,
                                      alpha,
                                      beta,
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


def predict(model_path,
            depths_path,
            alpha=0.05,
            beta=0.05,
            output_path=None,
            min_quality=0.90,
            assume_female=None,
            genome=util.ReferenceGenomes.B37,
            outputvcf=False,
            ref_path=None,
            emit_target_path=None,
            samplename=None):
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

    cnv_calls = call_cnvs(cmodel, depths, alpha, beta, assume_female, genome=genome, emit_each_target_fh=emit_each_target_fh)


    output_fh = sys.stdout
    if output_path is not None:
        logging.info("Done computing probabilities, emitting output to {}".format(output_path))
        output_fh = open(output_path, "w")

    if outputvcf:
        emit_vcf(cnv_calls, samplename=samplename, ref_path=ref_path, min_quality=min_quality, output_fh=output_fh)
    else:
        emit_bed(cnv_calls, min_quality=min_quality, output_fh=output_fh)



