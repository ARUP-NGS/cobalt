
# Cobalt

Cobalt is a tool to detect germline copy-number variants (CNVs) from targeted (hybridization-capture) NGS data. It can build a 'model' from a set of control samples, then use the model to call CNVs from a test sample. The samples aren't strictly BAM / CRAM files, they're actually read depths measured at a specific set of genomic locations we call the "CNV targets". Usually the targets are just a (unpadded) BED file of the locations of the capture probes. Cobalt contains a utility called 'covcounter' to count reads over a list of targets for a set of BAM or CRAM files.    


## Installation and requirements

Cobalt is written in python and requires python 3.5 or later. If you have python installed, the easiest thing to do is type:

    pip install git+https://github.com/ARUP-NGS/cobalt.git@0.7.3

The `@0.7.3` part refers to the version of cobalt you wish to install, check out the [releases page](https://github.com/ARUP-NGS/cobalt/releases) to find the latest one.

You can also install from source by cloning in the git repo using the following command:

    git clone https://github.com/ARUP-NGS/cobalt.git

then navigating into the newly created cobalt directory (`cd cobalt`), and installing via pip like this:

    pip install .

Cobalt depends on a several other python packages (numpy, pandas, sklearn, etc.) that are installed automatically by pip. 

## Usage

Cobalt has two main modes: training and prediction. In the training mode a 'model' is
constructed from a set of background samples that are assumed to have few or no real CNVs.
In prediction mode, Cobalt uses a model to detect CNVs in a test sample.

### Training

In training mode, Cobalt reads a BED-formatted text file containing read depths for a set of background
samples and generates a model file. Basic usage looks like:

    cobalt train --depths my_background_depths.bed -o my_new_model.model

 Each row of the BED file should contain information for one targeted genomic region (often, one capture probe),
 and each column beyond the first three should contain information for one sample. For instance:

     #chrom   start   end    sampleA    sampleB    sampleC
     1        100     200     172          327      114
     1        253     342     212          197      142
     1        2104    2203    123          223      541
     2        1023    1153    43           55       97


Training may take some time. For an exome with several hundred thousand targets and 50 or so samples training may take 1-2 hours.



Additional options for training:

    --chunk-size [1000]
Approximate number of targets to include in one 'chunk'. Results are relatively insensitive to this value, but training
will become slower if this is much greater than 10,000 or so. Values less than 100 seem like a bad idea.

    --no-mask [False]
Cobalt by default will ignore very low depth, very high depth, or highly variable targets during training, this is called target 'masking'. Using this flag disables the masking feature and cobalt will try to fit every target. Note that for some targets fitting
may still be impossible, for instance if every background sample has zero depth.

    --var-cutoff [0.90]
Fraction of variance in depth matrix to remove for each chunk. The number of singular vectors subtracted from depth matrix will be computed using this value. Default = 0.90, which often results in 5-6 singular vectors being used.

    --min-depth [20]
Minimum absolute mean depth for each target; targets with less mean depth less than this value will be masked. Default = 20. 

    --low-depth-trim-frac [0.01]
Fraction of targets to mask due to low depth. If this is set to X, then the Xth fraction of targets with the lowest absolute depth (prior to normalization) will be masked.

    --high-depth-trim-frac [0.01]
Fraction of targets to mask due to high depth. If this is set to X, then the Xth fraction of targets with the highest absolute depth (prior to normalization) will be masked.

    --high-cv-trim-frac [0.01]
Fraction of targets to mask due to high coefficient of variation (CV). Targetwise CV is computed for each target, then the Xth highest are masked.

    --cluster-width [50]
Number of adjacent regions to include in a chunk. The effect of changing this value is likely to be small, but might be fun to play with for fine-tuning

We recommend at least 50 samples for training. Most importantly, the samples use for training MUST represent the
type of conditions used to create the test samples. For instance, training samples should have insert sizes,
on-target ratios, PCR duplicate percentages, that are typical of the test samples  - and these are just necessary,
but not sufficient conditions! As with all CNV detection tools, results will be dependent on how similar the training
samples are to the control samples. Even a few degrees difference during hybridization can have a profound impact
on the quality of the CNV calls.


### CNV discovery

Detecting CNVs in a test sample requires a model generated using the training procedure above. CNVs can
be written either in BED for VCF (4.2) format. In addition, log2 ratios and supporting information can
be written to a BED file if desired.

Typical CNV discovery usage looks like:

    cobalt predict -m my_new_model.model -d sample_depths.bed -o sample_cnvs.bed

 The 'sample_depths.bed' file should be a in format identical to the 'background_depths.bed' file, but contain information
 for just a single sample.

 In this example the output will be written to sample_cnvs.bed. This is a BED-formatted file that contains one row for each CNV detected.

 The output format is BED with the following columns:

  1. Chrom : Chromosome on which CNV exists
  2. Start : Genomic start coordinate of CNV
  3. End   : Genomic end coordinate of CNV
  4. Copy number: Most likely number of alleles at position, for instance, 1 for a heterozygous deletion on an autosome, 3 for a het duplication on an autosome,
  0 for homozygous or hemizygous deletions, etc
  5. Quality: Confidence score of detected CNV, with 1.0 indicating highest confidence.
  Investigations have suggested a quality cutoff of 0.9 - 0.95 is sensible for most datasets.
  6. Targets: Number of BED targets spanned by CNV


To write output in VCF, a fasta-formatted and .fai indexed reference genome must be supplied.
A typical command line might look like:

    cobalt predict -m my_new_model.model -d sample_depths.bed -o sample_cnvs.vcf --vcf --reference my_reference_genome.fasta


To write information for ALL (non-masked) targets regardless of CNV status, supply the

    --target-info all_target.info.bed

 option. This will create a BED formatted file with the following columns:

  1. Chrom : Chromosome of target
  2. Start : Genomic start coordinate of target
  3. End   : Genomic end coordinate of target
  4. Mean copy number (Expectation of copy-number value)
  5. Standard deviation of copy number value
  6. Log2 ratio of copy number value

 The sixth column is simply log2(expectation of copy number / 2) for the target.

 The all-target info BED file does NOT including information for masked targets. To identify which
 targets are masked in a given model, use the ```desc -b``` subcommand, described below.

### Model information

 Cobalt model files store a small amount of meta-information, including version, the number of targets included, etc.
  To display this information use:

     cobalt desc -m my_model.model

### Listing target information

 Cobalt can list information about targets, including masking status, with the following command:

     cobalt desc -m my_model.model -b

This produces BED-formatted output with one line for every target, and one additional column containing either
  'MASKED' if the target is masked, or a number indicating the statistical power of CNV calling at the target.
Values less than 10 indicate relatively weak power, while values greater than 20 indicate strong power. The value
  is the Kullback-Leibler divergence between the emission distribution associated with the diploid state and the heterozygous
  deletion state.

For instance in the following output:

    1	621744	622044	MASKED
    1	861297	861417	2.23
    1	865505	865745	87.12
    1	866384	866504	38.19

In this example, the first target is masked, and the second target should be regarded as relatively low quality.
In the remaining two targets CNV detection is predicted to be relatively accurate.

The intent behind this feature is to enable users to create subsets of targeted regions that are associated with confident CNV calls.
For instance, when designing a new CNV detection assay it may be desirable to exclude low-confidence regions or masked regions.
By processing the BED file designers can easily remove masked or poorly performing regions to create a set of high-confidence targets.


### Generating sample QC metrics (new in version 0.7.1)

Cobalt has the ability to compute a metric that reflects how 'close' a given sample is to those samples used to create the background. Small values
indicate that the sample is far away, and CNV calls might not reflect actual genomic events. To compute a QC value, use

    cobalt qc -m [model path] -d [sample depths] -o [output csv file]

The output file is in .csv format, and contains one line for each sample in the 'sample depths' file. The line includes the sample name, the mean distance to the background samples (less is better), and the qc score (less is worse).
Values for the QC score range from 0-1, with 1 indicating the sample is very close to other background samples, and 0 indicating its really far away.  In practice, values greater than about 0.75 indicate an OK fit, while values less than about 0.6 indicate very poor fit (with those in between being questionable). The format of the sample depths file is identical to that used for prediction (BED, with each row indicating depths for one target).


### Creating a depth BED file for training

The input to the training procedure requires a BED-formatted file containing read depths from the training
samples. Such a file can be created in several ways, including use of [GATK's DepthOfCoverage tool](https://software.broadinstitute.org/gatk/documentation/tooldocs/current/org_broadinstitute_gatk_tools_walkers_coverage_DepthOfCoverage.php)
 or [BEDTools multicov](http://bedtools.readthedocs.io/en/latest/content/tools/multicov.html). We also include a simple
 and multithreaded utility that creates a properly formatted file called ```covcounter```, which should be installed
 along with cobalt. To use ```covcounter```, just supply a BED file containing targets to analyze and one or more BAMS to
 compute coverage for, like this:

     covcounter --bed my_bed_file.bed --bams some_bam.bam other_bam.bam third.bam --threads 24 > control_sample_coverages.bed

 By default ```covcounter``` will use 2 threads, but should scale reasonably well with more.
 
 
## License

Cobalt is licensed under the [GNU General Public License (GPL) v3](https://www.gnu.org/licenses/gpl.txt). As they say: Cobalt is 
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

