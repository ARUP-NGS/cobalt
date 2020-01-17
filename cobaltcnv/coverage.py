#!/usr/bin/env python3

import pysam
import os
import multiprocessing as mp
from functools import partial
import argparse

from cobaltcnv import util

threads = mp.cpu_count()

def filter(r, min_mapq):
    return not (r.is_duplicate
                or r.is_secondary
                or r.is_supplementary
                or r.is_unmapped
                or (not r.is_proper_pair)
                or r.mapping_quality < min_mapq)

def count(args, ref_genome, readfilter):
    """
    Compute read counts over a list of regions for a single BAM file
    :param args: Tuple of (path to bam file, list of regions)
    :return: List of read counts, 1 entry for each region in input list
    """
    bampath, regions = args
    if ref_genome:
        alnfile = pysam.AlignmentFile(bampath, reference_filename=ref_genome)
    else:
        alnfile = pysam.AlignmentFile(bampath)
    counts = [alnfile.count(region[0], region[1], region[2], read_callback=readfilter) for region in regions]
    alnfile.close()
    return counts

def coverage(bed, bams, threads, min_mapq, ref_genome=None, outfile=None):
    outlines = []
    pool = mp.Pool( threads )
    sample_names = "\t".join([os.path.basename(b) for b in bams])
    header = f"#chrom\tstart\tend\t{sample_names}"
    if outfile is not None:
        outlines.append(header)
    else:
        print(header)
    #print("#chrom\tstart\tend\t" + "\t".join([b.split("/")[-3] for b in bams]))
    chunksize = 250
    chunks = []
    readfilter = partial(filter, min_mapq=min_mapq)
    countfunc = partial(count, ref_genome=ref_genome, readfilter=readfilter)
    for line in open(bed):
        if line.startswith("#"):
            continue
        region = line.strip().split("\t")
        region[1] = int(region[1])
        region[2] = int(region[2])
        chunks.append(region)
        if len(chunks) >= chunksize:
            counts = pool.map(countfunc, zip(bams, [chunks for _ in range(len(bams))]))
            for i, region in enumerate(chunks):
                cov = "\t".join(str(c[i]) for c in counts)
                cov_line = f"{region[0]}\t{region[1]}\t{region[2]}\t{cov}"
                if outfile is not None:
                    outlines.append(cov_line)
                else:
                    print(cov_line)
            chunks = []

    #Don't forget last few
    counts = pool.map(countfunc, zip(bams, [chunks for _ in range(len(bams))]))
    for i, region in enumerate(chunks):
        covs = "\t".join(str(c[i]) for c in counts)
        covs_line = f"{region[0]}\t{region[1]}\t{region[2]}\t{covs}"
        if outfile is not None:
            outlines.append(covs_line)
        else:
            print(covs_line)
    if outfile is not None:
        util.write_file(outfile, outlines)