
import numpy as np

VCF_HEADER = """##fileformat=VCFv4.2
##CobaltVersion="{ver}"
##CobaltCMD="{cmd}"
##INFO=<ID=TARGETS,Number=1,Type=String,Description="Number of targets spanned by variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##ALT=<ID=CN4,Description="Copy number allele: 4 copies">
##ALT=<ID=CN3,Description="Copy number allele: 3 copies">
##ALT=<ID=CN2,Description="Copy number allele: 2 copies">
##ALT=<ID=CN1,Description="Copy number allele: 1 copy">
##ALT=<ID=CN0,Description="Copy number allele: 0 copies">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}"""


def cnv_to_vcf(cnv, ref, passqual):
    """
    Return a VCF-record formatted string containing the information for the given cnv
    :param cnv:
    """
    refbase = ref.fetch(cnv.chrom, cnv.start, cnv.start+1)
    alt = "<CN{}>".format(cnv.copynum)

    if cnv.quality > passqual:
        filter = "PASS"
    else:
        filter = "LOWQUAL"

    if cnv.quality >= 1.0:
        phredqual = 1000
    else:
        phredqual = min(1000, int(round(-10.0 * np.log10(1.0 - cnv.quality))))

    info = "TARGETS={};END={};SVTYPE=CNV;CN={}".format(cnv.targets, cnv.end,cnv.copynum)
    return "\t".join([
        cnv.chrom,
        str(cnv.start+1),
        ".",
        refbase,
        alt,
        "{}".format(phredqual),
        filter,
        info,
        ".",
        "."
    ])
