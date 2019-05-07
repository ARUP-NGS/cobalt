
import numpy as np

VCF_HEADER = """##fileformat=VCFv4.2
##CobaltVersion="{ver}"
##CobaltCMD="{cmd}"
##INFO=<ID=TARGETS,Number=1,Type=String,Description="Number of targets spanned by variant">
##INFO=<ID=SVEND,Number=1,Type=Integer,Description="End position of variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number estimate">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence boundary for CNV start position">
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence boundary for CNV end position">
##INFO=<ID=LOG2,Number=1,Type=Float,Description="Log2 adjusted deviation in read depth">
##INFO=<ID=LOG2UPPER,Number=1,Type=Float,Description="Upper confidence bound for log2">
##INFO=<ID=LOG2LOWER,Number=1,Type=Float,Description="Lower confidence bound for log2">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}"""


def cnv_to_vcf(cnv, ref, passqual):
    """
    Return a VCF-record formatted string containing the information for the given cnv
    :param cnv:
    """
    refbase = ref.fetch(cnv.chrom, cnv.start, cnv.start+1)
    if cnv.copynum < cnv.ref_ploidy:
        alt = "<DEL>"
    else:
        alt = "<DUP>"

    if cnv.quality > passqual:
        filter = "PASS"
    else:
        filter = "LOWQUAL"

    if cnv.quality >= 1.0:
        phredqual = 1000
    else:
        phredqual = min(1000, int(round(-10.0 * np.log10(1.0 - cnv.quality))))

    if cnv.ref_ploidy == 2:
        if cnv.copynum == 1:
            gt = "0/1"
        elif cnv.copynum == 0:
            gt = "1/1"
        elif cnv.copynum == 3:
            gt = "0/1"
        elif cnv.copynum > 3:
            gt = "./1"
        else:
            gt = "0/0"
    elif cnv.ref_ploidy == 1:
        gt = "1"

    cn_log2 = np.log2(cnv.cn_exp / cnv.ref_ploidy)
    cn_log2_lower = np.log2(cnv.cn_lower_conf / cnv.ref_ploidy)
    cn_log2_upper = np.log2(cnv.cn_upper_conf / cnv.ref_ploidy)
    info = ";".join([
        "TARGETS={}".format(cnv.targets),
        "SVEND={}".format(cnv.end),
        "SVTYPE=CNV",
        "CIPOS={},{}".format(cnv.outer_start + 1, cnv.start + 1),
        "CIEND={},{}".format(cnv.end + 1, cnv.outer_end + 1),
        "CN={}".format(cnv.copynum),
        "LOG2={:.4f}".format(cn_log2),
        "LOG2LOWER={:.4f}".format(cn_log2_lower),
        "LOG2UPPER={:.4f}".format(cn_log2_upper)])

    return "\t".join([
        cnv.chrom,
        str(cnv.start+1),
        ".",
        refbase,
        alt,
        "{}".format(phredqual),
        filter,
        info,
        "GT",
        gt
    ])
