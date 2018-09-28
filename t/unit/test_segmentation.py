
import pytest
from cobaltcnv.prediction import segment_cnvs

class MockHMM(object):

    def __init__(self, emissions):
        self.em = emissions

class MockDist(object):

    def __init__(self, copynum):
        self.cn = copynum

    def copy_number(self):
        return self.cn

class Region(object):

    def __init__(self, chrom, start, end):
        self.chrom = chrom
        self.start = start
        self.end = end


def test_simple_seg():
    stateprobs = [[0.0, 1.0, 0.0], # 0-10
                  [0.0, 0.3, 0.7], # 20-30
                  [0.0, 0.4, 0.6], # 40-50
                  [0.0, 0.9, 0.1], # 50-60
                  [0.9, 0.1, 0.0], # 70-80
                  [0.1, 0.9, 0.0], # 90-100
                  [0.8, 0.2, 0.0]] # 10-20

    regions = [
        ("1", 0, 10),
        ("1", 20, 30),
        ("1", 40, 50),
        ("1", 50, 60),
        ("1", 70, 80),
        ("1", 90, 100),
        ("2", 10, 20)
    ]

    modelhmm = MockHMM([
        MockDist(1),
        MockDist(2),
        MockDist(3)
    ])

    segments = segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy=2)

    assert len(segments)==3
    assert segments[0].targets == 2
    assert segments[0].start == 20
    assert segments[0].end == 50

    assert segments[1].start == 70
    assert segments[1].end == 80
    assert segments[1].targets == 1
    assert segments[1].quality == 0.90

    assert segments[2].chrom == "2"
    assert segments[2].start == 10
    assert segments[2].end == 20
    assert segments[2].targets == 1
    assert segments[2].quality == 0.80
