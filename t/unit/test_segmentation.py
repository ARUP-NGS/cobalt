
import pytest
from io import StringIO
from cobaltcnv.prediction import segment_cnvs, copynumber_expectation, emit_target_info, ProtoCNV, _chrom_include_exclude

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
        ("1", 53, 60),
        ("1", 70, 80),
        ("1", 90, 100),
        ("2", 10, 20)
    ]

    modelhmm = MockHMM([
        MockDist(1),
        MockDist(2),
        MockDist(3)
    ])

    segments = segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy=2, trim_low_quality_edges=False)

    assert len(segments)==3
    assert segments[0].targets == 2
    assert segments[0].start == 20
    assert segments[0].end == 50
    assert segments[0].outer_start == 10
    assert segments[0].outer_end == 53

    assert segments[1].start == 70
    assert segments[1].outer_start == 60
    assert segments[1].end == 80
    assert segments[1].outer_end == 90
    assert segments[1].targets == 1
    assert segments[1].quality == 0.90

    assert segments[2].chrom == "2"
    assert segments[2].start == 10
    assert segments[2].outer_start == 10
    assert segments[2].end == 20
    assert segments[2].outer_end == 20
    assert segments[2].targets == 1
    assert segments[2].quality == 0.80


def test_seg_all():
    stateprobs = [[0.0, 0.0, 1.0], # 0-10
                  [0.0, 0.3, 0.7], # 20-30
                  [0.0, 0.4, 0.6], # 40-50
                  [0.0, 0.1, 0.9]] # 53-60

    regions = [
        ("1", 0, 10),
        ("1", 20, 30),
        ("1", 40, 50),
        ("1", 53, 60),
    ]

    modelhmm = MockHMM([
        MockDist(1),
        MockDist(2),
        MockDist(3)
    ])

    segments = segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy=2, trim_low_quality_edges=False)

    assert len(segments)==1
    assert segments[0].targets == 4
    assert segments[0].start == 0
    assert segments[0].end == 60
    assert segments[0].outer_start == 0
    assert segments[0].outer_end == 60

def test_seg_switch():
    stateprobs = [[0.0, 1.0, 0.0], # 0-10
                  [0.0, 0.3, 0.7], # 20-30
                  [0.6, 0.4, 0.0], # 40-50
                  [0.6, 0.4, 0.0], # 53-60
                  [0.0, 0.9, 0.1]] # 70-80

    regions = [
        ("1", 0, 10),
        ("1", 20, 30),
        ("1", 40, 50),
        ("1", 53, 60),
        ("1", 70, 80),
    ]

    modelhmm = MockHMM([
        MockDist(1),
        MockDist(2),
        MockDist(3)
    ])

    segments = segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy=2, trim_low_quality_edges=False)

    assert len(segments)==2
    assert segments[0].targets == 1
    assert segments[0].start == 20
    assert segments[0].end == 30
    assert segments[0].outer_start == 10
    assert segments[0].outer_end == 40

    assert segments[1].targets == 2
    assert segments[1].start == 40
    assert segments[1].end == 60
    assert segments[1].outer_start == 30
    assert segments[1].outer_end == 70


def test_copynums_log2s_sexchroms():
    stateprobs = [[0.0, 1.0, 0.0], # 0-10
                  [0.0, 0.3, 0.7], # 20-30
                  [0.6, 0.4, 0.0], # 40-50
                  [0.6, 0.4, 0.0], # 53-60
                  [0.0, 0.9, 0.1]] # 70-80

    regions = [
        ("X", 0, 10),
        ("X", 20, 30),
        ("X", 40, 50),
        ("X", 53, 60),
        ("X", 70, 80),
    ]

    modelhmm = MockHMM([
        MockDist(0),
        MockDist(1),
        MockDist(2),
    ])

    segments = segment_cnvs(regions, stateprobs, modelhmm, ref_ploidy=1, trim_low_quality_edges=False)
    assert len(segments) == 2
    assert segments[0].targets == 1
    assert segments[0].copynum == 2
    assert segments[0].start == 20
    assert segments[0].end == 30

    assert segments[1].targets == 2
    assert segments[1].copynum == 0
    assert segments[1].start == 40
    assert segments[1].end == 60


def test_copynum_exp():
    stateprobs = [[0.0, 1.0, 0.0], # 0-10
                  [0.0, 0.3, 0.7], # 20-30
                  [0.6, 0.4, 0.0]] # 40-50

    modelhmm = MockHMM([
        MockDist(0),
        MockDist(2),
        MockDist(4),
    ])

    ex = copynumber_expectation(stateprobs[0], modelhmm=modelhmm)
    assert ex == pytest.approx(2.0)


    ex = copynumber_expectation(stateprobs[2], modelhmm=modelhmm)
    assert ex == pytest.approx(0.8)


def test_emit_target_info_ploidy1():
    stateprobs = [[0.0, 1.0, 0.0], # 0-10
                  [0.0, 0.0, 1.0], # 20-30
                  [1.0, 0.0, 0.0], # 40-50
                  [0.6, 0.4, 0.0], # 53-60
                  [0.0, 0.9, 0.1]] # 70-80

    regions = [
        ("X", 0, 10),
        ("X", 20, 30),
        ("X", 40, 50),
        ("X", 53, 60),
        ("X", 70, 80),
    ]

    stds = [1.0 for _ in range(len(regions))]


    mockhmm = MockHMM([
        MockDist(0),
        MockDist(1),
        MockDist(2),
    ])

    outfh = StringIO()

    for region, probs, std in zip(regions, stateprobs, stds):
        emit_target_info(region, probs, outfh, modelhmm=mockhmm, std=std, ref_ploidy=1)

    result = outfh.getvalue().strip().split("\n")
    assert len(result) == len(regions)
    for i, res in enumerate(result):
        toks = res.split("\t")
        assert str(toks[1]) == str(regions[i][1])
        assert str(toks[2]) == str(regions[i][2])
        if i == 0:
            assert float(toks[3]) == pytest.approx(1.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == pytest.approx(0.0) # Should be log2(1.0 / 1.0) = 0
        elif i == 1:
            assert float(toks[3]) == pytest.approx(2.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == pytest.approx(1.0) # Should be log2(2.0 / 1.0) = 1
        elif i == 2:
            assert float(toks[3]) == pytest.approx(0.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == float('-inf') # Should be log2(0.0 / 1.0) = -infinity
        elif i == 3:
            assert float(toks[3]) == pytest.approx(0.4)
            assert float(toks[5]) == pytest.approx(-1.3219) # Should be log2(0.4 / 1.0) = -1.321928...




def test_emit_target_info_ploidy2():
    stateprobs = [[0.0, 0.0, 1.0, 0.0, 0.0], # 0-10
                  [0.0, 0.0, 0.0, 1.0, 0.0], # 20-30
                  [0.0, 1.0, 0.0, 0.0, 0.0], # 40-50
                  [1.0, 0.0, 0.0, 0.0, 0.0], # 53-60
                  [0.0, 0.0, 0.0, 0.0, 1.0]] # 70-80

    regions = [
        ("X", 0, 10),
        ("X", 20, 30),
        ("X", 40, 50),
        ("X", 53, 60),
        ("X", 70, 80),
    ]

    stds = [1.0 for _ in range(len(regions))]


    mockhmm = MockHMM([
        MockDist(0),
        MockDist(1),
        MockDist(2),
        MockDist(3),
        MockDist(4),
    ])

    outfh = StringIO()

    for region, probs, std in zip(regions, stateprobs, stds):
        emit_target_info(region, probs, outfh, modelhmm=mockhmm, std=std, ref_ploidy=2)

    result = outfh.getvalue().strip().split("\n")
    assert len(result) == len(regions)
    for i, res in enumerate(result):
        toks = res.split("\t")
        assert str(toks[1]) == str(regions[i][1])
        assert str(toks[2]) == str(regions[i][2])
        if i == 0:
            assert float(toks[3]) == pytest.approx(2.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == pytest.approx(0.0) # Should be log2(2.0 / 2.0) = 0
        elif i == 1:
            assert float(toks[3]) == pytest.approx(3.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == pytest.approx(0.585) # Should be log2(3.0 / 2.0) = 0.5849...
        elif i == 2:
            assert float(toks[3]) == pytest.approx(1.0)
            assert float(toks[4]) == pytest.approx(0.0)
            assert float(toks[5]) == pytest.approx(-1.0) # Should be log2(1.0 / 2.0) = -1
        elif i == 3:
            assert float(toks[3]) == pytest.approx(0.0)
            assert float(toks[5]) == float('-inf') # Should be log2(0.0 / 2.0) = -infinity
        elif i == 4:
            assert float(toks[3]) == pytest.approx(4.0)
            assert float(toks[5]) == pytest.approx(1.0)  # Should be log2(4.0 / 2.0) = +1



def test_trim_lowqual_edges_left():
    allregions = [ ("X", i, i+5) for i in range(10, 100, 10)]

    protocnv = ProtoCNV(allregions, copynumber=1, ref_ploidy=2)
    protocnv.add_region(2, 0.7, 1.0, 0.1)
    protocnv.add_region(3, 1.0, 1.0, 0.1)
    protocnv.add_region(4, 1.0, 1.0, 0.1)

    cnv = protocnv.build_call(trim_lowqual_edges=True)
    assert cnv.targets == 2
    assert cnv.quality == 1.0
    assert cnv.start == 40
    assert cnv.end == 55

def test_trim_lowqual_edges_left_multiple():
    allregions = [ ("X", i, i+5) for i in range(10, 100, 10)]

    protocnv = ProtoCNV(allregions, copynumber=1, ref_ploidy=2)
    protocnv.add_region(2, 0.7, 1.0, 0.1)
    protocnv.add_region(3, 0.6, 1.0, 0.1)
    protocnv.add_region(4, 1.0, 1.0, 0.1)
    protocnv.add_region(5, 0.9, 1.0, 0.1)
    protocnv.add_region(6, 0.99, 1.0, 0.1)

    cnv = protocnv.build_call(trim_lowqual_edges=True)
    assert cnv.targets == 3
    assert cnv.quality == pytest.approx(0.9633333)
    assert cnv.start == 50
    assert cnv.end == 75


def test_trim_lowqual_edges_right():
    allregions = [ ("X", i, i+5) for i in range(10, 100, 10)]

    protocnv = ProtoCNV(allregions, copynumber=1, ref_ploidy=2)
    protocnv.add_region(2, 0.9, 1.0, 0.1)
    protocnv.add_region(3, 1.0, 1.0, 0.1)
    protocnv.add_region(4, 0.93, 1.0, 0.1)
    protocnv.add_region(5, 0.95, 1.0, 0.1)
    protocnv.add_region(6, 0.6, 1.0, 0.1)

    cnv = protocnv.build_call(trim_lowqual_edges=True)
    assert cnv.targets == 4
    assert cnv.quality == pytest.approx(0.945)
    assert cnv.start == 30
    assert cnv.end == 65


def test_trim_lowqual_edges_right_multiple():
    allregions = [ ("X", i, i+5) for i in range(10, 100, 10)]

    protocnv = ProtoCNV(allregions, copynumber=1, ref_ploidy=2)
    protocnv.add_region(2, 0.9, 1.0, 0.1)
    protocnv.add_region(3, 1.0, 1.0, 0.1)
    protocnv.add_region(4, 0.93, 1.0, 0.1)
    protocnv.add_region(5, 0.95, 1.0, 0.1)
    protocnv.add_region(6, 0.6, 1.0, 0.1)
    protocnv.add_region(7, 0.65, 1.0, 0.1)
    protocnv.add_region(8, 0.5, 1.0, 0.1)

    cnv = protocnv.build_call(trim_lowqual_edges=True)
    assert cnv.targets == 4
    assert cnv.quality == pytest.approx(0.945)
    assert cnv.start == 30
    assert cnv.end == 65


def test_trim_lowqual_edges_thatcase():
    """
    A real-world test case
    :return:
    """
    allregions = [ ("X", i, i+5) for i in range(10, 100, 10)]

    protocnv = ProtoCNV(allregions, copynumber=1, ref_ploidy=2)
    protocnv.add_region(2, 0.9999, 1.0, 0.1)
    protocnv.add_region(3, 0.82, 1.0, 0.1)
    protocnv.add_region(4, 0.53, 1.0, 0.1)

    cnv = protocnv.build_call(trim_lowqual_edges=True)
    assert cnv.targets == 2
    assert cnv.quality == pytest.approx(0.90995)
    assert cnv.start == 30
    assert cnv.end == 45


@pytest.mark.parametrize("c, includes, excludes, expected", [
    ('a', None, ['a', 'b', 'c'], False),
    ('a', None, ['c', 'd', 'e'], True),
    ('a', ['a', 'b', 'c'], None, True),
    ('a', ['a', 'b', 'c'], ['c', 'd', 'e'], True),
    ('a', ['a', 'b', 'c'], [], True),
    ('a', ['b', 'c'], [], False),
    ('a', [], [], False),
    ('a', None, None, True),
])
def test_chrom_include_exclude(c, includes, excludes, expected):
    assert _chrom_include_exclude(c, includes, excludes) == expected


