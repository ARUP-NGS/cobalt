
import numpy as np
from cobaltcnv import util

def almost_equal(a,b, tol=1e-6):
    return abs(a-b) < tol


def test_calc_cv():
    m = np.array([
        [0.1, 0.2, 17.2],
        [0.2, 0.4, 0.1],
        [0.6, 1.2, 33.1],
        [1.1, 2.2, 1.0],
        [0.12, 0.24, 0.2],
    ])

    cv = util.calc_depth_cv(m)
    truevals = [0.90457876, 0.90457876, 1.2708125532]
    for a,b in zip(cv, truevals):
        assert almost_equal(a, b)

def test_remove_samples():
    m = np.array([
        [0.1, 0.2, 0.2, 17.2],
        [0.2, 0.4, 0.9, 0.1],
        [0.6, 100.2, 2.6, 33.1],
        [1.1, 2.2, 1.3, 1.0],
        [0.12, 0.24, 0.4, 0.2],
    ])

    max_cv = 1.0
    m2 = util.remove_samples_max_cv(m, max_cv=max_cv)
    assert m2.shape == (5,2)
    assert all(util.calc_depth_cv(m2) < max_cv)


def test_find_index():
    regions = [
        ("1", 10, 20),
        ("1", 30, 50),
        ("2", 5, 10)
    ]

    a = util.find_site_index_for_region("1", 5, 15, regions)
    assert a == [0]

    b = util.find_site_index_for_region("1", 5, 25, regions)
    assert b == [0]

    c = util.find_site_index_for_region("1", 15, 25, regions)
    assert c == [0]

    d = util.find_site_index_for_region("1", 15, 55, regions)
    assert d == [0,1]

    e = util.find_site_index_for_region("1", 25, 55, regions)
    assert e == [1]

    f = util.find_site_index_for_region("1", 55, 65, regions)
    assert f == []

    g = util.find_site_index_for_region("1", 5, 8, regions)
    assert g == []


def test_xratio():
    regions = [
        ("1", 10, 20),
        ("1", 30, 50),
        ("X", 5, 10),
        ("X", 50, 100),
        ("X", 250, 350),
        ("Y", 100, 200)
    ]

    depths = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 25.0])

    xratio = util.x_depth_ratio(regions, depths, genome=util.ReferenceGenomes.B37, min_num_x_regions=3)
    assert xratio == 1.0

    depths = np.array([10.0, 10.0, 5.0, 5.0, 5.0, 17.23])
    xratio = util.x_depth_ratio(regions, depths, genome=util.ReferenceGenomes.B37, min_num_x_regions=3)
    assert xratio == 0.5

def test_interval_overlap():

    assert util.intervals_overlap(
        ("1", 10, 20),
        ("1", 15, 20))

    assert util.intervals_overlap(
        ("1", 10, 20),
        ("1", 12, 17))

    assert util.intervals_overlap(
        ("1", 5, 15),
        ("1", 12, 17))

    assert util.intervals_overlap(
        ("1", 5, 15),
        ("1", 14, 17))

    assert not util.intervals_overlap(
        ("1", 5, 10),
        ("1", 12, 17))

    assert not util.intervals_overlap(
        ("1", 5, 10),
        ("2", 8, 17))
