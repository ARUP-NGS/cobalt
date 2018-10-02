

import numpy as np
from cobaltcnv import transform

def test_iterate_transform_zscore():

    # 2x6
    components = np.matrix([
        [1, 2, 3, 0.1, 0.2, 0.3],
        [1.1, 2.2, 0.3, 0.4, 1.6, 1.3],
    ]).T

    data = np.matrix([
        [4,6,2,3,4,6]
    ], dtype=np.float64)

    comp_mat = components.dot(components.T)

    a = transform.transform_raw(data, comp_mat=comp_mat, zscores=True)

    b = transform.transform_raw_iterative(data, components, zscores=True)

    assert a.shape == b.shape
    for x,y in zip(a.getA1(), b.getA1()):
        assert abs(x - y) < 1e-6

def test_iterate_transform():

    # 2x6
    components = np.matrix([
        [1, 2, 3, 0.1, 0.2, 0.3],
        [1.1, 2.2, 0.3, 0.4, 1.6, 1.3],
    ]).T

    data = np.matrix([
        [4,6,2,3,4,6]
    ], dtype=np.float64)

    comp_mat = components.dot(components.T)

    a = transform.transform_raw(data, comp_mat=comp_mat, zscores=False)

    b = transform.transform_raw_iterative(data, components, zscores=False)

    assert a.shape == b.shape
    for x,y in zip(a.getA1(), b.getA1()):
        assert abs(x - y) < 1e-6

def test_transform_single_site():
    components = np.matrix([
        [1, 2, 3, 0.1, 0.2, 0.3],
        [1.1, 2.2, 0.3, 0.4, 1.6, 1.3],
    ]).T

    data = np.matrix([
        [4, 6, 2, 3, 4, 6]
    ], dtype=np.float64)

    comp_mat = components.dot(components.T)

    sites = [0,1,2,3,4,5]
    for site in sites:
        orig_scores = transform.transform_raw(data, comp_mat=comp_mat, zscores=False)
        sitevals = transform.transform_single_site(data, components, orig_scores=orig_scores, site=site, zscores=False)
        for a,b in zip(orig_scores[:, site], sitevals):
            assert abs(a - b) < 1e-6




