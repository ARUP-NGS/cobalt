
from cobaltcnv import training, model, prediction, util, transform, qc
import numpy as np
import pytest
import os
import unittest.mock as mock

def test_model_train_save_load(tmpdir, bknd_depths_100):
    """ Can we train and save, and reload a tiny model """
    modelpath = os.path.join(tmpdir, "testmodel.model")
    training.MIN_CHUNK_SIZE = 1  # Otherwise we'll fail a check for sane input...
    training.train(bknd_depths_100,
                   modelpath,
                   use_depth_mask=True,
                   var_cutoff=0.95,
                   chunk_size=37,
                   min_depth=20,
                   low_depth_trim_frac=0.01,
                   high_depth_trim_frac=0.01,
                   high_cv_trim_frac=0.01,
                   cluster_width=4)

    mod = model.load_model(modelpath)
    assert isinstance(mod, model.CobaltModel)
    assert mod.ver == model.COBALT_MODEL_VERSION
    assert len(mod.mask) == 99
    assert len(mod.regions) == 99
    assert len(mod.params) == 5

def test_model_train_save_load_no_mask(tmpdir, bknd_depths_100):
    """ Can we train and save, and reload a tiny model """
    modelpath = os.path.join(tmpdir, "testmodel_nomask.model")
    training.MIN_CHUNK_SIZE = 1  # Otherwise we'll fail a check for sane input...
    training.train(bknd_depths_100,
                   modelpath,
                   use_depth_mask=False,
                   var_cutoff=0.95,
                   chunk_size=37,
                   min_depth=20,
                   low_depth_trim_frac=0.01,
                   high_depth_trim_frac=0.01,
                   high_cv_trim_frac=0.01,
                   cluster_width=4)

    mod = model.load_model(modelpath)
    assert isinstance(mod, model.CobaltModel)
    assert mod.ver == model.COBALT_MODEL_VERSION
    assert len(mod.mask) == 99
    assert all(m for m in mod.mask)
    assert len(mod.regions) == 99
    assert len(mod.params) == 5
    assert all(len(p) == 99 for p in mod.params)
    assert mod.comps is not None
    assert mod.comps.shape == (36, 2)
    assert mod.directions is not None
    assert mod.directions.shape == (99, 2)

def test_model_load_predict(simple_model, sample_depths_100):
    """ Load a small model and predict CNVs from it """

    cmodel = model.load_model(simple_model)
    sample_depths, _ = util.read_data_bed(sample_depths_100)
    depths = np.matrix(sample_depths[:, 0]).reshape((sample_depths.shape[0], 1))
    cnvs = prediction.call_cnvs(cmodel, depths, 0.05, 0.05, assume_female=None, genome=util.ReferenceGenomes.B37)
    assert len(cnvs)==2
    assert cnvs[0].copynum == 1
    assert cnvs[0].start == 880354
    assert cnvs[0].end == 880593

    assert cnvs[1].copynum == 3
    assert cnvs[1].start == 889156
    assert cnvs[1].end == 889276

def test_model_load_qc(tmpdir, bknd_depths_100):
    modelpath = os.path.join(tmpdir, "testmodel_qc.model")
    training.MIN_CHUNK_SIZE = 1  # Otherwise we'll fail a check for sane input...
    training.train(bknd_depths_100,
                   modelpath,
                   use_depth_mask=True,
                   var_cutoff=0.95,
                   chunk_size=37,
                   min_depth=20,
                   low_depth_trim_frac=0.01,
                   high_depth_trim_frac=0.01,
                   high_cv_trim_frac=0.01,
                   cluster_width=4)

    mod = model.load_model(modelpath)
    sample_depths, _ = util.read_data_bed(bknd_depths_100)
    depths = np.matrix(sample_depths[:, 0]).reshape((sample_depths.shape[0], 1))
    depths = depths[mod.mask, :]
    prepped = transform.prep_data(depths)
    transformed_depths = transform.transform_by_genchunks(prepped, mod)

    point = qc.project_sample(mod, transformed_depths)
    assert len(point) == 2

    mean_dist, score = qc.compute_mean_dist(mod, transformed_depths)
    assert mean_dist == pytest.approx(5.3540836)

def test_compute_background_pca():
    params = [ # Only need to populate index 2
        [], [],
        [(0.0, 1.0, 0.01),
         (0.0, 0.5, 0.02),
         (0.0, 2.0, 0.01),
         (0.0, 1.0, 0.025),
         (0.0, 0.0, 0.01)],
        [], [],
    ]

    depths = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [2.0, 1.0, 2.0, 1.0, 2.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [-2.0, -2.0, -2.0, -2.0, -2.0],
        [-2.0, 2.0, -2.0, 2.0, -2.0],
    ])

    comps, vectors = qc.compute_background_pca(params, depths)
    assert comps.shape == (6, 2)
    assert vectors.shape == (5, 2)

    # Assert that the directions are orthogonal? We don't really want to test sklearn internals
    dotprod = sum( vectors[i,0]*vectors[i,1] for i in range(vectors.shape[0]))
    assert dotprod == pytest.approx(0.0)


def test_project_sample():
    params = [ # Only need to populate index 2
        [], [],
        [(0.0, 1.0, 0.01),
         (0.0, 0.5, 0.02),
         (0.0, 2.0, 0.01),
         (0.0, 1.0, 0.025),
         (0.0, 0.0, 0.01)],
        [], [],
    ]

    directions = np.array([
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    ]).T

    cmodel = mock.Mock(params=params, directions=directions)
    sample_depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    proj = qc.project_sample(cmodel, sample_depths)
    assert proj.shape == (2,)


@pytest.mark.parametrize('numregions, chunksize', [
    (117, 8),
    (100, 10),
    (5, 5),
    (17, 3),
    (28, 4),
    (10001, 1000),
])
def test_chunk_gen(numregions, chunksize):
    """
    Test to make sure gen_chunk_indices always includes every index exactly once
    """
    training.MIN_CHUNK_SIZE = 1 # Otherwise we'll fail a check for sane input...
    regions = list(range(numregions))
    indices = training.gen_chunk_indices(regions, chunksize, skip_chunk_size_check=True)

    # Now combine all indices back into a big set
    allindices = []
    for i in indices:
        allindices.extend(i)
    assert len(allindices) == numregions
    assert set(allindices) == set(regions)

# @pytest.mark.parametrize('numregions, chunksize, expectednum', [
#     (100, 20, 5),
#     # (100, 10),
# ])
# def test_chunk_gen_correct_chunk_num(numregions, chunksize, expectednum):
#     """
#     Test to make sure gen_chunk_indices generates the expected number of chunks
#     """
#     training.MIN_CHUNK_SIZE = 1 # Otherwise we'll fail a check for sane input...
#     regions = list(range(numregions))
#     indices = training.gen_chunk_indices(regions, chunksize)
#
#     # Now combine all indices back into a big set
#     allchunks = list(indices)
#     assert len(allchunks) == expectednum