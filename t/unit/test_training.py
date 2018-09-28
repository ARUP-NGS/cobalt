
from cobaltcnv import training, model, prediction, util
import numpy as np
import os
import pytest

def test_model_train_save_load(tmpdir, bknd_depths_100):
    """ Can we train and save, and reload a tiny model """
    modelpath = os.path.join(tmpdir, "testmodel.model")
    training.MIN_CHUNK_SIZE = 1  # Otherwise we'll fail a check for sane input...
    training.train(bknd_depths_100,
                   modelpath,
                   use_depth_mask=True,
                   var_cutoff=0.95,
                   max_cv=1.0,
                   chunk_size=37,
                   min_depth=20)

    mod = model.load_model(modelpath)
    assert isinstance(mod, model.CobaltModel)


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
    indices = training.gen_chunk_indices(regions, chunksize)

    # Now combine all indices back into a big set
    allindices = []
    for i in indices:
        allindices.extend(i)
    assert len(allindices) == numregions
    assert set(allindices) == set(regions)