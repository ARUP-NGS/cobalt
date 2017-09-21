
from cobaltcnv import training, model, prediction, util
import numpy as np
import os

def test_model_train_save_load(tmpdir, bknd_depths_100):
    """ Can we train and save, and reload a tiny model """
    modelpath = os.path.join(tmpdir, "testmodel.model")

    training.train(bknd_depths_100,
                   modelpath,
                   use_depth_mask=True,
                   num_components=6,
                   max_cv=1.0,
                   chunk_size=100)

    mod = model.load_model(modelpath)
    assert isinstance(mod, model.CobaltModel)




def test_model_load_predict(simple_model, sample_depths_100):
    """ Load a small model and predict CNVs from it """

    cmodel = model.load_model(simple_model)
    sample_depths, _ = util.read_data_bed(sample_depths_100)
    depths = np.matrix(sample_depths[:, 0]).reshape((sample_depths.shape[0], 1))
    cnvs = prediction.call_cnvs(cmodel, depths, 0.05, 0.05, assume_female=None)
    assert len(cnvs)==2
    assert cnvs[0].copynum == 1
    assert cnvs[0].start == 880354
    assert cnvs[0].end == 880593

    assert cnvs[1].copynum == 3
    assert cnvs[1].start == 889156
    assert cnvs[1].end == 889276
