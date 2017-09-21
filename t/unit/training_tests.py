
from cobaltcnv import training, model
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




