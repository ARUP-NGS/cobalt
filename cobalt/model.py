
import pickle
import logging

class CobaltModel(object):

    def __init__(self, chunk_data, params, mods, regions, rowstats=None, prep_mode=None, mask=None, opt_sites=None):
        self.chunk_data = chunk_data
        self.params = params
        self.regions = regions
        self.rowstats = rowstats
        self.prep_mode = prep_mode
        self.mask = mask
        self.mods = mods
        self.opt_sites = opt_sites


def save_model(model, dest_path):
    """ Pickle a CNVModel and dump it to a file """
    logging.info("Saving model to {}".format(dest_path))
    with open(dest_path, "w") as fh:
        pickle.dump(model, fh)

def load_model(path):
    """ Unpickle a CNVModel from a file and return it"""
    logging.info("Loading model from {}".format(path))
    with open(path, "r") as fh:
        model = pickle.load(fh)
    return model

