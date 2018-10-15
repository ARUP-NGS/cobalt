
import pickle
import logging
from datetime import datetime
import sys
import numpy as np
from cobaltcnv import __version__

COBALT_MODEL_VERSION="1.1"

class CobaltModel(object):

    def __init__(self, chunk_data, params, mods, regions, mask=None, samplenames=None, cobaltargs=None):
        self.ver = COBALT_MODEL_VERSION
        self.birthday = datetime.now()
        self.chunk_data = chunk_data
        self.params = params
        self.regions = regions
        self.mask = mask
        self.mods = mods
        self.samplenames = samplenames
        self.args = cobaltargs

    def describe(self, outputfh=None):
        """
        Write a short descriptive message about this model to the given file handle, or stdout
        """
        if outputfh is None:
            outputfh = sys.stdout

        outputfh.write("Cobalt CNV Calling model v{}, created on {}\n".format(self.ver, self.birthday.strftime("%Y-%m-%d %H:%M:%S")))
        outputfh.write("Cobalt version used to create model: {}\n".format(__version__))
        outputfh.write("Cobalt model generation arguments:\n")
        for arg in self.args:
            outputfh.write("\t{}\n".format(arg))
        outputfh.write("\n")

        outputfh.write("Total targets: {}\n".format(len(self.regions)))
        if self.mask is None:
            outputfh.write("Masked targets:   0\n")
        else:
            outputfh.write("Masked targets:   {}\n".format(int(np.sum(1.0 - self.mask))))
        outputfh.write("\n")
        if self.samplenames is None:
            outputfh.write("Training samples: Unknown")
        else:
            outputfh.write("Training samples: {}\n".format(len(self.samplenames)))
            for name in self.samplenames:
                outputfh.write("\t{}\n".format(name))

def save_model(model, dest_path):
    """ Pickle a CNVModel and dump it to a file """
    logging.info("Saving model to {}".format(dest_path))
    with open(dest_path, "wb") as fh:
        pickle.dump(model, fh)

def load_model(path):
    """ Unpickle a CNVModel from a file and return it"""
    logging.info("Loading model from {}".format(path))
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    return model

