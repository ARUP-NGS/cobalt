# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import subprocess

install_requires = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'pysam']
tests_require    = ['pytest', 'coverage']

classifiers = """
Development Status :: 3 - Beta
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows :: Windows NT/2000
Operating System :: OS Independent
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Bioinformatics
"""


def version_tag():
    version = [line for line in open(os.path.split(__file__)[0] + "/cobaltcnv/__init__.py").readlines() if "__version__" in line]
    if not version:
        raise ValueError("Could not find __version__ in cobaltcnv/__init__.py")
    version = version[0].strip().replace('__version__=', '').replace('"', '')
    return version
if __name__ == '__main__':
    setup(
        name = 'cobaltcnv',
        version=version_tag(),
        description = 'Cobalt CNV caller',
        author = "Brendan O'Fallon",
        maintainer =  "Brendan O'Fallon",
        author_email = 'brendan.ofallon@aruplab.com',
        classifiers = classifiers,
        test_suite = 'nose.collector',
        tests_require = tests_require,
        packages = find_packages(),
        install_requires = install_requires,
        scripts=['bin/cobalt', 'bin/covcounter'],
    )
