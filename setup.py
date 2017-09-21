# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import subprocess

install_requires = ['numpy', 'pandas', 'scikit-learn', 'scipy']
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
    # Fetch version from git tags, and write to version.py.
    # Also, when git is not available (PyPi package), use stored version.py.
    version_py = os.path.join(os.path.dirname(__file__), 'version.py')

    try:
        version_git = subprocess.check_output(["git", "describe"]).rstrip().decode(encoding='utf-8')
    except:
        with open(version_py, 'r') as fh:
            version_git = open(version_py).read().strip().split('=')[-1].replace('"', '')

    version_msg = "# Do not edit this file, pipeline versioning is governed by git tags"
    with open(version_py, 'w') as fh:
        fh.write(version_msg + os.linesep + "__version__=" + version_git)

    return version_git

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
        scripts=['bin/cobalt'],
    )
