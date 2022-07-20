#!/usr/bin/env python
# encoding: UTF8

import glob
import pdb
from setuptools import setup, find_packages

# Same author and maintainer.
name = 'Martin B. Eriksen'
email = 'eriksen@pic.es'

setup(
    name = 'deepz',
    version = '1',
    packages = find_packages(),

    install_requires = [
        'fire',
        'matplotlib',
        'pandas',
        'pytables',
        'torch',
        'xarray'
    ],
    author = name,
    author_email = email,
    license = 'GPLv3',
    maintainer = name,
    maintainer_email = email,
#    scripts = ['bcnz/bin/run_bcnz.py'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
