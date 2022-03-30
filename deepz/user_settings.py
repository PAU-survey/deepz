#!/usr/bin/env python
# encoding: UTF8

# Created at 2021-05-27T16:48:32.642811 by corral 0.3


# =============================================================================
# DOCS
# =============================================================================

"""Global configuration for deepz."""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATIONS
# =============================================================================

#: Path where the settings.py lives
PATH = os.path.abspath(os.path.dirname(__file__))

# Path to the folder of the input files
path = os.path.join('../../input')

cosmos_path = os.path.join(path, 'cosmos.csv')
galcat_path = os.path.join(path, 'coadd_v8.h5')
indexp_path = os.path.join(path, 'fa_v8.pq')