#!/usr/bin/env python
# encoding: UTF8

# ## Match the exact splitting in Vanessas files.
# 
# One problem when comparing the results is the random component in the splitting.
# While the catalogs should statistically be the same, we currently want to compare
# object by object.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(cat):
    """Random split of the input cataloge."""

    # This part here follow what Vanessa was doing. We keep the same structure.
    print('# Gal with spec-z')
    dummy = np.ones(len(cat))
    X_train, X_test, dummy_train, _ = train_test_split(cat, dummy, test_size=0.3, random_state=42)
    X_train, X_val, _, _ = train_test_split(X_train, dummy_train, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_test

def test_overlap(ref_id_train, ref_id_val, ref_id_test):
    """Check for overlap between different sets."""
    
    S1 = set(ref_id_train.tolist())
    S2 = set(ref_id_val.tolist())
    S3 = set(ref_id_test.tolist())
    
    Nunion = len(S1 | S2 | S3)
    Nsum = len(S1) + len(S2) + len(S3)
    
    assert Nunion == Nsum, 'There are overlaps between train, val and test sets.'

def load_indexes(split_path):
    """Load the reference IDs in different splits."""
    
    def get_ref_id(fname):
        X = pd.read_csv(split_path / fname)
        ref_id = X.ref_id.astype(int).values
        
        return ref_id
    
    ref_id_train = get_ref_id('train_E0_Complemento.csv')
    ref_id_val = get_ref_id('val_E0_Complemento.csv')
    ref_id_test = get_ref_id('test_E0_Complemento.csv')

    # Should *never* trigger. But at least then we are sure.
    test_overlap(ref_id_train, ref_id_val, ref_id_test)
    
    return ref_id_train, ref_id_val, ref_id_test

def split_by_existing(cat, split_path):
    """Split catalogue in the same way as existing catalogues."""
    
    # Loads the existing split.
    ref_id_train, ref_id_val, ref_id_test = load_indexes(split_path)
    
    cat = cat.set_index('ref_id')
    
    def subset(idx):
        return cat.loc[idx].reset_index()
    
    X_train = subset(ref_id_train)
    X_val = subset(ref_id_val)
    X_test = subset(ref_id_test)
    
    return X_train, X_val, X_test
