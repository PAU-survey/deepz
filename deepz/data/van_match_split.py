#!/usr/bin/env python
# encoding: UTF8

# TO BE DELETED. ONLY INCLUDED HERE FOR COMPARING WITH VANESSAS RESULTS.

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

def load_indexes(split_fmt):
    """Load the reference IDs in different splits.
       split_fmt: Path to catalogue containing the split.
    """
    
    def get_ref_id(dset):
        path_cat = Path(str(split_fmt).format(dset=dset))
        X = pd.read_csv(path_cat)
        ref_id = X.ref_id.astype(int).values
        
        return ref_id
    
    ref_id_train = get_ref_id('train')
    ref_id_val = get_ref_id('val')
    ref_id_test = get_ref_id('test')

    # Should *never* trigger. But at least then we are sure.
    test_overlap(ref_id_train, ref_id_val, ref_id_test)
    
    return ref_id_train, ref_id_val, ref_id_test

def split_by_existing(cat, split_fmt):
    """Split catalogue in the same way as existing catalogues."""
    
    # Loads the existing split.
    ref_id_train, ref_id_val, ref_id_test = load_indexes(split_fmt)
    
    cat = cat.set_index('ref_id')
    
    def subset(idx):
        return cat.loc[idx].reset_index()
    
    X_train = subset(ref_id_train)
    X_val = subset(ref_id_val)
    X_test = subset(ref_id_test)
    
    return X_train, X_val, X_test

def retarded_split(df_train, df_val, df_train_Comple, df_val_Comple, xa, xb):
    """Reproducing a wrong split in Vanessas catalogue."""
    
    # How things are done here is extremely dangerous and ended up going wrong.
    columns = list(set(df_train.columns) - set(['ra', 'dec', 'zb_bb', 'zb_bcnz', 'catalog']))
    df_complemento = pd.concat([df_train_Comple[columns], df_val_Comple[columns]])
    df_E1 = pd.concat([df_train[columns], df_val[columns]])
    
    df_E1_NaN = df_complemento[~df_complemento['ref_id'].isin(df_E1.ref_id.values)]
    df_E1_NaN_train = pd.concat([df_train[columns], df_E1_NaN[columns].iloc[0:xa]])
    df_E1_NaN_val = pd.concat([df_val[columns], df_E1_NaN[columns].iloc[xa:xb]])

    Strain = set(df_E1_NaN_train.ref_id)
    Sval = set(df_E1_NaN_val.ref_id)
    assert not len(Strain & Sval), 'Overlap between training and validation sample.'
    
    return df_E1_NaN_train, df_E1_NaN_val
