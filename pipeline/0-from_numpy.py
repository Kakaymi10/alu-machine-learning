#!/usr/bin/emv python3
import pandas as pd
import numpy as np
import string


def from_numpy(array):
    columns_title = list(string.ascii_uppercase)
    n = len(array[0])
    df = pd.DataFrame(array, columns=[columns_title[i] for i in range(n)])
    return df
    
