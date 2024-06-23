#1/usr/bin/env python3
import numpy as np
import pandas as pd

def from_file(filename, delimiter):
    df = pd.read_csv(filename, delimiter=delimiter)
    return pd.DataFram(df)
