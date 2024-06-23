#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df_transposed = df.transpose()

# Sort the transposed DataFrame in reverse chronological order
df_transposed_sorted = df_transposed.sort_index(axis=1, ascending=False)

print(df_transposed_sorted.tail(8))
