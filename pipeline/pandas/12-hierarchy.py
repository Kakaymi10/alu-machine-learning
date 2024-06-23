#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file  # Assuming the function to load DataFrame is correctly implemented

# Load DataFrames from files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to-2020-04-22.csv', ',')

# Index DataFrames on 'Timestamp'
df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

# Concatenate df2 and df1 from timestamps 1417411980 to 1417417980, inclusive
df = pd.concat([df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]], keys=['bitstamp', 'coinbase'])

# Rearrange MultiIndex levels so that 'Timestamp' is the first level
df = df.swaplevel().sort_index()

# Print the concatenated and rearranged DataFrame
print(df)
