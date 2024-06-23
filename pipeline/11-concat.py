#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file  # Assuming the function to load DataFrame is correctly implemented

# Load DataFrames from files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to-2020-04-22.csv', ',')

# Index DataFrames on 'Timestamp'
df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

# Concatenate df2 onto the top of df1 up to and including timestamp 1417411920
df = pd.concat([df2.loc[:1417411920], df1], keys=['bitstamp', 'coinbase'])

# Print the concatenated DataFrame
print(df)
