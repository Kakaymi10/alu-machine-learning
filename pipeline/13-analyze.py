#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file  # Assuming the function to load DataFrame is correctly implemented

# Load data from file into DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Calculate descriptive statistics for all columns except 'Timestamp'
stats = df.drop(columns=['Timestamp']).describe()

# Print the descriptive statistics
print(stats)
