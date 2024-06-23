#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file  # Assuming the function to load DataFrame is correctly implemented

# Load data from file into DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename 'Timestamp' column to 'Datetime'
df = df.rename(columns={'Timestamp': 'Datetime'})

# Convert 'Datetime' column values to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Display only 'Datetime' and 'Close' columns
df = df[['Datetime', 'Close']]

# Print the last few rows to verify
print(df.tail())
