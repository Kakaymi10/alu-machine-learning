#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(['Weighted_Price'])
df = df['Close'].fillna(method='ffill')
df = df[['High', 'Low', 'Open']].fillna(value=df['Close'], axis=0)
df = df[['Volume_(BTC_)', 'Volume_(Currency)']].fillna(value=0)
print(df.head())
print(df.tail())
