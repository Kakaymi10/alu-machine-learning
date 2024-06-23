#!/usr/bin/env python3
import numpy as np
import pandas as pd

dic = {'A':[0.0, 'one'], 'B':[0.5, 'two'], 'C':[1.0, 'three'], 'D':[1.5, 'four']}
df = pd.DataFrame.from_dict(dic, orient='index', columns=['First', 'Second'])
print(df)
