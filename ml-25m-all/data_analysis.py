import numpy as np
import pandas as pd

path = '2015top15.csv'

data = pd.read_csv(path,header=0,sep=',')
print(data)
print(data.groupby(by='userId').size())#.max())