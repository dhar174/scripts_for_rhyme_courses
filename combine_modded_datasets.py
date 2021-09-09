import csv
import pandas
from os import chdir
from glob import glob
import pandas as pd
import numpy as np

df1 = pd.read_csv('train_data1.csv',index_col='id')
df2 = pd.read_csv('train_data2.csv',index_col='id')

df = pd.concat([df1,df2])

df.dropna()

df.to_csv('train_data.csv')
