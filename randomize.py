
import pandas as pd
import random
import os

n = 400000000 #number of records in file
s = 50000000 #desired sample size
skip = sorted(random.sample(range(n),n-s))



df=pd.read_csv('train_datax.csv',skiprows=skip,low_memory=False)

out_csv = os.getcwd()+'\\train_data.csv' 



df.to_csv(out_csv,

          index=False,

          header=['id','title','text','label'],

          mode='a')#size of data to append for each l
