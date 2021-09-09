import itertools
import pandas as pd
import numpy as np

import os


import collections
import pathlib
import re
import string




in_csv = 'train_data.csv'

#get the number of lines of the csv file to be read

##number_lines = sum(1 for row in (open(in_csv,encoding='utf-8',mode='r')))

rowsize = 500




   
df = pd.DataFrame(pd.read_csv(in_csv),columns=['id','title','text','label'])

#size of rows of data to write to the csv, 

#you can change the row size according to your need

a=0
for row in df.iterrows():
    for col in row:
        if type(col)==int:
            a=col
        else:
            if col.label==0:
                df.drop(index = int(col.name),inplace=True)

df.reset_index(drop=True,inplace=True)
print(df.count())

number_lines=len(df)
print(number_lines)
print(df.count())
#start looping through data writing it to a new file for each set

for i in range(1,number_lines,rowsize):
    
##    print(df.columns)
    print(i)
##    

    #csv to write data to a new file with indexed name. input_1.csv etc.

    out_csv = os.getcwd()+'\\train\\true\input' + str(i) + '.txt'
    
   
    df2 = pd.DataFrame(df.loc[0:rowsize],columns=['id','title','text','label'])
    
##    print(df2.count())
##    print(df2)
    df2.to_csv(out_csv,

          index=False,

          header=['id','title','text','label'],

          mode='a',#append data to csv file

          chunksize=rowsize)#size of data to append for each loop
##    print(df2.index)
    for a in df2.iterrows():
        for ii in a:
            if type(ii)!=int:
                pass
            else:
##                print(df.loc[ii])
                df.drop(index=ii,inplace=True)
    df.reset_index(drop=True,inplace=True)

    print(df.count())




print(df.count())
##train_data=pd.read_csv('train_data.csv', index_col=None)
##test_data=pd.read_csv('test_data.csv',index_col=None)
##
##lines_per_file = 500
##smallfile = None
##with open('test_data.csv',mode='r',encoding='utf-8') as bigfile:
##    for lineno, line in enumerate(bigfile):
##        if lineno % lines_per_file == 0:
##            if smallfile:
##                smallfile.close()
##            small_filename = 'small_file_{}.txt'.format(lineno + lines_per_file)
##            smallfile = open(small_filename, "w")
##        smallfile.write(line)
##    if smallfile:
##        smallfile.close()
