import pandas as pd


train_data=pd.read_csv('train_data.csv', index_col=None)
test_data=pd.read_csv('test_data.csv', index_col=None)

test_data.dropna(inplace=True)
train_data.dropna(inplace=True)


train_data.to_csv('train_data2.csv',na_rep = 'NaN',index=False)
test_data.to_csv('test_data2.csv',na_rep = 'NaN',index=False)

train_data=pd.read_csv('train_data2.csv', index_col=None)
test_data=pd.read_csv('test_data2.csv', index_col=None)

test_data.dropna(inplace=True)
train_data.dropna(inplace=True)


train_data.to_csv('train_data2.csv',na_rep = 'NaN',index=False)
test_data.to_csv('test_data2.csv',na_rep = 'NaN',index=False)

