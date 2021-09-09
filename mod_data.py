import csv
import pandas
df = pandas.read_csv('fake_or_real_news_csv.csv',index_col='id')


df.loc[df['label'] == 'FAKE'] = 0
df.loc[df['label'] == 'REAL'] = 1

df.to_csv('fake_or_real_mod.csv')

