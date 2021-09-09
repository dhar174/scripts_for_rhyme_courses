import csv
import pandas
df = pandas.read_csv('fake_or_real_news.csv',index_col='id')


df.loc[df['label'] == 'FAKE','label'] = 0
df.loc[df['label'] == 'REAL','label'] = 1

df.to_csv('fake_or_real_mod.csv')

