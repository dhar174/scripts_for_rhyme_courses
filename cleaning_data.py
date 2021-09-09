import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.cluster import DBSCAN




my_data = pd.read_csv('olympics.csv', header =1)

pd.set_option("display.max_rows", None, "display.max_columns", None)

 
new_names =  {'Unnamed: 0': 'Country',
               '? Summer': 'Summer Olympics',
               '01 !': 'Gold',
               '02 !': 'Silver',
               '03 !': 'Bronze',
               '? Winter': 'Winter Olympics',
               '01 !.1': 'Gold.1',
               '02 !.1': 'Silver.1',
               '03 !.1': 'Bronze.1',
               '? Games': '# Games',
               '01 !.2': 'Gold.2',
               '02 !.2': 'Silver.2',
               '03 !.2': 'Bronze.2'}

my_data.rename(columns=new_names, inplace=True)

##print(my_data.head(3))
##
##column_names = my_data.columns
##print(my_data['Silver'].isnull())

#ctry= my_data['Country']
#print(ctry)
##for name in ctry:
##    extra=False
##    if('!' in name):
##        #print(name)
##        extra=True
##    b =np.where(extra, name.replace('!', ''), print(''))
##    if(b):
##        #print(b)
##        my_data[my_data==name] = b
##        print(name)
old = my_data
new = np.where(my_data.Country.str.contains('!'),my_data.Country.str.replace('!', ''), my_data.Country.str.replace('!', ''))
#for a in new:
new = pd.DataFrame(new)
new = new.dropna()
#new.rename()
print(new)
#my_data.index = list(my_data.index)
##new.stack
##my_data.stack
print(new.index,my_data.index)
#new.reshape(1,1)
new.columns =["Country"]

#new.set_index('Country').join(my_data.set_index('Country'))
#new.Country.astype(str)
#my_data.Country.astype(str)

##new.reset_index(inplace=True)
##my_data.reset_index(inplace=True)
##new.index(dtype=str)
##my_data.index(dtype=str)


print(type(new))



my_data.Country=new.Country



#new = my_data.merge(new,my_data,left_on='Country',right_index=True)
#new=new.stack()
##print('new2: '+new)
print(new.shape)
count=0
##for index, row in new.iterrows():
##    count+=1
##    print(new.iloc[int(index)])
##    
##    for a,b in my_data.iterrows():
##        b=np.where(row is "Country",np.where(new.iloc[index] in my_data.Country, b.str.replace(my_data.Country.iloc[a],new.loci[index]),print('')),print('')) 
my_data.fillna(42,inplace=True)
my_data.drop_duplicates(inplace=True)
my_data=my_data.drop(index=147)
print(my_data.describe())
print(my_data)
#new = np.array([a for a in new if a])
#new.reshape(1,147)
#print(new)
#poop = np.concatenate((my_data.Country, new), axis=0, out=None)
#poop = poop.reshape((294,1))
#print('poop: '+poop)

Q1 = my_data.quantile(0.25)
Q3 = my_data.quantile(0.75)
IQR = Q3 - Q1
##print(IQR)
##print(my_data.skew())
##print(my_data < (Q1 - 1.5 * IQR)),(my_data > (Q3 + 1.5 * IQR))

productsByState = my_data.pivot_table(index='Country', columns='Gold')

print(productsByState)
