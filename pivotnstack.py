import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.cluster import DBSCAN




data = pd.read_csv('melt.csv')


data = pd.melt(data, id_vars='country', value_vars=['2010', '2011', '2012'])

print(data)
