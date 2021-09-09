import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy import stats


np.random.seed(1)


data = np.random.randn(50000,2)  * 20 + 20


clf = IsolationForest(max_samples=100, random_state = 1, contamination= 'auto')
preds = clf.fit_predict(data)
print(preds)
z = np.abs(stats.zscore(data))
print(z)
