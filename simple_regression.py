from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)
    return m, b

x = np.linspace(0,5,14)
y = np.array([4,9,27,13,6,11,23,29,38,16,17,21,48,3])


m, b = np.polyfit(x, y,1)
y_pred = (m*x)+b

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('Line Plot')
plt.xlabel('Seedling growth in cm')
plt.ylabel('Days')
plt.show()
