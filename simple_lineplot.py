import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,25)
y = np.random.rand(len(x))*2
print(x)
sns.regplot(x=x, y=y * x,)
plt.plot(x, y * x)
plt.title('Line Plot')
plt.xlabel('Seedling growth in cm')
plt.ylabel('Days')
plt.show()
