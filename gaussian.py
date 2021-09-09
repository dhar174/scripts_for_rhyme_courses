import numpy as np
import matplotlib.pyplot as plt
import sys

def gaussian(x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2 * x * np.exp(-x**2)

x = np.linspace(-10, 10)
plt.plot(x, gaussian(x))
plt.axis('tight')
plt.title('Activation Function :Gaussian')
plt.show()
