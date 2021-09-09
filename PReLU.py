import numpy as np
import matplotlib.pyplot as plt
import sys


# PReLU:
def PReLU(x,a=.01):
    x1=[]
    for i in x:
        if i<0:
            x1.append(i*a)
        else:
            x1.append(i)
    return x1
      
# PReLU derivative:
def PReLU_deriv(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(i)
        else:
            x1.append(1)
    return x1


x = np.linspace(-10, 10)

plt.plot(x, PReLU(x))
plt.axis('tight')
plt.title('Activation Function :PReLU')
plt.show()


