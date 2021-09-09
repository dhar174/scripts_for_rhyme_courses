import numpy as np
import matplotlib.pyplot as plt
import sys


# ELU:
def ELU(x,a=.01):
    x1=[]
    for i in x:
        if i<=0:
            x1.append(float(np.exp(i)-1))
        else:
            x1.append(i)
    return x1
      
# ELU derivative:
def ELU_deriv(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(i)
        else:
            x1.append(1)
    return x1


x = np.linspace(-10, 10)

plt.plot(x, ELU(x))
plt.axis('tight')
plt.title('Activation Function :ELU')
plt.show()
