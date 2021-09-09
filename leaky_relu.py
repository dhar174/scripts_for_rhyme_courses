import numpy as np
import matplotlib.pyplot as plt
import sys


# Leaky ReLU:
def leaky(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(i*0.01)
        else:
            x1.append(i)
    return x1
      
# Leaky ReLU derivative:
def leaky_deriv(x):
  return 0.01 if x < 0 else 1


x = np.linspace(-10, 10)

plt.plot(x, leaky(x))
plt.axis('tight')
plt.title('Activation Function :Leaky RELU')
plt.show()


print("Leaky ReLu of x: " + str(leaky(x)))
print("Leaky ReLu of x if x = -1.0: " + str(leaky(-1.0)))
print("Leaky ReLu derivative of x: " + str(leaky_deriv(x)))
print("Leaky ReLu derivative of x if x = -1.0: " + str(leaky_deriv(-1.0)))
