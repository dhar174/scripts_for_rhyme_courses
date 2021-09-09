import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def linear(x):
    ''' y = f(x) It returns the input as it is'''
    return x

def linear_deriv(x):
  return 1



x = np.linspace(-10, 10)
plt.plot(x, linear(x))
plt.axis('tight')
plt.title('Activation Function :Linear')
plt.show()

print("Identity should return x: " + str(linear(x)))
print("Identity derivative should return 1 always: " + str(linear_deriv(0)))
