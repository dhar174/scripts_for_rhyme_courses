import numpy as np
import matplotlib.pyplot as plt
import sys

def swish_function(x):
    return x/(1-np.exp(-x))

print(swish_function(-67), swish_function(4))
