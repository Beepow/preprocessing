import numpy as np

t = 7


x = 255 * 2 * (1/(1+ np.exp(- t)) - 0.5)
print(x)

y = 1/(1 + np.exp(-t))
print(y)

z = 255*(1 / (1+ np.exp(-t/2)))
print(z)