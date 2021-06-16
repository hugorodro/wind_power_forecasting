import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math 

# instantiate variables
x, y, j = [], [], []

# add values to each array
for i in range(1,365, 37):
    x.append(math.cos(2*math.pi*i/365))
    y.append(math.sin(2*math.pi*i/365))
    j.append("Day " + str(i))

# convert to np array
x = np.array(x)
y = np.array(y)

# set values between 0 and 1
x = (x-min(x)) / (max(x) - min(x))
y = (y-min(y)) / (max(y) - min(y))

# plot values
fig, ax = plt.subplots()
ax.scatter(x, y)

for i in range(len(j)):
    ax.annotate(j[i], (x[i], y[i]))

plt.title('Daily Order Mapped to cos(), sin()')
plt.xlabel('cos(2*pi*t/T)')
plt.ylabel('sin(2*pi*t/T)')
plt.show()