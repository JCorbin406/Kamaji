import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 1000)

y1 = -0 * x**2
y2 = -1 * x**2
y3 = -10 * x**2

plt.plot(x, y1, label=r'$\gamma = 0$')
plt.plot(x, y2, label=r'$\gamma = 1$')
plt.plot(x, y3, label=r'$\gamma = 10$')
plt.xlabel('Deviation')
plt.ylabel('Disutility')
plt.grid()
plt.legend()
plt.show()