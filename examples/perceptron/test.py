import numpy as np
from matplotlib import pyplot as plt

x1 = np.arange(-0.1, 1.1, 0.05)
x2 = np.arange(-0.1, 1.1, 0.05)
x1, x2 = np.meshgrid(x1, x2)
y = 0.4210 * x1 + 0.4210 * x2 - 0.1563

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(0, 0, 0, marker='^', c='r')
ax.scatter(0, 1, 0, marker='^', c='r')
ax.scatter(1, 0, 0, marker='^', c='r')
ax.scatter(1, 1, 1, marker='^', c='r')
ax.plot_surface(x1, x2, y)
plt.show()
