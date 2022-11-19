import matplotlib.pyplot as plt
import numpy as np

# our data
x = np.arange(3)
print(x)
yJax = [1, 1.545, 1.744]
yPyTorch = [1, 1.664, 2.276]
yIdeal = [1, 2, 4]
width = 0.2

# Major ticks every 1, minor ticks every .1
major_ticks = np.arange(0, 4.2, 1)
minor_ticks = np.arange(0, 4.2, .2)

# plot data in grouped manner
plt.bar(x+width, yJax, width, color='cyan')
plt.bar(x+width+width, yPyTorch, width, color='orange')
plt.bar(x+width+width+width, yIdeal, width, color='green')

# setup axis
plt.xticks(x+width+width, ['1 GPU', '2 GPUs', '4 GPUs'])
plt.yticks(major_ticks)
plt.yticks(minor_ticks, minor=True)
plt.grid(which='both', linestyle='--', axis='y')
plt.ylim(0)
plt.xlim(0)

# legends and titles
plt.title('Speed-up ratio')
plt.legend(["Jax", "PyTorch", "Ideal"])
plt.ylabel("Speed-up ratio")

plt.show()