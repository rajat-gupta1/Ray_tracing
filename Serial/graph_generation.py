import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "output"
path = os.path.abspath(file_name)
data = np.genfromtxt(path, delimiter=",")
plt.axis('off')
plt.imshow(-data, cmap='Greys', interpolation=None)
plt.title("n_rays=1.0e8, n=1000")
png_file = "plot_1000_100.png"
plt.savefig(png_file)