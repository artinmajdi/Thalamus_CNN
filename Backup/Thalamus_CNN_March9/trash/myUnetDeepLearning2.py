from __future__ import division, print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util

nx = 572
ny = 572

generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
x_test, y_test = generator(1)
fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
