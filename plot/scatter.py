"""Util for scatter plot
"""
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse

import seaborn


plt.rcParams.update({'axes.titlesize': 'xx-large'})
seaborn.set(style='whitegrid')


def imscatter(ax, X2d, data, zoom=1, inverse_cmap=True,
              custom_cmap=None, labels_true=None):
    img_size = int(math.sqrt(data.shape[1]))
    use_gray_cmap = (custom_cmap is None) or (labels_true is None)

    artists = []
    for i, [x0, y0] in enumerate(X2d):
        if use_gray_cmap:
            cmap = 'gray_r' if inverse_cmap else 'gray'
        else:
            label_i = labels_true[i]
            cmap = custom_cmap[label_i % 10]

        im = OffsetImage(data[i].reshape(img_size, img_size),
                         zoom=zoom, cmap=cmap, alpha=0.9)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        # if wanna using frame color, set: bboxprops=dict(edgecolor='red')
        artists.append(ax.add_artist(ab))
    return artists


def ellipse_scatter(ax, X2d_loc, X2d_scale, labels_true=None, alpha=0.5):
    colors = cm.tab10
    for xy, (scale_x, scale_y), label in zip(X2d_loc, X2d_scale, labels_true):
        e = Ellipse(xy=xy, angle=0, width=scale_x, height=scale_y)
        e.set_clip_box(ax.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(colors(label))
        ax.add_artist(e)
