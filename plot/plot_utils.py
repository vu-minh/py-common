import numpy as np
from matplotlib import pyplot as plt


def draw_seperator_between_subplots(fig, axes, color="black", linestyle="--", linewidth=1.0):
    """Draw line seperator between subplots.
    Note: `fig` should be "tighted", i.e., call `fig.tight_layout()` before using this func
    Ref: https://stackoverflow.com/questions/26084231/draw-a-separator-or-lines-between-subplots
    """
    import matplotlib.transforms as mtrans

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D(
            [0, 1],
            [y, y],
            transform=fig.transFigure,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        fig.add_artist(line)
