"""Util for scatter plot
"""
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb

# import seaborn


# plt.rcParams.update({"axes.titlesize": "xx-large"})
# seaborn.set(style="whitegrid")


def config_font_size(min_size=12):
    """https://stackoverflow.com/a/39566040/5088950
    """
    SMALL_SIZE = min_size
    MEDIUM_SIZE = min_size + 4
    BIGGER_SIZE = min_size + 4 + 6

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def scatter3d(X, labels=None, title="", out_name="scatter3d.html"):
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(size=6, color=labels, colorscale="jet", opacity=0.6),
            ),
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_html(out_name)


def simple_scatter(
    X,
    labels=None,
    title="",
    out_name="simple_scatter.png",
    aspect_equal=True,
    show_legend=True,
):
    """2D scatter plot and save figure to `out_name`
    """
    n_classes = len(np.unique(labels))
    cmap = "tab10_r" if n_classes <= 10 else "tab20"
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if aspect_equal:
        ax.set_aspect("equal")

    ax.set_title(title)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.6, cmap=cmap)

    if show_legend:
        legend1 = ax.legend(
            *scatter.legend_elements(num=n_classes), loc="best", title="Class"
        )
        ax.add_artist(legend1)

    fig.savefig(out_name, bbox_inches="tight")
    plt.close(fig)


def scatter_with_box(ax, all_pos, marker="s", color="blue"):
    """Scatter only with non-filled marker
    """
    ax.scatter(
        all_pos[:, 0],
        all_pos[:, 1],
        marker=marker,
        s=256,
        facecolor="none",
        edgecolor=color,
        linewidth=2.0,
    )


def annotate_text(ax, text, pos, text_color="blue", offset=(-10, 10)):
    """Set annotation text for only one point at fixed position
    """
    ax.annotate(
        s=str(text), xy=pos, xytext=offset, textcoords="offset points", color=text_color
    )


def get_custom_cmap():
    def create_cm(basecolor):
        colors = [(1, 1, 1), to_rgb(basecolor), to_rgb(basecolor)]  # R -> G -> B
        return LinearSegmentedColormap.from_list(colors=colors, name=basecolor)

    basecolors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    cmaps = []
    for basecolor in basecolors:
        cmaps.append(create_cm(basecolor))
    return cmaps


def imscatter(
    ax,
    X2d,
    data,
    zoom=1.0,
    inverse_cmap=True,
    custom_cmap=None,
    labels_true=None,
    frameon=False,
):
    img_size = int(math.sqrt(data.shape[1]))
    use_gray_cmap = (not custom_cmap) or (labels_true is None)

    if img_size < 16:
        zoom *= 2
    elif img_size > 64:
        zoom *= 0.5

    artists = []
    for i, [x0, y0] in enumerate(X2d):
        if use_gray_cmap:
            cmap = "gray_r" if inverse_cmap else "gray"
        else:
            label_i = int(labels_true[i])
            cmap = get_custom_cmap()[label_i % 10]

        im = OffsetImage(
            data[i].reshape(img_size, img_size), zoom=zoom, cmap=cmap, alpha=1.0
        )
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=frameon)
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
