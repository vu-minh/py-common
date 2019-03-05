import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb


SVG_META_DATA = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with matplotlib (http://matplotlib.org/),
    modified to stack multiple svg elemements,
    used for packing all images in a dataset
-->
<svg version="1.1" width="28" height="28" viewBox="0 0 28 28"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink">
<defs>
<style type="text/css">
    *{stroke-linecap:butt;stroke-linejoin:round;}
    .sprite { display: none;}
    .sprite:target { display: block; margin-left: auto; margin-right: auto; }
    .packed-svg-custom {/*override this css to customize style for svg image*/}
</style>
</defs>
"""

SVG_IMG_TAG = """
<g class="sprite" id="{}">
    <image class="packed-svg-custom"
        id="stacked_svg_img_{}"
        width="28"
        height="28"
        xlink:href="data:image/png;base64,{}"
    />
</g>
"""


def generate_stacked_svg(svg_out_name, dataset,
                         labels=None, default_cmap='gray_r'):
    """Create an SVG to store all image of a `dataset`.
        To access an image, use svg_out_name.svg#img_id, e.g. MNIST.svg#123
    """
    # current_dpi = plt.gcf().get_dpi()
    # fig = plt.figure(figsize=(28 / current_dpi, 28 / current_dpi))

    def _create_cm(basecolor):
        colors = [(1, 1, 1), to_rgb(basecolor), to_rgb(basecolor)]  # R->G->B
        return LinearSegmentedColormap.from_list(colors=colors, name=basecolor)

    def _create_custom_cmap():
        basecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        return list(map(_create_cm, basecolors))

    def _generate_figure_data(img, cmap):
        fig_file = BytesIO()
        plt.imsave(fig_file, img, cmap=cmap)
        plt.gcf().clear()
        fig_file.seek(0)
        return base64.b64encode(fig_file.getvalue()).decode('utf-8')

    N, D = dataset.shape
    img_size = int(math.sqrt(D))
    custom_cmap = _create_custom_cmap()

    with open(svg_out_name, 'w') as svg_file:
        svg_file.write(SVG_META_DATA)

        for i in range(N):
            img = dataset[i].reshape(img_size, img_size)
            cmap = (default_cmap if labels is None
                    else custom_cmap[int(labels[i]) % 10])
            fig_data = _generate_figure_data(img, cmap)
            svg_file.write(SVG_IMG_TAG.format(i, i, fig_data))

        svg_file.write("</svg>")
