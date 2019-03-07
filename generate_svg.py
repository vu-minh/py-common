import math
import numpy as np
from dataset.dataset import set_data_home, load_dataset
from plot.export_utils import generate_stacked_svg


set_data_home('./data')
output_svg_dir = './data/svg'
list_dataset_names = [
    'DIGITS',
    'FASHION100', 'FASHION200', 'FASHION500',
    'QUICKDRAW100', 'QUICKDRAW200',
    'COIL20_100', 'COIL20_200'
]

for dataset_name in list_dataset_names:
    X_original, X, y = load_dataset(dataset_name)
    print('\nDataset: ', dataset_name, X.shape, y.shape)
    if dataset_name.lower().startswith('coil'):
        N, D = X.shape
        img_size = int(math.sqrt(D))
        X_original = np.swapaxes(X_original.reshape(N, img_size, img_size),
                                 axis1=1, axis2=2)
        X_original = X_original.reshape(N, -1)

    for cmap_type in ['gray', 'gray_r', 'color']:
        labels = y if cmap_type == 'color' else None
        svg_out_name = f"{output_svg_dir}/{dataset_name}_{cmap_type}.svg"
        print('Generate: ', svg_out_name)
        generate_stacked_svg(svg_out_name=svg_out_name, dataset=X_original,
                             labels=labels, default_cmap=cmap_type)
