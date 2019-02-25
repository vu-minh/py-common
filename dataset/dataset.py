'''Util function for loading datasets
'''

import os
from collections import namedtuple
from functools import partial
import pickle

import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.preprocessing import Normalizer, MinMaxScaler  # type: ignore
import sklearn.datasets as sk_datasets  # type: ignore

from typing import List, Dict, Tuple
from typing import Callable


# global var for setting data home folder
DataConfig = namedtuple('DataConfig', ['DATA_HOME'])
# client should set this var
data_config = DataConfig(DATA_HOME='')


# util function to set global var outside of this module
def set_data_home(data_home: str) -> None:
    '''Set DATA_HOME global var'''
    global data_config
    data_config = data_config._replace(DATA_HOME=data_home)


def get_data_home() -> str:
    return data_config.DATA_HOME


def list_datasets() -> List[str]:
    print(f"\nData dir: {get_data_home()}")
    list_dir = os.listdir(get_data_home())
    print('\n'.join(list_dir))
    return list_dir


def load_pickle(name: str) -> Dict:
    '''Load pickle file in `name`'''
    with open(name, 'rb') as infile:
        data = pickle.load(infile)
    return data


def fashion_loader(N: int) -> Callable:
    return partial(load_pickle, f"{data_config.DATA_HOME}/fashion/{N}.pkl")


def quickdraw_loader(N: int) -> Callable:
    return partial(load_pickle, f"{data_config.DATA_HOME}/quickdraw/{N}.pkl")


def get_data_loaders() -> Dict[str, Callable]:
    """Build a mapping from dataset name to the function for loading data.
    Since the function makes use of the path to the data file,
    they must be built on the fly to take the newest values of DATA_HOME
    """
    return dict(
        [(f"FASHION{N}", fashion_loader(N))
         for N in [100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000]] +
        [(f"QUICKDRAW{N}", quickdraw_loader(N))
         for N in [50, 90, 100, 120, 200, 500, 1000]] +
        [("IRIS", sk_datasets.load_iris),
         ("DIGITS", sk_datasets.load_digits),
         ("WINE", sk_datasets.load_wine),
         ("BREAST_CANCER", sk_datasets.load_breast_cancer)]
    )


def load_dataset(name: str, preprocessing_method: str = 'standardize',
                 dtype: object = np.float32) -> Tuple:
    """Load dataset from `config_data.DATA_HOME` folder

    Args:
        name: code name of the dataset.
        preprocessing_method: in `['standardize', 'normalize', 'unitScale']`

    Return:
        Tuple of np array: X_original, X_processed, y
    """
    data_loaders = get_data_loaders()
    load_func = data_loaders.get(name, None)
    if load_func is None:
        raise ValueError("{} dataset is not available."
                         "The available ones are:\n\t{}".format(
                            name, '\n\t'.join(data_loaders.keys())))

    data = load_func()
    X_original = data['data'].astype(dtype)
    y = data['target'].astype(dtype)

    preprocessor = dict(
        standardize=StandardScaler,
        normalize=Normalizer,
        unitScale=MinMaxScaler,
    ).get(preprocessing_method, None)
    X_processed = (X_original if preprocessor is None else
                   preprocessor().fit_transform(X_original))
    return (X_original, X_processed, y)


if __name__ == '__main__':
    set_data_home('./data')
    print(get_data_home())
    X_original, X, y = load_dataset('BREAST_CANCER')
    print(X_original.shape, X.shape, y.shape)
