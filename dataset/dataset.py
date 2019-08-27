"""Util function for loading datasets
"""

import os
from collections import namedtuple
from functools import partial
import pickle
import joblib

import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.preprocessing import Normalizer, MinMaxScaler  # type: ignore
import sklearn.datasets as sk_datasets  # type: ignore
from sklearn.datasets import fetch_mldata

from typing import List, Dict, Tuple
from typing import Callable


# global var for setting data home folder
DataConfig = namedtuple("DataConfig", ["DATA_HOME"])
# client should set this var
data_config = DataConfig(DATA_HOME="")


# util function to set global var outside of this module
def set_data_home(data_home: str) -> None:
    """Set DATA_HOME global var"""
    global data_config
    data_config = data_config._replace(DATA_HOME=data_home)


def get_data_home() -> str:
    return data_config.DATA_HOME


def list_datasets() -> List[str]:
    print(f"\nData dir: {get_data_home()}")
    list_dir = os.listdir(get_data_home())
    print("\n".join(list_dir))
    return list_dir


def load_pickle(name: str) -> Dict:
    """Load pickle file in `name`"""
    with open(name, "rb") as infile:
        data = pickle.load(infile)
    return data


def load_coil20(N: int = 1440, fixed_random_seed: int = 1024) -> Dict:
    import scipy.io
    from sklearn.utils import shuffle

    mat = scipy.io.loadmat(f"{data_config.DATA_HOME}/COIL20/COIL20.mat")
    X, y = mat["X"], mat["Y"][:, 0]
    X, y = shuffle(X, y, n_samples=N, random_state=fixed_random_seed)
    labels = list(map(str, y.tolist()))
    return {"data": X, "target": y, "target_names": labels}


def load_country(year: int) -> Dict:
    pkl_obj = load_pickle(f"{data_config.DATA_HOME}/kaggle/country_indicators_{year}.pickle")
    return {"data": pkl_obj["data"], "target": pkl_obj["y"], "target_names": pkl_obj["labels"]}


def load_old_pickle(name: str) -> Dict:
    pkl_obj = load_pickle(f"{data_config.DATA_HOME}/kaggle/{name}.pickle")
    return {"data": pkl_obj["data"], "target": pkl_obj["y"], "target_names": pkl_obj["labels"]}


def coil20_loader(N: int) -> Callable:
    return partial(load_coil20, N)


def fashion_loader(N: int) -> Callable:
    return partial(load_pickle, f"{data_config.DATA_HOME}/fashion/{N}.pkl")


def quickdraw_loader(N: int) -> Callable:
    return partial(load_pickle, f"{data_config.DATA_HOME}/quickdraw/{N}.pkl")


def font_loader(ch: str, N: int) -> Callable:
    return partial(load_pickle, f"{data_config.DATA_HOME}/Font/{ch}_{N}.pkl")


def country_loader(year: int) -> Callable:
    return partial(load_country, year)


def old_pickle_loader(name: str) -> Callable:
    return partial(load_old_pickle, name)


def load_20newsgroups(n_samples: int = 2000, n_components: int = 20):
    file_name = f"{data_config.DATA_HOME}/20news/20NEWS{n_samples}_{n_components}.z"
    return joblib.load(file_name)


def load_scRNA_data(name):
    file_name = f"{data_config.DATA_HOME}/scRNA/{name}.z"
    return joblib.load(file_name)


def get_data_loaders() -> Dict[str, Callable]:
    """Build a mapping from dataset name to the function for loading data.
    Since the function makes use of the path to the data file,
    they must be built on the fly to take the newest values of DATA_HOME
    """
    return dict(
        [
            (f"FASHION{N}", fashion_loader(N))
            for N in [100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        ]
        + [(f"QUICKDRAW{N}", quickdraw_loader(N)) for N in [50, 90, 100, 120, 200, 500, 1000]]
        + [(f"FONT_{ch}_{N}", font_loader(ch, N)) for ch in ["A", "M", "E", "Z"] for N in [100]]
        + [(f"COIL20_{N}", coil20_loader(N)) for N in [100, 200, 500, 1000, 1440]]
        + [
            ("IRIS", sk_datasets.load_iris),
            ("DIGITS", sk_datasets.load_digits),
            ("WINE", sk_datasets.load_wine),
            ("BREAST_CANCER", sk_datasets.load_breast_cancer),
            ("COIL20", coil20_loader(N=1440)),
            ("MPI", old_pickle_loader("MPI_national")),
            ("DIABETES", old_pickle_loader("diabetes")),
            ("COUNTRY2014", country_loader(2014)),
            ("MNIST", lambda: fetch_mldata("MNIST original", data_home=get_data_home())),
            ("20NEWS", load_20newsgroups),
        ]
        + [
            ("PBMC_2K", partial(load_scRNA_data, "2k_pbmc_protein_3classes")),
            ("PBMC_5K", partial(load_scRNA_data, "5k_pbmc_protein_11classes")),
            ("PBMC_1K", partial(load_scRNA_data, "pbmc_1k_7classes")),
            ("NEURON_1K", partial(load_scRNA_data, "neuron_1k_6classes")),
            ("HEART_1K", partial(load_scRNA_data, "heart_1k_7classes")),
            ("QPCR", partial(load_scRNA_data, "guo_qpcr")),
        ]
    )


def load_dataset(
    name: str, preprocessing_method: str = "standardize", dtype: object = np.float32
) -> Tuple:
    """Load dataset from `config_data.DATA_HOME` folder

    Args:
        name: code name of the dataset.
        preprocessing_method: in `[None, 'standardize', 'normalize', 'unitScale']`

    Return:
        Tuple of np array: X_original, X_processed, y
    """
    data_loaders = get_data_loaders()
    load_func = data_loaders.get(name, None)
    if load_func is None:
        raise ValueError(
            "{} dataset is not available."
            "The available ones are:\n\t{}".format(name, "\n\t".join(data_loaders.keys()))
        )

    data = load_func()
    X_original = data["data"].astype(dtype)
    y = data["target"].astype(dtype)

    preprocessor = dict(
        standardize=StandardScaler, normalize=Normalizer, unitScale=MinMaxScaler
    ).get(preprocessing_method, None)
    X_processed = (
        X_original if preprocessor is None else preprocessor().fit_transform(X_original)
    )
    return (X_original, X_processed, y)


def load_dataset_multi_label(dataset_name):
    # dataset_name = "Automobile_transformed"
    in_name = f"./data/kaggle/{dataset_name}.pkl"
    data = joblib.load(in_name)
    return (data["data"], data["multi_aspects"])


if __name__ == "__main__":
    set_data_home("./data")
    print(get_data_home())
    # X_original, X, y = load_country(2014)
    # print(X_original.shape, X.shape, y.shape)
    _, X, y = load_dataset("PBMC_1K", preprocessing_method=None)
    print(X.shape, X.min(), X.max())
    print(len(np.unique(y)))
