"""Util function for loading datasets
"""

import os
from collections import namedtuple
from functools import partial
import pickle
import joblib

import numpy as np  # type: ignore
from scipy import stats
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.preprocessing import Normalizer, MinMaxScaler  # type: ignore
from sklearn.decomposition import PCA
import sklearn.datasets as sk_datasets  # type: ignore
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

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

    mat = scipy.io.loadmat(f"{data_config.DATA_HOME}/COIL20/COIL20.mat")
    X, y = mat["X"], mat["Y"][:, 0]
    X, y = shuffle(X, y, n_samples=N, random_state=fixed_random_seed)
    labels = list(map(str, y.tolist()))
    return {"data": X, "target": y, "target_names": labels}


def load_mnist(N: int = 2000, fixed_random_seed: int = 1024) -> Dict:
    data = fetch_openml("mnist_784", data_home=get_data_home())
    X, y = data["data"], data["target"]
    if N is not None:
        X, y = shuffle(X, y, n_samples=N, random_state=fixed_random_seed)
    labels = list(map(str, y.tolist()))
    return {"data": X, "target": y, "target_names": labels}


def load_country(year: int) -> Dict:
    pkl_obj = load_pickle(
        f"{data_config.DATA_HOME}/kaggle/country_indicators_{year}.pickle"
    )
    return {
        "data": pkl_obj["data"],
        "target": pkl_obj["y"],
        "target_names": pkl_obj["labels"],
    }


def load_old_pickle(name: str) -> Dict:
    pkl_obj = load_pickle(f"{data_config.DATA_HOME}/kaggle/{name}.pickle")
    return {
        "data": pkl_obj["data"],
        "target": pkl_obj["y"],
        "target_names": pkl_obj["labels"],
    }


def coil20_loader(N: int) -> Callable:
    return partial(load_coil20, N)


def mnist_loader(N: int) -> Callable:
    return partial(load_mnist, N)


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


def load_20newsgroups(n_samples: int = 2000, n_components: int = 20, subset=None):
    if subset is None:
        file_name = f"{data_config.DATA_HOME}/20news/20NEWS{n_samples}_{n_components}.z"
    else:
        file_name = f"{data_config.DATA_HOME}/20news/20NEWS{subset}.z"
    return joblib.load(file_name)


def load_scRNA_data(name):
    file_name = f"{data_config.DATA_HOME}/scRNA/{name}.z"
    return joblib.load(file_name)


def load_pretrained_data(name):
    data = joblib.load(f"{data_config.DATA_HOME}/pretrained/{name}.z")
    return {"data": data["data"], "target": data["target"]}


def get_data_loaders() -> Dict[str, Callable]:
    """Build a mapping from dataset name to the function for loading data.
    Since the function makes use of the path to the data file,
    they must be built on the fly to take the newest values of DATA_HOME
    """
    return dict(
        [  # subset of Fashion-MNIST/zalando research
            (f"FASHION{N}", fashion_loader(N))
            for N in [100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        ]
        + [  # google quickdraw
            (f"QUICKDRAW{N}", quickdraw_loader(N))
            for N in [50, 90, 100, 120, 200, 500, 1000]
        ]
        + [  # custom font dataset (build myself)
            (f"FONT_{ch}_{N}", font_loader(ch, N))
            for ch in ["A", "M", "E", "Z"]
            for N in [100]
        ]
        + [  # subset of COIL20
            (f"COIL20_{N}", coil20_loader(N)) for N in [100, 200, 500, 1000, 1440]
        ]
        + [  # subset of MNIST
            (f"MNIST{N}", mnist_loader(int(N) if N else None))
            for N in ["", 1000, 2000, 5000]
        ]
        + [  # common dataset from sklearn
            ("IRIS", sk_datasets.load_iris),
            ("DIGITS", sk_datasets.load_digits),
            ("WINE", sk_datasets.load_wine),
            ("BREAST_CANCER", sk_datasets.load_breast_cancer),
            ("COIL20", coil20_loader(N=1440)),
            ("MPI", old_pickle_loader("MPI_national")),
            ("DIABETES", old_pickle_loader("diabetes")),
            ("COUNTRY2014", country_loader(2014)),
            # ("MNIST", lambda: fetch_mldata("MNIST original", data_home=get_data_home())),
            (
                "OLIVETTI",
                lambda: sk_datasets.fetch_olivetti_faces(
                    data_home=f"{get_data_home()}/mldata"
                ),
            ),
        ]
        + [  # 20 newsgroups and a subset of 5 groups
            ("20NEWS", partial(load_20newsgroups, subset=None)),
            ("20NEWS5", partial(load_20newsgroups, subset=5)),
        ]
        + [  # genetic dataset from 10X Genomics
            ("PBMC_2K", partial(load_scRNA_data, "2k_pbmc_protein_3classes")),
            ("PBMC_5K", partial(load_scRNA_data, "5k_pbmc_protein_11classes")),
            ("PBMC_1K", partial(load_scRNA_data, "pbmc_1k_7classes")),
            ("NEURON_1K", partial(load_scRNA_data, "neuron_1k_6classes")),
            ("HEART_1K", partial(load_scRNA_data, "heart_1k_7classes")),
            ("QPCR", partial(load_scRNA_data, "guo_qpcr")),
        ]
        + [  # feature extraction from CNN
            (
                "FASHION_MOBILENET",
                partial(load_pretrained_data, "FASHION_MOBILENET_128"),
            )
        ]
    )


def load_dataset(
    name: str,
    preprocessing_method: str = "auto",
    pca: any = 0.9,
    dtype: object = np.float32,
) -> Tuple:
    """Load dataset from `config_data.DATA_HOME` folder

    Args:
        name: code name of the dataset.
        pca: float or int: run PCA for the dataset.
            if pca is a float in [0, 1], it is the percentage of expected variance.
            if pca is an int in [1, N], it is the number of dimentions to keep.
            if pca is zero or None, PCA is not performed.
        preprocessing_method: in `[None, 'standardize', 'normalize', 'unitScale']`

    Return:
        Tuple of np array: X_original, X_processed, y
    """
    data_loaders = get_data_loaders()
    load_func = data_loaders.get(name, None)
    if load_func is None:
        raise ValueError(
            "{} dataset is not available."
            "The available ones are:\n\t{}".format(
                name, "\n\t".join(data_loaders.keys())
            )
        )

    data = load_func()
    X_original = data["data"].astype(dtype)
    y = data["target"].astype(dtype)

    if preprocessing_method == "auto":
        preprocessing_method = {
            "COIL20": None,
            "QPCR": None,
            "NEURON_1K": None,
            "HEART_1K": None,
            "PBMC_1K": None,
            "PBMC_2K": None,
            "PBMC_5K": None,
            "FASHION_MOBILENET": None,
            "20NEWS5": None,
            "OLIVETTI": None,
        }.get(
            name, "unitScale"
        )  # default for image dataset

    preprocessor = dict(
        standardize=StandardScaler, normalize=Normalizer, unitScale=MinMaxScaler
    ).get(preprocessing_method, None)

    print(f"[DEBUG] X{X_original.shape}, y{y.shape}, n_class:{len(np.unique(y))}")
    X_processed = (
        preprocessor().fit_transform(X_original) if preprocessor else X_original
    )
    print(
        "[DEBUG] Preprocessing: ",
        preprocessing_method,
        stats.describe(X_processed, axis=None),
    )

    # do not run PCA for 10X genetic and 20News dataset
    if name.startswith(("QPCR", "NEURON", "HEART", "PBMC",) + ("20NEWS",)):
        pca = None
    X_processed = PCA(pca).fit_transform(X_processed) if pca else X_processed
    print("[DEBUG] PCA: ", pca, X_processed.shape)

    return (X_original, X_processed, y)


def load_dataset_multi_label(dataset_name):
    """[Deprecated] load multi labels dataset, e.g.

    Args:
        dataset_name: str, e.g. "Automobile_transformed"

    Return:
        Dict of 'data' and 'multi_aspects' labels
    """
    in_name = f"./data/kaggle/{dataset_name}.pkl"
    data = joblib.load(in_name)
    return (data["data"], data["multi_aspects"])


def load_additional_labels(dataset_name, label_name=""):
    """Load additional labels for a datatset for a given `label_name`
    Return:
        Tuple of (labels, description text)
        (None, error message) in case `label_name` not found.
    """
    in_name = {
        "NEURON_1K": "scRNA/neuron_1k_multi_labels",
        # ['graph_based_cluster', 'umi']
        "HEART_1K": "scRNA/heart_1k_multi_labels",
        # ['graph_based_cluster', 'umi']
        "PBMC_1K": "scRNA/pbmc_1k_multi_labels",
        # ['graph_based_cluster', 'umi']
        "FASHION_MOBILENET": "pretrained/FASHION_MOBILENET_128",
        # ['class_gender', 'class_subcat', 'class_matcat']
        "20NEWS5": "20news/20NEWS5",
        # ['cat', 'matcat']
    }.get(dataset_name, None)
    if in_name is None:
        return (None, None)
    data = joblib.load(f"{data_config.DATA_HOME}/{in_name}.z")
    other_labels = data["all_targets"]
    print(list(other_labels.keys()))
    return other_labels.get(label_name, (None, f"{label_name} does not exist."))


def is_image_dataset(dataset_name):
    return dataset_name.startswith(
        ("DIGITS", "MNIST", "FASHION", "COIL20", "QUICKDRAW", "FONT")
    ) and not dataset_name.endswith(("MOBILENET"))


if __name__ == "__main__":
    set_data_home("./data")
    print(get_data_home())

    dataset_name = "FASHION_MOBILENET"
    other_label_name = "class_subcat"  # None
    _, X, y = load_dataset(dataset_name, pca=0.9)

    if other_label_name is not None:
        labels2, des = load_additional_labels(dataset_name, label_name=other_label_name)
        if labels2 is not None:
            print(labels2.shape, des)
            print(np.unique(labels2, return_counts=True))
        else:
            print("Can not find other label data: ", des)
