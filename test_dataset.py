'''Test dataset util functions
'''

import unittest
import numpy as np

from dataset.dataset import set_data_home, get_data_home
from dataset.dataset import list_datasets
from dataset.dataset import load_dataset


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        set_data_home('./data')
        self.list_datasets = [
            'FONT_A_100',
            'FASHION100', 'QUICKDRAW200',
            'IRIS', 'DIGITS', 'WINE', 'BREAST_CANCER',
            'COIL20', 'COIL20_500'
        ]

    def test_data_home(self):
        self.assertEqual(get_data_home(), './data')

    def test_list_dir(self):
        self.assertIsInstance(list_datasets(), list)

    def test_dataset_name_does_not_exist(self):
        with self.assertRaises(ValueError):
            load_dataset('NameNotExist')

    def test_load_dataset_return_3_array(self):
        for dataset_name in self.list_datasets:
            data = load_dataset(dataset_name)
            self.assertEqual(len(data), 3)

    def test_dataset_type(self):
        for dataset_name in self.list_datasets:
            data = load_dataset(dataset_name)
            self.assertListEqual(
                list(map(lambda X: X.dtype, data)), [np.float32]*3
            )

    def test_dataset_size(self):
        expected_size = dict(
            COIL20=[(1440, 32*32), (1440, 32*32), (1440,)],
            COIL20_500=[(500, 32*32), (500, 32*32), (500,)],
            FONT_A_100=[(100, 24*24), (100, 24*24), (100,)],
            FASHION100=[(100, 28*28), (100, 28*28), (100,)],
            QUICKDRAW200=[(200, 28*28), (200, 28*28), (200,)],
            IRIS=[(150, 4), (150, 4), (150,)],
            DIGITS=[(1797, 8*8), (1797, 8*8), (1797,)],
            WINE=[(178, 13), (178, 13), (178,)],
            BREAST_CANCER=[(569, 30), (569, 30), (569,)]
        )

        for dataset_name, dataset_size in expected_size.items():
            data = load_dataset(dataset_name)
            self.assertListEqual(
                list(map(lambda X: X.shape, data)), dataset_size
            )


# TODO TestCase class to test data preprocessing:
# test_standardized_data (zero-mean, unit-variance)
# test_normalized_data (unit-variance)
# test_unit_scaled_data (min=0, max=1)


if __name__ == "__main__":
    unittest.main()
