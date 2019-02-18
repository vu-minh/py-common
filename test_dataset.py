'''Test dataset util functions
'''

import unittest

from dataset.dataset import set_data_home, get_data_home
from dataset.dataset import list_datasets


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        set_data_home('./data')

    def test_data_home(self):
        self.assertEqual(get_data_home(), './data')

    def test_list_dir(self):
        self.assertIsInstance(list_datasets(), list)



if __name__ == "__main__":
    unittest.main()
