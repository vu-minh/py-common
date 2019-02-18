'''Util function for loading datasets
'''

import os
from collections import namedtuple


# global var for setting data home folder
DataConfig = namedtuple('DataConfig', ['DATA_HOME'])
# client should set this var
data_config = DataConfig(DATA_HOME='')


# util function to set global var outside of this module
def set_data_home(data_home):
    global data_config
    data_config = data_config._replace(DATA_HOME=data_home)


def get_data_home():
    return data_config.DATA_HOME


def list_datasets():
    print(f"\nData dir: {get_data_home()}")
    list_dir = os.listdir(get_data_home())
    print('\n'.join(list_dir))
    return list_dir
