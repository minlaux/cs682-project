# imports
import sqlite3
from sklearn.model_selection import train_test_split


# # downloaded hart16 files from https://data.galaxyzoo.org/#section-7
# # downloaded images and mapping from https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images/data
# FILENAME_MAPPINGS = "data/gz2_filename_mapping.csv"
# HART16 = "data/gz2_hart16.csv"

# ORIGINAL_IMAGES_DIR = "data/images/"
# PROCESSED_IMAGES_DIR = "data/images_processed/"
# RANDOM_STATE = 32
# TRAIN_IMAGES_DIR = "data/train_images/"
# TEST_IMAGES_DIR = "data/test_images/"


import pandas as pd

# follow the directory structure specified in the repository 
# downloaded heart16 file from https://data.galaxyzoo.org/#section-7
df_hart16 = pd.read_csv('../data/gz2_hart16.csv')
df_hart16.info()
# downloaded images and mapping from https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images/data
df_mappings = pd.read_csv('../data/gz2_filename_mapping.csv')
df_mappings.info()