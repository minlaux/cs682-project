# imports
import sqlite3
from sklearn.model_selection import train_test_split
import gdown

# Replace with the shared link or file ID
url = 'https://drive.google.com/uc?id=your_file_id'
output = 'file_name.extension'
gdown.download(url, output, quiet=False)



# download hart16 files from: https://data.galaxyzoo.org/#section-12
# download images and mapping from: https://zenodo.org/record/3565489#.Y3vFKS-l0eY
FILENAME_MAPPINGS = "data/gz2_filename_mapping.csv"
HART16 = "data/gz2_hart16.csv"

ORIGINAL_IMAGES_DIR = "data/images/"
PROCESSED_IMAGES_DIR = "data/images_processed/"
RANDOM_STATE = 32
TRAIN_IMAGES_DIR = "data/train_images/"
TEST_IMAGES_DIR = "data/test_images/"


