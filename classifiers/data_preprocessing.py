# imports
import glob
import re
import sqlite3
import time
import traceback
from pathlib import Path

# import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from skimage import measure, filters
from skimage.filters import gaussian
import tensorflow as tf
import random
from PIL import Image
import os
import torchvision.transforms as transforms



# follow the directory structure specified in the repository 
# downloaded heart16 file from https://data.galaxyzoo.org/#section-7
df_hart16 = pd.read_csv('../data/gz2_hart16.csv')
# print("HART16 Dataset Info:")
# df_hart16.info()

# downloaded images and mapping from https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images/data
df_mappings = pd.read_csv('../data/gz2_filename_mapping.csv')
# print("\nMappings Dataset Info:")
# df_mappings.info()

# only keep first 4 columns and debiased data
columns_to_keep = ["dr7objid", "ra", "dec", "gz2_class"] + [col for col in df_hart16.columns if col.endswith('_debiased')]
filtered_hart16 = df_hart16[columns_to_keep]

# merge dataframes on object id
df_merged = pd.merge(df_mappings, filtered_hart16, 
                     left_on='objid',
                     right_on='dr7objid')

# check for null rows
null_rows = df_merged.isnull().any(axis=1)
# get total number of null rows
num_null_rows = null_rows.sum()
# drop null rows
df_merged.dropna(inplace=True)

print(df_merged.shape)

# get list of all images in the image folder
image_files = glob.glob('../data/images_gz2/images/*.jpg')
print("Image count:", len(image_files))

# regex to match numeric image file names
regex = re.compile(r".*[/\\](\d+)\.jpg")

# extract image names
# image name corresponds to assed_id in data table
image_names = []
for img in image_files:
    match = re.search(regex, img)
    if match:
        image_names.append(int(match.group(1)))

# sort and display
image_names.sort()
# print("First 10 image names:", image_names[:10])

print(df_merged.head())

class_counts = df_merged['gz2_class'].value_counts()
print("Number of classes:", class_counts.shape[0])

# # collect rare classes
# rare_classes = class_counts.loc[class_counts <= 15]

# # collect very rare classes
# very_rare_classes = class_counts.loc[class_counts <= 5]


# code below visualised data distribution of the 20 most frequent and 20 least frequent classes
# import seaborn as sns

# # Top 20 most frequent classes
# top_20_classes = class_counts.head(20)

# # Top 20 least frequent classes
# least_20_classes = class_counts.tail(20)

# # Plot the top 20 most frequent classes
# plt.figure(figsize=(12, 6))
# sns.barplot(x=top_20_classes.index, y=top_20_classes.values, palette='viridis')
# plt.title('20 Most Frequent Classes', fontsize=16)
# plt.xlabel('Galaxy Class', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(rotation=45)
# plt.show()

# # Plot the 20 least frequent classes
# plt.figure(figsize=(12, 6))
# sns.barplot(x=least_20_classes.index, y=least_20_classes.values, palette='viridis')
# plt.title('20 Least Frequent Classes', fontsize=16)
# plt.xlabel('Galaxy Class', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(rotation=45)
# plt.show()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(contrast=(0.98, 1.02)),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])

image = Image.open("../data/images_gz2/images/223272.jpg")
processed_image = transform(image)