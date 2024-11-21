# # imports
# import sqlite3
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import matplotlib.pyplot as plt
import glob
import re
import sqlite3
import time
import traceback
from pathlib import Path

#import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

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
print("First 10 image names:", image_names[:10])

print(df_merged.head())

class_counts = df_merged['gz2_class'].value_counts()
print("Number of classes:", class_counts.shape[0])

import seaborn as sns

# Top 10 most frequent classes
top_10_classes = class_counts.head(10)

# Top 10 least frequent classes
least_10_classes = class_counts.tail(10)

# # Plot the top 10 most frequent classes
# plt.figure(figsize=(12, 6))
# sns.barplot(x=top_10_classes.index, y=top_10_classes.values, palette='viridis')
# plt.title('Top 10 Most Frequent Classes', fontsize=16)
# plt.xlabel('Galaxy Class', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(rotation=45)
# plt.show()

# # Plot the 10 least frequent classes
# plt.figure(figsize=(12, 6))
# sns.barplot(x=least_10_classes.index, y=least_10_classes.values, palette='viridis')
# plt.title('10 Least Frequent Classes', fontsize=16)
# plt.xlabel('Galaxy Class', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(rotation=45)
# plt.show()
# # Heatmap to visualize sparsity
# plt.figure(figsize=(15, 8))
# sns.histplot(data=df_merged['gz2_class'], kde=False, color='skyblue', bins=50)
# plt.title('Galaxy Class Frequency Distribution', fontsize=16)
# plt.xlabel('Class Frequency', fontsize=14)
# plt.ylabel('Number of Classes', fontsize=14)
# plt.show()


# # Bar plot for class distribution
# plt.figure(figsize=(10, 6))
# class_counts.plot(kind='bar', color='skyblue')
# plt.title('Class Distribution', fontsize=16)
# plt.xlabel('Galaxy Classes', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# # Constants
# IMAGE_DIR = '../data/images_gz2/images'
# IMG_SIZE = (128, 128)  # Resize all images to 128x128
# BATCH_SIZE = 32

# # Load data (assuming `df_merged` contains 'gz2_class' and 'dr7objid')
# df_merged['image_path'] = df_merged['dr7objid'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
# df_merged = df_merged[df_merged['image_path'].apply(os.path.exists)]  # Keep rows with existing images

# # Train-validation-test split
# train_df, test_df = train_test_split(df_merged, test_size=0.2, stratify=df_merged['gz2_class'], random_state=42)
# train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['gz2_class'], random_state=42)

# print("Train size:", len(train_df))
# print("Validation size:", len(val_df))
# print("Test size:", len(test_df))

# # Data generators
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_test_datagen = ImageDataGenerator(rescale=1./255)

# def data_generator(df, datagen, batch_size=BATCH_SIZE, target_size=IMG_SIZE):
#     return datagen.flow_from_dataframe(
#         df,
#         x_col='image_path',
#         y_col='gz2_class',
#         target_size=target_size,
#         class_mode='categorical',
#         batch_size=batch_size
#     )

# train_generator = data_generator(train_df, train_datagen)
# val_generator = data_generator(val_df, val_test_datagen)
# test_generator = data_generator(test_df, val_test_datagen)

# # Plot class distribution
# plt.figure(figsize=(12, 6))
# train_df['gz2_class'].value_counts().plot(kind='bar', color='skyblue')
# plt.title('Class Distribution in Training Set')
# plt.xlabel('Classes')
# plt.ylabel('Frequency')
# plt.show()
