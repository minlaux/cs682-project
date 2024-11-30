# imports
import glob
import re
import sqlite3
# import time

# from pathlib import Path

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import distance
# from sklearn.model_selection import train_test_split
# from skimage import measure, filters
# from skimage.filters import gaussian
# import tensorflow as tf
# import random
# from PIL import Image
# import os
# import torchvision.transforms as transforms
# import torch
import random
import torch
from torchvision import transforms
import os
from PIL import Image
from torchvision.utils import save_image



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

# find duplicates in mappings
print("Duplicates in objid to image mapping file:", df_mappings.duplicated("objid").value_counts())

# find duplicates in hart16 file
print("Duplicates in hart16 file:", df_hart16.duplicated("dr7objid").value_counts())

duplicate_mappings = df_mappings[df_mappings.duplicated("objid", keep=False)]\
                                .sort_values("asset_id")\
                                .reset_index(drop=True)
first_duplicate_id = duplicate_mappings['asset_id'].iloc[0]
last_duplicate_id = duplicate_mappings['asset_id'].iloc[-1]
print(f"asset_id in objid duplicates: First: {first_duplicate_id}, Last: {last_duplicate_id}, diff:{last_duplicate_id - first_duplicate_id}, count:{duplicate_mappings.shape[0]}")

# drop rows with asset_id not in image list to remove duplicates
df_mappings_clean = df_mappings[df_mappings["asset_id"] < first_duplicate_id]

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

list_asset_ids = df_merged["asset_id"].to_list()
# list of instances in data table but not in images
not_in_images = list(set(list_asset_ids) - set(image_names))
print("Number of instances in data table but not in images:", len(not_in_images))

# list of instances in images but not in data table
not_in_table = list(set(image_names) - set(list_asset_ids))
print("Number of instances in images but not in data table:", len(not_in_table))

# only keep rows that have data-image pairings
df_merged_clean = df_merged[~df_merged["asset_id"].isin(not_in_images)]
print(df_merged_clean.shape)

class_counts = df_merged_clean['gz2_class'].value_counts()
print("Number of classes:", class_counts.shape[0])

# # collect rare classes (median value)
# rare_classes = class_counts.loc[class_counts <= 10]
# print(rare_classes)

# # collect very rare classes
# very_rare_classes = class_counts.loc[class_counts <= 3]
# print(very_rare_classes)

# save cleaned DataFrame to SQLite
connection = sqlite3.connect("galaxy.db")
df_merged_clean.to_sql("galaxy_data", connection, index=False, if_exists="replace")
connection.close()

# define paths to input and output folders
input_folder = '../data/images_gz2/images'
# output_folder = '../data/images_gz2/processed_images'
output_folder = '../data/images_gz2/images_processed'

# define transformation function
def compute_train_transform(seed=123456):
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    colour_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    train_transform = transforms.Compose([
        # randomly resize and crop to 32 x 32
        transforms.RandomResizedCrop(32),  
        # horizontal flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),  
        # colour jitter with 80% probability
        transforms.RandomApply([colour_jitter], p=0.8),  
        # convert to greyscale with 20% probability
        transforms.RandomGrayscale(p=0.2),  
        # convert image to tensor
        transforms.ToTensor(),  
    ])
    return train_transform

# # load image for testing
# image_path = "/content/images_gz2/images/223272.jpg"  # Update with your image path
# img = Image.open(image_path)

# train_transform = compute_train_transform()
# transformed_img_tensor = train_transform(img)

# transformed_img = transforms.ToPILImage()(transformed_img_tensor)

# # display the processed image
# plt.imshow(transformed_img)
# plt.axis('off')
# plt.show()


# ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

processed = 0

# # loop over all images in the input folder
# for filename in image_names:
#     img_path = os.path.join(input_folder, filename)
    
#     # load image
#     img = Image.open(img_path)
    
#     # apply transformation
#     train_transform = compute_train_transform()
#     img_transformed = train_transform(img)
    
#     # save transformed image to output folder
#     # IMPORTANT: keep image filename the same as for not processed
#     # image filename corresponds to asset id in dataframe
#     output_path = os.path.join(output_folder, filename)
#     save_image(img_transformed, output_path)
#     processed +=1

#     if (processed % 10000 == 0):
#         print("Processed", processed, "images")
# loop over all images in the input folder
for filename in image_names:
    img_path = os.path.join(input_folder, str(filename) + '.jpg')  # Ensure the filename is a string with .jpg extension
    
    # load image
    img = Image.open(img_path)
    
    # apply transformation
    train_transform = compute_train_transform()
    img_transformed = train_transform(img)
    
    # save transformed image to output folder
    output_path = os.path.join(output_folder, str(filename) + '.jpg')  # Save with same filename
    save_image(img_transformed, output_path)
    processed += 1
    print(f"Saved processed image: {output_path}")

    if (processed % 10000 == 0):
        print("Processed", processed, "images")
