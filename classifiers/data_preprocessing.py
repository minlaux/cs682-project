# imports
import glob
import re
import sqlite3
import time
import traceback
from pathlib import Path

import cv2
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

# # Heatmap to visualize sparsity
# plt.figure(figsize=(15, 8))
# sns.histplot(data=df_merged['gz2_class'], kde=False, color='skyblue', bins=50)
# plt.title('Galaxy Class Frequency Distribution', fontsize=16)
# plt.xlabel('Class Frequency', fontsize=14)
# plt.ylabel('Number of Classes', fontsize=14)
# plt.show()


# # Constants for image processing
# TARGET_SIZE = (106, 106)
# RECT_106_START = 212 - 106 // 2
# RECT_212_START = 212 - 212 // 2
# RECT_106_END = RECT_106_START + 106
# RECT_212_END = RECT_212_START + 212

# def process_image(image_path: str, visualize: bool = False, blur_sigma: float = 1.0):
#     """Preprocess an image for galaxy classification without OpenCV."""
#     # Load image
#     image_orig = plt.imread(image_path)  # Matplotlib for reading

#     # Convert to grayscale
#     gray = np.mean(image_orig, axis=2)  # Assuming RGB channels

#     # apply gaussian blur
#     blurred = gaussian(gray, sigma=blur_sigma)

#     # apply sobel edge detection
#     edges = filters.sobel(blurred)

#     # Thresholding
#     threshold = gray > 0.1  # Adjust threshold value as needed

#     # Find contours
#     contours = measure.find_contours(threshold, level=0.5)
#     if not contours:
#         return None

#     # Find the largest contour near the center
#     contour_info = [
#         (contour, distance.euclidean((212, 212), np.mean(contour, axis=0))) 
#         for contour in contours
#     ]
#     closest_contour = min(contour_info, key=lambda x: x[1])[0]

#     # Bounding box (min/max coordinates of the contour)
#     min_row, min_col = np.min(closest_contour, axis=0).astype(int)
#     max_row, max_col = np.max(closest_contour, axis=0).astype(int)

#     # Determine if the bounding box fits in the 106x106 central rectangle
#     in_106_rect = (min_row >= RECT_106_START and min_col >= RECT_106_START and
#                    max_row <= RECT_106_END and max_col <= RECT_106_END)

#     # Crop or resize image
#     if in_106_rect:
#         final_image = gray[RECT_106_START:RECT_106_END, RECT_106_START:RECT_106_END]
#     else:
#         cropped = gray[RECT_212_START:RECT_212_END, RECT_212_START:RECT_212_END]
#         final_image = np.array(Image.fromarray(cropped).resize(TARGET_SIZE))

#     if visualize:
#         # Display original image, blurred image, edges, and final processed image
#         fig, axs = plt.subplots(1, 5, figsize=(25, 5))
#         axs[0].imshow(image_orig, cmap='gray')
#         axs[0].set_title("Original Image")
#         axs[0].axis('off')

#         axs[1].imshow(gray, cmap='gray')
#         axs[1].set_title("Grayscale Image")
#         axs[1].axis('off')

#         axs[2].imshow(blurred, cmap='gray')
#         axs[2].set_title("Gaussian Blur")
#         axs[2].axis('off')

#         axs[3].imshow(edges, cmap='gray')
#         axs[3].set_title("Edges Detected")
#         axs[3].axis('off')

#         axs[4].imshow(final_image, cmap='gray')
#         axs[4].set_title("Final Processed Image")
#         axs[4].axis('off')

#         plt.tight_layout()
#         plt.show()

#     return final_image

# # Example: Process one image and visualize
# test_image_path = '../data/images_gz2/images/223272.jpg'  # Replace with actual image path
# processed_image = process_image(test_image_path, visualize=True)

def load_and_resize(image_path):
    """Load image, resize to 256x256."""
    image = plt.imread(image_path)  # or use cv2.imread for .png images
    image_resized = cv2.resize(image, (256, 256))  # Resize image to 256x256
    return image_resized

def convert_to_greyscale(image):
    """Convert image to greyscale by averaging over channels."""
    greyscale_image = np.mean(image, axis=2)  # Average RGB to get a single channel
    return greyscale_image

def random_flip(image):
    """Randomly flip the image horizontally and/or vertically."""
    if random.random() > 0.5:  # Random horizontal flip
        image = np.fliplr(image)
    if random.random() > 0.5:  # Random vertical flip
        image = np.flipud(image)
    return image

def random_rotation(image):
    """Rotate the image by a random angle between 0 and 90 degrees."""
    angle = random.randint(0, 90)  # Random angle between 0 and 90 degrees
    rotated_image = np.array(Image.fromarray(image.astype(np.uint8)).rotate(angle, resample=Image.NEAREST))
    return rotated_image

def adjust_contrast(image):
    """Adjust image contrast."""
    contrast_factor = random.uniform(0.98, 1.02)  # Random contrast between 98% and 102%
    image_contrast = np.clip(image * contrast_factor, 0, 255)  # Ensure pixel values stay in range
    return image_contrast

def random_crop(image, zoom_type="smooth", zoom_level=1.1):
    """Randomly crop or central crop based on zoom type."""
    height, width = image.shape[0], image.shape[1]
    
    if zoom_type == "smooth" or zoom_type == "featured":
        zoom_factor = random.uniform(1.1, 1.3)  # Zoom between 1.1x and 1.3x
    else:  # "bar"
        zoom_factor = random.uniform(1.7, 1.9)  # Zoom between 1.7x and 1.9x

    crop_height, crop_width = int(height / zoom_factor), int(width / zoom_factor)
    
    # Central crop
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Resize the cropped image to 256x256
    cropped_resized_image = cv2.resize(cropped_image, (256, 256))
    
    return cropped_resized_image

def final_resize(image):
    """Resize the final image to 128x128."""
    final_image = cv2.resize(image, (128, 128))  # Resize image to 128x128
    return final_image

def create_example(image):
    """Create a TFRecord example."""
    # Convert the image to bytes
    image_bytes = image.tobytes()

    # Create a feature dictionary
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))  # Example label
    }

    # Create an Example from the feature dictionary
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def save_to_tfrecord(images, output_path):
    """Save a list of images to a TFRecord file."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for image in images:
            example = create_example(image)
            writer.write(example.SerializeToString())

def preprocess_image(image_path, zoom_type="smooth", visualize=False):
    """Preprocess the image for training/testing."""
    image_resized = load_and_resize(image_path)
    image_greyscale = convert_to_greyscale(image_resized)
    image_flipped = random_flip(image_greyscale)
    image_rotated = random_rotation(image_flipped)
    image_contrast_adjusted = adjust_contrast(image_rotated)
    image_cropped = random_crop(image_contrast_adjusted, zoom_type=zoom_type)
    final_image = final_resize(image_cropped)

    if visualize:
        plt.imshow(final_image, cmap='gray')
        plt.title("Processed Image")
        plt.show()

    return final_image

image_paths = ["../data/images_gz2/images/223272.jpg", "../data/images_gz2/images/13062.jpg"]  # Add your image paths
processed_images = []

# Preprocess each image
for image_path in image_paths:
    processed_image = preprocess_image(image_path, zoom_type="smooth", visualize=False)
    processed_images.append(processed_image)

# Save images to a TFRecord file
save_to_tfrecord(processed_images, "output.tfrecord")



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