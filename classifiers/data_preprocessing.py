# imports
import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

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
print(f"Number of null rows: {num_null_rows}")
# drop null rows
df_merged.dropna(inplace=True)

# # Distribution of Features where value > 0.5
# labels = [text.replace("_debiased", "") for text in columns_to_keep[4:]]
# feature_distribution = (df_merged[columns_to_keep[4:]] > 0.5).sum()
# display(feature_distribution)
# ax = feature_distribution.plot.bar(figsize=(16, 6))
# ax.set_xticklabels(labels)
# plt.show()
