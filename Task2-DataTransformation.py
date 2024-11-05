#!/usr/bin/env python
# coding: utf-8

# # Task 2: Data Transformation of the Iris Dataset

# ## Objective

# This notebook covers data transformation techniques, including encoding categorical data, feature engineering, and aggregating data. Each transformation is explained, and key changes are demonstrated with before-and-after views.
# 

# ## Step 1: Load the Dataset
# The Iris dataset is loaded from iris.csv. This dataset will undergo several transformations to make it more analysis-ready.
# 

# In[3]:


# Import libraries
import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\user\\Downloads\\Iris.csv")

# Display the first few rows of the dataset
df.head()


# ## Step 2: Encoding Categorical Data

# The species column, which contains nominal data, is transformed using one-hot encoding. This technique creates binary columns for each species, making the data suitable for modeling. We use drop_first=True to avoid multicollinearity by omitting one of the categories.
# 

# In[4]:


# One-hot encoding for nominal variables
df_encoded = pd.get_dummies(df, columns=['Species'], drop_first=True)

# Display the dataset after encoding
df_encoded.head()


# ## Step 3: Feature Engineering
# To add more insight, we derived two new features:
# 
# Petal Area: Calculated by multiplying petal length and petal width.
# 
# Sepal Area: Calculated by multiplying sepal length and sepal width.
# 
# These engineered features might be more useful than raw measurements for some analysis.
# 

# In[5]:


# Create new features
df_encoded['petal_area'] = df['PetalLengthCm'] * df['PetalWidthCm']
df_encoded['sepal_area'] = df['SepalLengthCm'] * df['SepalWidthCm']

# Display the dataset after feature engineering
df_encoded.head()


# ## Step 4: Data Aggregation
# To summarize the data, we grouped by species and calculated the mean of each feature (sepal length, sepal width, petal length, and petal width). This provides insights into the average size of different iris species.

# In[6]:


# Group by species and calculate mean values for each feature
df_aggregated = df.groupby('Species').agg({
    'SepalLengthCm': 'mean',
    'SepalWidthCm': 'mean',
    'PetalLengthCm': 'mean',
    'PetalWidthCm': 'mean'
}).reset_index()

# Display aggregated dataset
df_aggregated


# ## Before-and-After Comparison
# The following snapshots provide a comparison of the dataset before and after transformations:

# Before Transformations: Shows the raw data loaded from iris.csv.
# 
# After Encoding and Feature Engineering: Highlights the dataset after applying one-hot encoding to species and adding new features (petal/sepal areas and petal-sepal length ratio).

# In[7]:


# Display dataset before transformations
print("Dataset before transformations:")
print(df.head())

# Display dataset after transformations (encoding and feature engineering)
print("\nDataset after encoding and feature engineering:")
print(df_encoded.head())


# # Conclusion

# This notebook demonstrated key data transformation steps on the Iris dataset. We encoded categorical variables, created additional features, and performed data aggregation. These transformations enhance the dataset's usability for machine learning and data analysis.

# In[ ]:




