#!/usr/bin/env python
# coding: utf-8

# # Task 1: Data Cleaning of the Iris Dataset

# ## Objective

# This notebook demonstrates data cleaning techniques on the Iris dataset (loaded from iris.csv). Steps include handling missing values, removing duplicates, detecting and dealing with outliers, and feature scaling. Each step is explained in detail, and key transformations are shown with before-and-after views of the data.

# ## Step 1: Load the Dataset

# The Iris dataset is loaded from iris.csv into a Pandas DataFrame for easy data manipulation. This dataset contains measurements of iris flowers, including sepal length, sepal width, petal length, and petal width, along with a species column.

# In[13]:


# Import necessary libraries
import pandas as pd


# In[23]:


# Load the iris dataset from a CSV file
df=pd.read_csv('C:\\Users\\user\\Downloads\\Iris.csv')


# ## Step 2: Handling Missing Data

# To ensure data integrity, we first check for missing values using isnull().sum(). If missing values are found, they are imputed with the mean for each numerical column. This approach is particularly effective for small, numerical datasets like Iris.

# In[24]:


# Check for missing values in each column
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing values in numeric columns only
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)


# ## Step 3: Removing Duplicate Records

# Duplicate rows are identified using duplicated(). Any duplicate rows found are removed with drop_duplicates(). This step helps in maintaining the dataset’s integrity and preventing biases in analysis.

# In[25]:


# Check for duplicate records
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicate records, if any
df = df.drop_duplicates()
print("Duplicates removed.")


# ## Step 4: Outlier Detection and Handling

# Outliers can affect analysis, so we identify them using the Z-score method. Values beyond 3 standard deviations from the mean are considered outliers. These outliers are then removed to ensure the dataset remains reliable.

# In[26]:


from scipy.stats import zscore

# Calculate Z-scores for each feature column
z_scores = np.abs(zscore(df.iloc[:, :-1]))  # Exclude the species column if it is numeric
outliers = (z_scores > 3).any(axis=1)  # Identify rows with any Z-score > 3
print(f"Number of outliers detected: {outliers.sum()}")

# Remove outliers from the dataset
df_cleaned = df[outliers]


# ## Step 5: Feature Scaling

# To normalize the features, we apply StandardScaler to the feature columns (excluding species). This transformation standardizes the data, giving each feature a mean of 0 and a standard deviation of 1, which can improve the performance of machine learning algorithms.

# In[31]:


from sklearn.preprocessing import StandardScaler

# Standardize feature columns (excluding the 'species' column)
scaler = StandardScaler()
df_cleaned.loc[:, df_cleaned.columns != 'Species'] = scaler.fit_transform(df_cleaned.loc[:, df_cleaned.columns != 'Species'])

# Display the first few rows to show scaled features
df_cleaned.head()


# ## Before-and-After Comparison

# These snapshots provide a comparison of the dataset at different stages:
# 

# 1.Before Handling Outliers and Scaling: Shows the raw data loaded from iris.csv.
# 2.After Handling Outliers and Scaling: Shows the cleaned dataset, with outliers removed and features standardized.

# In[30]:


# Display dataset before and after handling outliers and scaling
print("Dataset before handling outliers and scaling:")
print(df.head())

print("\nDataset after handling outliers and scaling:")
print(df_cleaned.head())


# ## Conclusion

# In this notebook, we performed essential data cleaning steps on the Iris dataset, including handling missing values, removing duplicates, detecting and handling outliers, and feature scaling. The final dataset is now ready for analysis, with improved quality and consistency.
