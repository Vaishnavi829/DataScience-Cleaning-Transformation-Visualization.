#!/usr/bin/env python
# coding: utf-8

# # Task 3: Data Visualization of the Iris Dataset

# ## Objective

# This notebook provides data visualizations for the Iris dataset using Matplotlib and Seaborn. Visualizations are customized with labels, titles, and colors to highlight insights about iris species and feature relationships.
# 

# ## Step 1: Load the DatasetThe Iris dataset is loaded from iris.csv into a Pandas DataFrame. 
# This dataset contains measurements of iris flowers,including sepal and petal dimensions, along with a species column representing the flower class.
# 
# 

# In[2]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
df = pd.read_csv("C:\\Users\\user\\Downloads\\Iris.csv")

# Display the first few rows of the dataset
df.head()


# ## Visualization 1: Histogram of Sepal and Petal Lengths
# These histograms show the distribution of sepal and petal lengths in the dataset. The KDE (Kernel Density Estimate) overlay hel- ps visualize the distribution’s shape. Sepal lengths appear to be more uniformly spread, while petal lengths show a more distin- ct bimodal pattern.

# In[3]:


# Set the figure size
plt.figure(figsize=(12, 6))

# Create histograms for sepal length and petal length
plt.subplot(1, 2, 1)
sns.histplot(df['SepalLengthCm'], kde=True, color="skyblue")
plt.title("Distribution of Sepal Length")
plt.xlabel("SepalLengthCm")

plt.subplot(1, 2, 2)
sns.histplot(df['PetalLengthCm'], kde=True, color="salmon")
plt.title("Distribution of Petal Length")
plt.xlabel("PetalLengthCm")

plt.tight_layout()
plt.show()


# ## Visualization 2: Scatter Plot of Sepal Length vs. Petal Length by Species
# This scatter plot shows the relationship between sepal length and petal length for each species. Species appear to cluster distinctly, with Setosa having shorter petals and Versicolor and Virginica showing increasing petal lengths, indicating potential searability by these features.

# In[4]:


# Create a scatter plot with species as hue
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', hue='Species', data=df, palette="viridis")
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.legend(title="Species")
plt.show()


# ## Visualization 3: Correlation Heatmap of All Features
# The heatmap displays correlations between numerical features in the Iris dataset. Petal length and petal width are highly correlated, which may suggest redundancy. Sepal width shows a weaker relationship with other features, indicating its distinct nature.
# 
# 

# In[5]:


# Calculate correlation matrix and plot heatmap
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()


# ## Conclusion
# This notebook provided insights into the Iris dataset through various visualizations. Histograms helped show feature distributions, scatter plots illustrated relationships across species, and the correlation heatmap revealed relationships among features. These insights are valuable for further analysis and model building.

# In[ ]:




