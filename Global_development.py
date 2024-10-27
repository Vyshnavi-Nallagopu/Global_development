#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("World_development_mesurement (1).csv")
data.head()


# In[2]:


data.tail()


# In[3]:


print(data.shape)


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


#Visualizing Missing Values
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.ylabel('Rows')
plt.xlabel('Columns')
plt.title('Missing Values Heatmap')
plt.show()


# In[7]:


# Drop columns with excessive missing values
data.drop(['Ease of Business', 'Business Tax Rate'], axis=1, inplace=True)


# In[8]:


import pandas as pd
import numpy as np

def clean_and_convert(data, col_name):
    # Convert to string and remove non-numeric characters
    data[col_name] = data[col_name].astype(str).str.replace('[^0-9.]', '', regex=True)

    # Remove leading/trailing whitespace
    data[col_name] = data[col_name].str.strip()

    # Convert to numeric, handling errors
    try:
        data[col_name] = pd.to_numeric(data[col_name], errors='coerce')
    except ValueError as e:
        print(f"Error converting {col_name}: {e}")
        print(data[col_name].head(20))

    # Handle missing values (e.g., using mean imputation)
    if data[col_name].isnull().sum() > 0:
        data[col_name].fillna(data[col_name].mean(), inplace=True)

    return data

# Assuming your DataFrame is 'df'
currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']

for col in currency_cols:
    data= clean_and_convert(data, col)

# Print the data types after conversion
print(data.dtypes)


# In[9]:


data.info()


# In[10]:


# Impute missing values in numerical columns
numerical_cols = ['Birth Rate','CO2 Emissions','Days to Start Business','Energy Usage','Health Exp % GDP','Hours to do Tax','Infant Mortality Rate','Internet Usage','Lending Interest','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage','Number of Records','Population 0-14','Population 15-64','Population 65+','Population Total','Population Urban']
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())


# In[11]:


categorical_cols = ['Country']
mode_value = data['Country'].mode()
data['Country'].fillna(mode_value, inplace=True)


# In[12]:


data.info()


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_cleaned = data.copy()

def identify_outliers(data, column_name):

    z_scores = np.abs((data[column_name] - data[column_name].mean()) / data[column_name].std())
    outliers = data[z_scores > 3].index
    return outliers

def handle_outliers(data, column_name, method='trimming'):

    if method == 'trimming':
        outliers = identify_outliers(data, column_name)
        data.drop(outliers, inplace=True)  # Trimming outliers directly from 'data'
    elif method == 'capping':
        threshold = data[column_name].quantile(0.95)
        data[column_name] = np.where(data[column_name] > threshold, threshold, data[column_name])
    elif method == 'winsorization':
        lower_percentile = 0.05
        upper_percentile = 0.95
        lower_bound = data[column_name].quantile(lower_percentile)
        upper_bound = data[column_name].quantile(upper_percentile)
        data[column_name] = np.clip(data[column_name], lower_bound, upper_bound)

    return data

# Identify and handle outliers in all numerical columns
numerical_cols = data.select_dtypes(include=np.number).columns

for col in numerical_cols:
    # Visualize outliers using box plot (on the original data)
    sns.boxplot(x=data[col])
    plt.title(f'Box Plot for {col} (Before Outlier Handling)')
    plt.show()

   


# In[14]:


# Handle outliers (adjust the method as needed)
data_cleaned = handle_outliers(data_cleaned, col, method='trimming')  # Handle on the copy


# In[15]:


# After handling outliers, re-visualize the data to verify the impact
for col in numerical_cols:
    # Visualize outliers using box plot (on the cleaned data)
    sns.boxplot(x=data_cleaned[col])
    plt.title(f'Box Plot for {col} (After Outlier Handling)')
    plt.show()


# In[16]:


data_cleaned.head()
data_new=data_cleaned


# In[17]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
# Identify numerical and categorical columns
numeric_cols = data_new.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data_new.select_dtypes(include=['object']).columns





# In[18]:


# Scale numerical columns
scaler = StandardScaler()
data_new[numeric_cols] = scaler.fit_transform(data_new[numeric_cols])


# In[19]:


# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    data_new[col] = label_encoder.fit_transform(data_new[col])


# In[20]:


print(data_new)


# In[21]:


import matplotlib.pyplot as plt
# Pair plot
sns.pairplot(data_cleaned)
plt.show()


# In[22]:


# Correlation matrix
corr_matrix = data_cleaned.corr()


# In[23]:


# Heatmap
import warnings
# Create a larger figure for better clarity
plt.figure(figsize=(15,8))
# Heatmap with adjustments
# Suppress the FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")  # Adjust parameters for better visualization

# Customize labels and title for clarity
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.title('Correlation Matrix', fontsize=16)  # Set larger title font size
plt.show()
# Adjust spacing between columns
plt.tight_layout()


# In[24]:


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
principal_components = pca.fit_transform(data_cleaned)

# Visualize the principal components
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()


# In[25]:


data_cleaned.head()


# In[26]:


import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# In[27]:


from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method   

wcss = []
for i in range(1, 11):
   
     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
     kmeans.fit(data_cleaned)
     wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[28]:


# Based on the elbow plot, choose the optimal number of clusters
optimal_num_clusters = 3

# Create the KMeans model
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)

# Fit the model to the data
kmeans.fit(data_cleaned)


# In[29]:


# Predict cluster labels for each data point
labels = kmeans.predict(data_cleaned)


# In[30]:


# Add cluster labels to the original DataFrame
data_cleaned['Cluster'] = labels


# In[31]:


# Analyze the clusters
print(data_cleaned.groupby('Cluster').mean())


# In[32]:


data_cleaned.head()


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA  # Or t-SNE

# Reduce dimensionality using PCA (or t-SNE)
pca = PCA(n_components=2)  # Adjust n_components as needed
reduced_data = pca.fit_transform(data_new.drop('Cluster', axis=1))

# Create a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data_new['Cluster'], palette='viridis')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[34]:


# Calculate evaluation metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
silhouette_avg = silhouette_score(reduced_data, labels)
calinski_harabasz_score_value = calinski_harabasz_score(reduced_data, labels)
davies_bouldin_score_value = davies_bouldin_score(reduced_data, labels)

print("Silhouette Coefficient:", silhouette_avg)
print("Calinski-Harabasz Index:", calinski_harabasz_score_value)
print("Davies-Bouldin Index:", davies_bouldin_score_value)


# In[40]:


#Deploy
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")


import streamlit as st
# Display cluster information
st.write("Cluster Information:")
st.dataframe(data_cleaned)


# In[36]:


# Visualize the clusters
if st.checkbox("Visualize Clusters"):
  # Reduce dimensionality using PCA
  pca = PCA(n_components=2)
  reduced_data = pca.fit_transform(data_cleaned.drop('Cluster', axis=1))


# In[37]:


# Create a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data_cleaned['Cluster'], palette='viridis')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
st.pyplot()


# In[38]:


# Calculate and display evaluation metrics
st.write("Evaluation Metrics:")
silhouette_avg = silhouette_score(reduced_data, labels)
calinski_harabasz_score_value = calinski_harabasz_score(reduced_data, labels)
davies_bouldin_score_value = davies_bouldin_score(reduced_data, labels)


# In[39]:


st.write("Silhouette Coefficient:", silhouette_avg)
st.write("Calinski-Harabasz Index:", calinski_harabasz_score_value)
st.write("Davies-Bouldin Index:", davies_bouldin_score_value)


# In[ ]:




