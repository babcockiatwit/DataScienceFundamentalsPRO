import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# load data
high_popularity = pd.read_csv('Data/raw/high_popularity_spotify_data.csv')
low_popularity = pd.read_csv('Data/raw/low_popularity_spotify_data.csv')

# ensure columns match all the way through
high_popularity = high_popularity.sort_index(axis=1)
low_popularity = low_popularity.sort_index(axis=1)

# concatenate the full data
low_popularity = low_popularity.iloc[1:]
low_popularity.columns = high_popularity.columns
spotify_data = pd.concat([high_popularity, low_popularity], ignore_index=True)

# keep only numbers with track name as index
track_names = spotify_data['track_name']
spotify_data_numerics = spotify_data.select_dtypes(include='number')
spotify_data_numerics.index = track_names

# impute mean for nan values
imputer = SimpleImputer(strategy='mean')
spotify_imputed = imputer.fit_transform(spotify_data_numerics)

# standardize the data
# high_pop_normal = high_pop_numerics.apply(lambda x: (x - x.mean()) / x.std())
scaler = StandardScaler()
spotify_data_normal = scaler.fit_transform(spotify_imputed)

print("Imputed and scaled shape:", spotify_data_normal.shape)

print("Numeric features:")
print(spotify_data_numerics.head(5))
print(spotify_data_numerics.tail(5))

print("Number of features")
print(spotify_data_normal.shape[1])

# Initialize PCA with 2 components
pca = PCA(n_components=spotify_data_normal.shape[1])

# Fit PCA to the scaled data
pca.fit(spotify_data_normal)

X_pca = pca.transform(spotify_data_normal)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")

print(f"Principal components: \n{pca.components_}")

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Data')
plt.show()

exit(0)
