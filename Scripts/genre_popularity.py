import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

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
genre_index = spotify_data.set_index(['playlist_genre'])
genre_index = genre_index.rename(index={"r&b": "RnB"})

num_data = spotify_data.select_dtypes(include='number')
spotify_data_numerics = num_data.set_index(genre_index.index)



def run_pca_for_all_genres(df: pd.DataFrame, numeric_columns):

    for genre in df.index.unique():
        # subset rows for this genre
        print(genre)
        df_genre = df[df.index == genre]

        if df_genre.empty or len(df_genre) < 3:
            print(f"Skipping '{genre}' — not enough samples ({len(df_genre)} rows).")
            continue

        print(f"\n===== {genre.upper()} =====")
        print(f"Samples: {len(df_genre)}")

        # run your PCA pipeline function
        pca_result = run_full_pca(df_genre, numeric_columns)

        # retrieve PCA coordinates
        X_pca = pca_result["X_pca"]
        explained = pca_result["explained_variance"]

        # print explained variance ratio
        print("Explained variance ratio:")
        for i, value in enumerate(explained[:5]):   # print first 5 components
            print(f"  PC{i+1}: {value:.4f}")
        print(f"  Cumulative (first 2 PCs): {(explained[0] + explained[1]):.4f}")

        # plot PC1 vs PC2
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        plt.title(f"PCA: {genre}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



order = (
    genre_index.groupby("playlist_genre")["track_popularity"]
    .median()
    .sort_values()
    .index
)

plt.figure(figsize=(14, 8))

sns.violinplot(
    data=genre_index,
    x="playlist_genre",
    y="track_popularity",
    order=order,
    palette="coolwarm"
)

plt.title("Popularity Distribution by Genre", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

