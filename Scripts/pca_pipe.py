import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

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

# # keep only numbers with track name as index
# indexes = spotify_data[['playlist_genre','track_name']]
# myindex = pd.MultiIndex.from_frame(indexes)
# num_data = spotify_data.select_dtypes(include='number').reset_index(drop=True)
# spotify_data_numerics =  pd.DataFrame(num_data, index=myindex)

# keep only numbers with track name as index
genre_index = spotify_data.set_index(['playlist_genre'])
genre_index = genre_index.rename(index={"r&b": "RnB"})

num_data = spotify_data.select_dtypes(include='number')
spotify_data_numerics = num_data.set_index(genre_index.index)



#print(genre_index.index.unique())

# print(myindex.head(5))
# print(spotify_data_numerics.head(5))
# print(spotify_data_numerics.loc[("pop","Taste")])


def run_full_pca(df: pd.DataFrame, numeric_columns: list):
    """
    Runs PCA on the entire dataset using only the numeric feature columns.
    Returns the PCA pipeline, transformed PCA coordinates, and diagnostics.
    """

    # --- Select numeric data only ---
    X = df[numeric_columns]

    # --- Preprocessing pipeline: impute + scale ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Full column transformer (in case you add non-numerics later)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns)
        ],
        remainder="drop"
    )

    # --- PCA (all components) ---
    #pca = PCA(n_components=len(numeric_columns))

    # --- PCA (2 components) ---
    pca = PCA(n_components=2)

    # --- Full pipeline ---
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", pca)
    ])

    # --- Fit the pipeline ---
    pipe.fit(X)

    # --- Transform data ---
    X_pca = pipe.transform(X)

    # --- Extract diagnostics ---
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    loadings = pd.DataFrame(
        pca.components_,
        columns=numeric_columns,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return {
        "pipeline": pipe,
        "X_pca": X_pca,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "loadings": loadings
    }



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


numeric_columns = [
    "energy", "tempo", "danceability", "loudness", "liveness", "valence",
    "time_signature", "speechiness", "instrumentalness", "mode", "key",
    "duration_ms", "acousticness", "track_popularity"
]

# genres = [
#     'pop', 'rock', 'jazz', 'classical', 'hip-hop', 'afrobeats', 'latin',
#     'indian', 'country', 'r"&"b', 'electronic', 'soul', 'gaming', 'j-pop',
#     'metal', 'reggae', 'k-pop', 'arabic', 'punk', 'blues', 'folk', 'lofi',
#     'brazilian', 'turkish', 'ambient', 'korean', 'world', 'indie',
#     'mandopop', 'cantopop', 'wellness', 'gospel', 'funk', 'soca', 'disco'
# ]

#run_pca_for_all_genres(genre_index, numeric_columns)

results = run_full_pca(spotify_data, numeric_columns)

X_pca = results["X_pca"]
variance = results["explained_variance"]
cumulative = results["cumulative_variance"]
loadings = results["loadings"]
pipe = results["pipeline"]




print(f"Explained variance ratio: {variance}")
print(f"Cumulative variance: {cumulative}")

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Full Dataset PCA")
plt.show()

exit(0)
