import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from typing import List, Dict, Union, Optional
from collections import Counter

from SpotifyClientData import AUDIO_FEATURES

################################################################################

## Functions for getting stats from single users

def get_top_n_genres(artists: List[Dict[str, Union[str, int, List[str]]]],
                     n: int = 5):
    """Return the top n genres from a list of artists.

    If n > number of unique genres represented by artists, then just return
    all the genres sorted from highest frequency to lowest frequency.
    """
    # all genres represented by artists
    genres = list(pd.core.common.flatten([artist['genres'] for artist in artists]))

    # sort genres by how often they are represented by artists, in descending order
    genres_counted = Counter(genres)

    # sort genres in descending order from frequency they're represented by artists
    genres_sorted = sorted(list(set(genres)), key=lambda g: genres_counted[g],
                           reverse=True)

    if n > len(genres_sorted):
        return genres_sorted

    return genres_sorted[:n]


def get_top_n_artists(artists: List[Dict[str, Union[str, int, List[str]]]],
                      n: int = 5):
    """Return the top n artists from a list of artists.

    If n is greater than number of artists, just return all artists
    """
    if n > len(artists):
        n = len(artists)

    top_n = artists[:n]
    return [artist['name'] for artist in top_n]


def get_top_n_tracks(tracks: List[Dict[str, Union[str, int, np.ndarray]]],
                     n: int = 5):
    """Return the top n tracks from a list of tracks.

    If n is greater than number of tracks, just return all tracks
    """
    if n > len(tracks):
        n = len(tracks)

    top_n = tracks[:n]
    return [f"{track['artist']} - {track['name']}" for track in top_n]


def plot_top_track_popularity_distribution(tracks: List[Dict[str, Union[str, int, np.ndarray]]], \
                                           username: str, \
                                           out_dir: str = '.') -> None:
    """Create a histogram of the distribution of popularity scores for a list
    of tracks.
    
    Arguments:
        tracks: list of tracks for a user, taken from a SpotifyClientData instance

        username: name of this user

        out_dir: directory to save resulting histogram to, default is current
        directory
    """
    fig, ax = plt.subplots()
    sns.histplot(data = {'popularity': [track['popularity'] for track in tracks]},
                 x = 'popularity',
                 ax = ax)
    
    ax.set_xlim(0, 100)
    plt.xlabel('Track popularity score')
    plt.ylabel('Number of tracks')
    plt.title(f"Popularity of {username}'s top tracks")

    plt.savefig(f"{out_dir}/popularity_distn.png")


def plot_track_features(tracks: List[Dict[str, Union[str, int, np.ndarray]]],
                        username: str,
                        out_dir: str = '.') \
    -> None:
    # get average audio features for all tracks
    average_track_features = np.average(
        np.vstack([track['audio_features'] for track in tracks]), axis=0
        )

    # make dataframe for average audio feature values
    features_df = pd.DataFrame({'feature': ['acousticness',
                                            'danceability',
                                            'energy',
                                            'instrumentalness',
                                            'speechiness',
                                            'positivity'],
                                'value': [average_track_features[0],
                                          average_track_features[1],
                                          average_track_features[2],
                                          average_track_features[3],
                                          average_track_features[5],
                                          average_track_features[6]]})

    sns.barplot(data = features_df,
                x = 'feature',
                y = 'value')

    plt.xticks(rotation = 45)
    plt.xlabel('Audio feature')
    plt.ylabel('Average value per track')
    plt.title(f"{username}'s track features")

    plt.savefig(f"{out_dir}/track_features.png")

################################################################################

## Functions for comparing stats between two users

def get_most_similar_track(user_1_tracks: List[Dict[str, Union[str, int, Dict[str, float]]]],
                           user_2_tracks: List[Dict[str, Union[str, int, Dict[str, float]]]]) \
                            -> Optional[str]:
    """Return the track in user_2_tracks most similar to tracks in user_1_tracks.

    If there are ties for most similar tracks in user_2_tracks, return the one
    that appears first in user_2_tracks.

    If either user has no tracks, or one user's tracks completely overlaps with
    the other user's tracks, return None.
    """
    # TODO - check case where only 1 track for either user
    # remove all overlapping tracks between users
    user_1_tracks_unique = [track for track in user_1_tracks if track['uri'] \
        not in [track['uri'] for track in user_2_tracks]]

    user_2_tracks_unique = [track for track in user_2_tracks if track['uri'] \
        not in [track['uri'] for track in user_1_tracks]]

    if not user_1_tracks_unique or not user_2_tracks_unique:
        return

    # get audio feature vectors for unique tracks from each user
    user_1_vectors = np.vstack([track['audio_features'] for track in \
        user_1_tracks_unique])

    user_2_vectors = np.vstack([track['audio_features'] for track in \
        user_2_tracks_unique])

    # get average distance for each track from user_2 to tracks from user_1
    distances = \
        np.array([np.average(np.linalg.norm(v - user_1_vectors, axis=1)) \
            for v in user_2_vectors])

    # most similar track = track with lowest average distance to user 1 tracks
    most_similar_track = user_2_tracks_unique[np.argmin(distances)]

    return f"{most_similar_track['artist']} - {most_similar_track['name']}"


def plot_user_track_pca(user_1_tracks: List[Dict[str, Union[str, int, Dict[str, float]]]],
                        user_2_tracks: List[Dict[str, Union[str, int, Dict[str, float]]]],
                        user_1_name: str,
                        user_2_name: str,
                        out_dir: str = '.') -> None:
    """Creates and writes a PCA plot for user_1_name and user_2_name's top tracks.
    """
    # 1) make dataframes for each user's track features
    user_1_df = pd.DataFrame([track['audio_features'] for track in user_1_tracks],
                            columns=AUDIO_FEATURES)
    user_1_df['user'] = user_1_name

    user_2_df = pd.DataFrame([track['audio_features'] for track in user_2_tracks],
                            columns=AUDIO_FEATURES)
    user_2_df['user'] = user_2_name

    # 2) combine these dataframes
    all_user_df = pd.concat([user_1_df, user_2_df]).reset_index()

    # 3) standardize each audio feature for sd = 1 and mean = 0
    audio_features_standardized = StandardScaler().fit_transform(
        all_user_df.loc[:, AUDIO_FEATURES].values
    )

    # 4) perform PCA on standardized features
    pca = PCA(n_components=2)
    pca_audio_features = pca.fit_transform(audio_features_standardized)
    pca_df = pd.DataFrame(data=pca_audio_features,
                          columns=['pc1', 'pc2']).assign(
                            user=all_user_df['user']
                          )

    # 5) plot principle components 1 + 2, labelling points by which user they
    #    belong to, then save the plot to out_dir
    sns.scatterplot(data=pca_df, x='pc1', y='pc2', hue='user')
    plt.xlabel(f"PC1 ({round(pca.explained_variance_ratio_[0] * 100, 3)}%)")
    plt.ylabel(f"PC2 ({round(pca.explained_variance_ratio_[1] * 100, 3)}%)")
    plt.title(f"Similarity of top tracks for {user_1_name} and {user_2_name}")

    plt.savefig(f"{out_dir}/users_track_pca.png")

################################################################################

from credentials import *
from SpotifyClientData import SpotifyClientData, AUDIO_FEATURES

client = SpotifyClientData(CLIENT_ID, CLIENT_SECRET, APP_REDIRECT_URI)