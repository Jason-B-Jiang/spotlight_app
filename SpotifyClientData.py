import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Union
import numpy as np

################################################################################

## Track attributes chosen for tracks
## Note: renamed 'valence' to 'positivity' as I thought valence was unclear
AUDIO_FEATURES = ['acousticness', 'danceability', 'energy', 'instrumentalness',
    'loudness', 'mode', 'speechiness', 'positivity']

################################################################################

class SpotifyClientData:
    """Class for using the Spotify API to retrieve store data on a Spotify
    user's listening habits.
    """
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self._client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(client_id=client_id,
                                      client_secret=client_secret,
                                      redirect_uri=redirect_uri,
                                      scope=["user-library-read", "user-top-read"])
                        )
        
        self._client_name = self._client.current_user()['display_name']
        
        self._top_artists = [
            {'uri': artist['uri'], 'name': artist['name'],
            'popularity': artist['popularity'], 'genres': artist['genres']} \
                for artist in \
                self._client.current_user_top_artists(time_range='long_term',
                                                      limit=50)['items']
        ]

        self._top_tracks = [
            {'uri': track['uri'], 'name': track['name'], 'popularity': track['popularity'],
             'artist': track['artists'][0]['name'], 'length_ms': track['duration_ms'],
             'audio_features': self._get_audio_features(track['uri'])} for track in \
                self._client.current_user_top_tracks(time_range='long_term',
                                                    limit=50)['items']
        ]

    def _get_audio_features(self, uri: str) -> np.ndarray:
        features = self._client.audio_features([uri])[0]

        return np.array([features['acousticness'],
                         features['danceability'],
                         features['energy'],
                         features['instrumentalness'],
                         # divide loudness by -60 so it's on a scale from
                         # 0.0 - 1.0
                         features['loudness'] / -60.0,
                         float(features['mode']),
                         features['speechiness'],
                         features['valence']])

    def get_client_name(self) -> str:
        return self._client_name

    def get_top_artists(self) -> List[Dict[str, Union[str, int, List[str]]]]:
        return self._top_artists

    def get_top_tracks(self) -> List[Dict[str, Union[str, int, np.ndarray]]]:
        return self._top_tracks